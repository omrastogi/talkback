"""Per-connection tap-to-talk ingest: raw PCM frames in -> turn boundaries + utterance out.

Ties together SileroVAD (silero.py), the TurnMachine (turn.py) and the prespeech ring
(ring.py). The server drives it: feed every inbound mic frame to push_pcm(); it resamples to
16 kHz, chunks into fixed 512-sample VAD frames, runs the VAD + state machine, and keeps the
ring topped up. On confirmed onset it flushes the ring (prespeech) into the utterance so the
clipped first syllable is preserved; on EOU it exposes the whole 16 kHz utterance for the
server to transcribe. STT/LLM/TTS are untouched — this only decides *when* a turn starts/ends
and *what* audio it contains.

Ordering guarantee (the subtle bit): the ring is appended one 512-frame at a time, right
before that frame is classified, so an onset flush contains prespeech + the onset frame and
never a future frame. The onset frame is therefore emitted exactly once (verified in
_selftest with frame-indexed audio).
"""
import os
import sys

if __package__ in (None, ""):          # allow `python vad/ingest.py` (repo runs selftests direct)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from vad.ring import PrespeechRing
from vad.turn import Event, TurnMachine

VAD_FRAME = 512
SR = 16000


class VadIngest:
    def __init__(self, vad, params, sample_rate_in):
        self.vad = vad
        self.machine = TurnMachine(params)
        self.ring = PrespeechRing(params.vad_prespeech_ms, SR)
        self.sr_in = sample_rate_in
        self._resid = np.zeros(0, dtype=np.float32)   # <512 leftover between calls
        self._utter = []                              # accumulating 16k utterance frames
        self._capturing = False
        self.final_utterance = None                   # set on EOU; server take_final()s it

    def arm(self):
        """Client 'turn_start': reset VAD hidden state + machine for a fresh turn, but KEEP the
        prespeech ring (and mid-frame residual) so audio captured just before the tap survives
        as prespeech. Full-connection reset() clears the ring instead."""
        self.vad.reset()
        self.machine.reset()
        self._utter = []
        self._capturing = False
        self.final_utterance = None
        self.machine.arm()

    def reset(self):
        self.vad.reset()
        self.machine.reset()
        self.ring.clear()
        self._resid = np.zeros(0, dtype=np.float32)
        self._utter = []
        self._capturing = False
        self.final_utterance = None

    def _to_16k(self, pcm):
        if self.sr_in == SR:
            return pcm
        import torch
        import torchaudio
        # ponytail: per-frame resample has tiny edge artifacts at frame joins; fine for VAD,
        # and the client requests 16k so this path is the rare fallback. Resample the whole
        # utterance properly at STT time (server already does), not here.
        return torchaudio.functional.resample(torch.from_numpy(pcm), self.sr_in, SR).numpy()

    def push_pcm(self, pcm_float):
        """Feed one inbound PCM frame (float32 @ sr_in). Returns a list of per-VAD-frame
        (Event, prob, State) signals for the server to act on and render. On Event.EOU,
        self.final_utterance holds the 16k utterance (prespeech+speech+trailing)."""
        samples = self._to_16k(np.asarray(pcm_float, dtype=np.float32).reshape(-1))
        self._resid = np.concatenate([self._resid, samples])
        signals, off, n = [], 0, len(self._resid)
        while off + VAD_FRAME <= n:
            frame = self._resid[off:off + VAD_FRAME]
            off += VAD_FRAME
            self.ring.append(frame)                   # prespeech: every frame, every state
            prob = self.vad(frame)
            ev = self.machine.push(prob)
            if ev is Event.ONSET:
                self._capturing = True
                self._utter = [self.ring.flush()]     # prespeech + onset frame, exactly once
            elif ev is Event.EOU:
                self._capturing = False
                self.final_utterance = (np.concatenate(self._utter) if self._utter
                                        else np.zeros(1, dtype=np.float32))
                self._utter = []
            elif self._capturing:
                self._utter.append(frame)
            signals.append((ev, prob, self.machine.state))
        self._resid = self._resid[off:]
        return signals

    def take_final(self):
        """Pop the finalized utterance audio (or None). Never returns it twice."""
        f, self.final_utterance = self.final_utterance, None
        return f


def _selftest():
    from vad.turn import Event as E
    from vad.turn import State, TurnParams

    class FakeVad:
        def __init__(self, probs):
            self.probs, self.i = list(probs), 0
        def __call__(self, frame):
            p = self.probs[self.i]; self.i += 1; return p
        def reset(self):
            self.i = 0

    def frame(v):                                     # 512 samples all == v, to trace ordering
        return np.full(VAD_FRAME, float(v), dtype=np.float32)

    # onset after 2 speech frames; prespeech = 2 frames (64ms); hangover = 64ms (2 frames).
    p = TurnParams(vad_onset_frames=2, vad_hangover_ms=64, vad_prespeech_ms=64)
    probs = [0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1]       # frames 0..6
    ig = VadIngest(FakeVad(probs), p, 16000)
    ig.arm()
    events = []
    for i in range(7):
        for ev, prob, st in ig.push_pcm(frame(i)):
            events.append(ev)

    assert E.ONSET in events and E.EOU in events, events
    final = ig.take_final()
    assert final is not None
    seg_vals = [int(final[i * VAD_FRAME]) for i in range(len(final) // VAD_FRAME)]
    # frame0 rolled off the 2-frame ring; prespeech frame1 + onset frame2 (once) + speech 3,4,5;
    # the EOU-triggering frame6 (silence) is not included.
    assert seg_vals == [1, 2, 3, 4, 5], seg_vals
    assert len(final) == 5 * VAD_FRAME
    assert ig.take_final() is None                    # not returned twice
    assert ig.machine.state is State.IDLE             # back to safe state after EOU

    print("ingest selftest OK")


if __name__ == "__main__":
    _selftest()
