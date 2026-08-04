"""Tap-to-talk turn state machine: IDLE -> ARMED -> SPEECH -> TRAILING -> (EOU) -> IDLE.

One instance per WebSocket connection. Driven frame-by-frame by Silero VAD speech
probabilities (vad/silero.py). Pure logic — no audio, no I/O, no config import — so the
whole thing is unit-tested from synthetic probability sequences (see _selftest()).

Timing is measured in *audio frames*, not wall clock: VAD frames are a fixed 32 ms
(512 samples @ 16 kHz), so elapsed = frames * 32 ms is exact and deterministic. That is
also the right clock for endpointing — we care about audio silence, not scheduler jitter.

State semantics (barge-in is out of scope, but the shape is left open for it — which is
why VAD runs during IDLE too):
  IDLE     - VAD still runs every frame (future barge-in) but onset does NOT start a turn.
             push() returns NONE regardless of probability.
  ARMED    - entered via arm() on the client 'turn_start'. Waiting for confirmed onset;
             returns to IDLE after vad_arm_timeout_s of no speech.
  SPEECH   - onset confirmed (vad_onset_frames consecutive frames >= speech_threshold).
             Caller feeds audio to STT. Hard cap at vad_max_utterance_s forces finalize.
  TRAILING - probability dropped below silence_threshold (hysteresis). A hangover timer
             runs; speech resuming (>= speech_threshold) before expiry returns to SPEECH
             with NO EOU. Expiry emits EOU and returns to IDLE.

Clinical safety: every terminal/timeout path returns to IDLE (a safe state). The caller
logs ARM_TIMEOUT loudly. Audio is never silently dropped here — the machine only decides
turn boundaries; the caller owns the audio buffer.
"""
from dataclasses import dataclass
from enum import Enum

FRAME_MS = 32.0   # 512 samples @ 16 kHz — the one Silero frame size we accept


class State(Enum):
    IDLE = "IDLE"
    ARMED = "ARMED"
    SPEECH = "SPEECH"
    TRAILING = "TRAILING"


class Event(Enum):
    NONE = "none"
    ONSET = "onset"              # SPEECH entered: start feeding STT (+ flush prespeech buffer)
    EOU = "eou"                  # end of utterance: finalize STT, run the reply
    ARM_TIMEOUT = "arm_timeout"  # ARMED expired with no speech: back to IDLE


@dataclass
class TurnParams:
    """All runtime-configurable (config.vad_params reads env overrides onto these).
    vad_hangover_ms is the primary latency knob — tuned against real post-op patients who
    pause mid-sentence more than general users — so it must never be hardcoded downstream."""
    vad_speech_threshold: float = 0.5
    vad_silence_threshold: float = 0.35     # hysteresis: prevents flapping mid-word
    vad_onset_frames: int = 2               # ~64 ms debounce: rejects clicks / door sounds
    vad_hangover_ms: int = 600              # trailing silence before EOU (primary latency knob)
    vad_prespeech_ms: int = 300             # ring-buffer depth — used by the Part-3 ring buffer,
    #                                         not by this machine; carried here so all turn-taking
    #                                         params live in one config surface.
    vad_max_utterance_s: float = 30.0       # hard cap: force finalize
    vad_arm_timeout_s: float = 10.0         # ARMED with no speech -> IDLE
    frame_ms: float = FRAME_MS


class TurnMachine:
    def __init__(self, params: TurnParams = None):
        self.p = params or TurnParams()
        # Metrics for the last completed turn, read by the Part-4 instrumentation.
        self.last_speech_duration_ms = 0.0
        self.last_hangover_used_ms = 0.0
        self.reset()

    def reset(self):
        self.state = State.IDLE
        self._onset_count = 0      # consecutive >= speech_threshold frames while ARMED
        self._armed_frames = 0
        self._speech_frames = 0    # frames since onset (SPEECH+TRAILING) -> utterance length
        self._trailing_frames = 0

    def arm(self) -> Event:
        """Client 'turn_start': IDLE -> ARMED. No-op if a turn is already in flight."""
        if self.state is State.IDLE:
            self.state = State.ARMED
            self._onset_count = 0
            self._armed_frames = 0
        return Event.NONE

    def _ms(self, frames):
        return frames * self.p.frame_ms

    def push(self, prob: float) -> Event:
        """Advance one 32 ms VAD frame with speech probability `prob` in [0,1]. Returns the
        event triggered this frame (Event.NONE most frames)."""
        st = self.state

        if st is State.IDLE:
            return Event.NONE                       # VAD runs upstream; onset does not arm a turn

        if st is State.ARMED:
            self._armed_frames += 1
            if prob >= self.p.vad_speech_threshold:
                self._onset_count += 1
                if self._onset_count >= self.p.vad_onset_frames:
                    self.state = State.SPEECH
                    self._speech_frames = 0
                    self._trailing_frames = 0
                    return Event.ONSET
            else:
                self._onset_count = 0               # debounce: a lone spike is rejected
            if self._ms(self._armed_frames) >= self.p.vad_arm_timeout_s * 1000:
                self.reset()                        # safe state; caller logs the timeout
                return Event.ARM_TIMEOUT
            return Event.NONE

        if st is State.SPEECH:
            self._speech_frames += 1
            if prob < self.p.vad_silence_threshold:
                self.state = State.TRAILING         # hysteresis: only a clear drop starts trailing
                self._trailing_frames = 0
            if self._ms(self._speech_frames) >= self.p.vad_max_utterance_s * 1000:
                return self._emit_eou()             # hard cap forces finalize
            return Event.NONE

        if st is State.TRAILING:
            self._speech_frames += 1
            if prob >= self.p.vad_speech_threshold:
                self.state = State.SPEECH           # speech resumed before hangover -> no EOU
                self._trailing_frames = 0
                return Event.NONE
            self._trailing_frames += 1              # prob between silence & speech: keep trailing
            if self._ms(self._speech_frames) >= self.p.vad_max_utterance_s * 1000:
                return self._emit_eou()
            if self._ms(self._trailing_frames) >= self.p.vad_hangover_ms:
                return self._emit_eou()
            return Event.NONE

        return Event.NONE

    def _emit_eou(self) -> Event:
        self.last_hangover_used_ms = self._ms(self._trailing_frames)
        self.last_speech_duration_ms = self._ms(self._speech_frames - self._trailing_frames)
        self.reset()
        return Event.EOU


# --- self-check: synthetic probability sequences, no audio/onnxruntime (repo convention) ---
def _drive(m, probs):
    """Push a list of probabilities, return the list of non-NONE events (in order)."""
    return [e for e in (m.push(p) for p in probs) if e is not Event.NONE]


def _selftest():
    P = TurnParams  # small timers so the tests stay short (frame = 32 ms)

    # 1) onset debounce: a single-frame spike must NOT start a turn.
    m = TurnMachine(P(vad_onset_frames=2)); m.arm()
    assert _drive(m, [0.9, 0.1, 0.1]) == [], "single-frame spike wrongly triggered onset"
    assert m.state is State.ARMED
    # two consecutive speech frames -> onset
    assert _drive(m, [0.9, 0.9]) == [Event.ONSET]
    assert m.state is State.SPEECH

    # 2) hysteresis: prob hovering near 0.5 (between silence .35 and speech .5) must not flap.
    m = TurnMachine(P()); m.arm(); _drive(m, [0.9, 0.9])          # -> SPEECH
    assert _drive(m, [0.45, 0.4, 0.45, 0.4]) == [], "hover between thresholds flapped"
    assert m.state is State.SPEECH
    # and while TRAILING, a hover value (0.45) must not resume SPEECH (needs >= 0.5)
    _drive(m, [0.2])                                              # clear drop -> TRAILING
    assert m.state is State.TRAILING
    assert _drive(m, [0.45]) == [] and m.state is State.TRAILING

    # 3) mid-utterance pause shorter than hangover: no EOU, back to SPEECH on resume.
    m = TurnMachine(P(vad_hangover_ms=600)); m.arm(); _drive(m, [0.9, 0.9])
    pause = [0.1] * 10                                            # 10*32=320 ms < 600 ms hangover
    assert _drive(m, pause) == [], "short pause wrongly emitted EOU"
    assert m.state is State.TRAILING
    assert _drive(m, [0.9]) == [] and m.state is State.SPEECH     # resumed, no EOU

    # ... but a pause past the hangover DOES emit exactly one EOU and returns to IDLE.
    m = TurnMachine(P(vad_hangover_ms=600)); m.arm(); _drive(m, [0.9, 0.9])
    long_pause = [0.1] * 25                                       # 25*32=800 ms > 600 ms
    assert _drive(m, long_pause) == [Event.EOU]
    assert m.state is State.IDLE
    assert 550 <= m.last_hangover_used_ms <= 650                  # ~600 ms tail (frame-granular)

    # 4) max-utterance cap forces finalize even with continuous speech.
    m = TurnMachine(P(vad_max_utterance_s=0.1)); m.arm(); _drive(m, [0.9, 0.9])  # onset (2 frames)
    # after onset, _speech_frames=0; need >=100 ms => >=4 frames of continued speech
    evs = _drive(m, [0.9, 0.9, 0.9, 0.9])
    assert evs == [Event.EOU], evs
    assert m.state is State.IDLE

    # 5) arm timeout with no speech returns to IDLE.
    m = TurnMachine(P(vad_arm_timeout_s=0.1)); m.arm()            # 0.1 s => 4 frames
    evs = _drive(m, [0.1, 0.1, 0.1, 0.1])
    assert evs == [Event.ARM_TIMEOUT], evs
    assert m.state is State.IDLE

    # 6) IDLE ignores speech: onset probability before arm() must not start a turn.
    m = TurnMachine(P())
    assert _drive(m, [0.9, 0.9, 0.9]) == [] and m.state is State.IDLE

    print("turn selftest OK")


if __name__ == "__main__":
    _selftest()
