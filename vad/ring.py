"""Prespeech ring buffer: the last vad_prespeech_ms of raw 16 kHz PCM, always on.

Appended on EVERY inbound frame regardless of turn state, so when speech onset is confirmed
the audio that was already in flight (the clipped first syllable the VAD needed a few frames
to detect) is still available. On onset the whole buffer is flushed into the STT feed *before*
the current frame — see the wiring in server.py.

Only works if the client mic streams continuously (Part 5). Pure data structure, no I/O, so
it is unit-tested directly (see _selftest()).
"""
from collections import deque

import numpy as np

SAMPLE_RATE = 16000


class PrespeechRing:
    def __init__(self, prespeech_ms, sample_rate=SAMPLE_RATE):
        # capacity in samples; deque(maxlen) drops the oldest sample automatically so the
        # buffer holds exactly the most recent prespeech_ms with no manual trimming.
        self.capacity = max(0, int(round(prespeech_ms / 1000.0 * sample_rate)))
        self._buf = deque(maxlen=self.capacity)

    def append(self, frame):
        """Append one PCM frame (float32, any length). Oldest samples fall off the back."""
        self._buf.extend(np.asarray(frame, dtype=np.float32).reshape(-1).tolist())

    def flush(self):
        """Return the buffered samples (oldest->newest) and clear. Call once at onset."""
        out = np.array(self._buf, dtype=np.float32)
        self._buf.clear()
        return out

    def clear(self):
        self._buf.clear()


def _selftest():
    # capacity math: 300 ms @ 16 kHz = 4800 samples
    assert PrespeechRing(300, 16000).capacity == 4800

    # order preserved, onset frame appears exactly once (no duplication), flush clears.
    r = PrespeechRing(1000, 100)                       # cap = 100 samples (roomy)
    r.append(np.array([1, 2, 3], dtype=np.float32))
    r.append(np.array([4, 5], dtype=np.float32))
    r.append(np.array([6, 7], dtype=np.float32))       # the "onset" frame
    assert list(r.flush()) == [1, 2, 3, 4, 5, 6, 7]    # in order, each sample once
    assert list(r.flush()) == []                       # flush cleared the buffer

    # capacity trims the oldest, keeps the most recent `capacity` samples in order.
    r = PrespeechRing(40, 100)                          # cap = 4 samples
    assert r.capacity == 4
    r.append(np.array([1, 2, 3], dtype=np.float32))
    r.append(np.array([4, 5, 6], dtype=np.float32))    # 6 samples in, only last 4 kept
    assert list(r.flush()) == [3, 4, 5, 6]

    # zero-depth ring is a no-op (never raises), for prespeech_ms=0.
    r = PrespeechRing(0, 16000)
    r.append(np.array([1, 2, 3], dtype=np.float32))
    assert list(r.flush()) == []

    print("ring selftest OK")


if __name__ == "__main__":
    _selftest()
