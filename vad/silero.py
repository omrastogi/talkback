"""Silero VAD wrapped via onnxruntime. 16 kHz, 512-sample (32 ms) frames ONLY.

Stateful (RNN hidden state) -> ONE instance per connection, with reset() between turns.
No module-level singleton on purpose: two connections sharing a session would corrupt each
other's hidden state. The provider list is passed in explicitly (vad/provider.py picks it
once at startup) — this class never guesses a backend.

Frame contract: exactly 512 float32 samples in [-1, 1] at 16 kHz. We ASSERT on mismatch
rather than resampling, because a silent resample here would desync the whole endpointer
(hangover timing assumes 32 ms frames). Re-chunk upstream instead.

Silero v5 signature (verified against the shipped model): inputs {input:(1,N) f32,
state:(2,1,128) f32, sr:int64}, outputs (prob, new_state). Crucially, v5 prepends a
64-sample CONTEXT (the tail of the previous frame) to each 512-sample frame, so the model
actually sees 576 samples per call; feeding a bare 512 runs without error but returns
garbage (~0.001). We carry that context internally — the public frame contract stays 512.
Older v4 models split the state into h/c and have no context; adjust reset()/__call__ then.
"""
import os

import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512   # 32 ms @ 16 kHz — Silero's fixed 16 kHz frame
CONTEXT = 64          # v5 prepends 64 samples of previous audio to each 16 kHz frame
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "silero_vad.onnx")

_FETCH_HINT = ("silero_vad.onnx not found at {path}. Fetch it once:\n"
               "  curl -L -o {path} https://github.com/snakers4/silero-vad/raw/master/"
               "src/silero_vad/data/silero_vad.onnx")


class SileroVAD:
    def __init__(self, providers, model_path=DEFAULT_MODEL, sample_rate=SAMPLE_RATE):
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"SileroVAD is 16 kHz only, got {sample_rate}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(_FETCH_HINT.format(path=model_path))
        import onnxruntime as ort
        so = ort.SessionOptions()
        # Pin ORT to one thread each: the default pools oversubscribe cores under
        # concurrent-session load testing and make latency misattribution likely.
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=so, providers=list(providers))
        self.sample_rate = sample_rate
        self._sr = np.array(sample_rate, dtype=np.int64)
        self.reset()

    def reset(self):
        """Clear the RNN hidden state AND the 64-sample context. Call once per connection AND
        between turns, or the previous utterance's tail leaks into the next turn's decision."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)

    def __call__(self, frame) -> float:
        """One 512-sample float32 frame -> speech probability in [0, 1]."""
        frame = np.asarray(frame, dtype=np.float32)
        assert frame.shape == (FRAME_SAMPLES,), (
            f"SileroVAD expects exactly {FRAME_SAMPLES}-sample frames, got {frame.shape}; "
            "re-chunk upstream, do not resample here")
        # v5: prepend the previous frame's 64-sample tail, then keep this frame's tail as the
        # next context. Without this the model returns ~0 for everything.
        x = np.concatenate([self._context, frame.reshape(1, -1)], axis=1)
        out, self._state = self.session.run(
            None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[:, -CONTEXT:]
        return float(np.asarray(out).reshape(-1)[0])
