"""Startup VAD provider selection — run ONCE at process start, never per connection.

Micro-benchmarks Silero on CUDA vs CPU (200 synthetic frames, first 20 discarded as
warmup, median + p95) and picks the lower median. A CUDA probe failure of ANY kind
(no CUDA, OOM, missing ORT GPU build, EP registration failure) is caught, logged at
WARNING, and falls back to CPU. This must never be fatal. VAD_PROVIDER_OVERRIDE forces a
backend for benchmarking.

Why CPU is expected to win here: the model is ~2 MB and the frame budget is 32 ms, so
kernel-launch overhead and PCIe transfer dominate a GPU run, while CPU inference is a few
hundred microseconds AND avoids contending with Parakeet for the (tight) 4 GB VRAM. The
benchmark is kept anyway so the coming 5090 can flip the decision on evidence, not a guess.

The returned `bench` record is written into log/bench/*.jsonl at startup (by the server) so
VAD placement is visible in the latency test matrix.
"""
import logging
import time

import numpy as np

log = logging.getLogger("voice")

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"
N_FRAMES = 200
WARMUP = 20
FRAME_SAMPLES = 512


def _percentile(xs, p):
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _bench(provider, make_vad):
    """Build a SileroVAD on `provider`, time N_FRAMES of synthetic audio, return
    (median_ms, p95_ms). Raises if the provider can't be built or run."""
    vad = make_vad([provider])
    rng = np.random.default_rng(0)
    lat = []
    for i in range(N_FRAMES):
        frame = (rng.standard_normal(FRAME_SAMPLES).astype("float32") * 0.1)
        t = time.perf_counter()
        vad(frame)
        if i >= WARMUP:
            lat.append((time.perf_counter() - t) * 1000.0)
    return _percentile(lat, 50), _percentile(lat, 95)


def select_provider(override="auto", make_vad=None, available=None):
    """Pick the ORT provider list for SileroVAD. Returns (providers, bench_record).

    override : 'auto' | 'cuda' | 'cpu' (config VAD_PROVIDER_OVERRIDE).
    make_vad : make_vad(providers) -> callable(frame)->prob. Injectable for tests; defaults
               to constructing a real SileroVAD.
    available: ORT provider names; defaults to onnxruntime.get_available_providers().

    Never raises on a CUDA failure — logs WARNING and falls back to CPU.
    """
    if make_vad is None:
        from vad.silero import SileroVAD
        make_vad = lambda provs: SileroVAD(providers=provs)
    if available is None:
        import onnxruntime as ort
        available = ort.get_available_providers()

    bench = {"vad_provider": None, "vad_override": override,
             "vad_cpu_median_ms": None, "vad_cpu_p95_ms": None,
             "vad_cuda_median_ms": None, "vad_cuda_p95_ms": None,
             "vad_cuda_error": None}

    if override == "cpu":
        bench["vad_provider"] = CPU
        log.info("VAD provider forced to CPU (VAD_PROVIDER_OVERRIDE=cpu)")
        return [CPU], bench
    if override == "cuda":
        bench["vad_provider"] = CUDA
        log.info("VAD provider forced to CUDA (VAD_PROVIDER_OVERRIDE=cuda)")
        return [CUDA, CPU], bench                 # keep CPU in the list as ORT's own fallback

    # auto: CPU is always benchable; CUDA only if the EP is present, and never fatally.
    cpu_med, cpu_p95 = _bench(CPU, make_vad)
    bench["vad_cpu_median_ms"], bench["vad_cpu_p95_ms"] = cpu_med, cpu_p95

    cuda_med = None
    if CUDA in available:
        try:
            cuda_med, cuda_p95 = _bench(CUDA, make_vad)
            bench["vad_cuda_median_ms"], bench["vad_cuda_p95_ms"] = cuda_med, cuda_p95
        except Exception as e:                     # OOM / EP registration / missing GPU build...
            bench["vad_cuda_error"] = f"{type(e).__name__}: {e}"
            log.warning("VAD CUDA probe failed, falling back to CPU: %s", e)
    else:
        log.info("VAD: CUDAExecutionProvider not in this ORT build; using CPU")

    if cuda_med is not None and cuda_med < cpu_med:
        bench["vad_provider"] = CUDA
        margin = (cpu_med - cuda_med) / cpu_med * 100 if cpu_med else 0.0
        log.info("VAD provider=CUDA: median %.3f ms vs CPU %.3f ms (%.0f%% faster)",
                 cuda_med, cpu_med, margin)
        return [CUDA, CPU], bench

    bench["vad_provider"] = CPU
    if cuda_med is not None:
        margin = (cuda_med - cpu_med) / cuda_med * 100 if cuda_med else 0.0
        log.info("VAD provider=CPU: median %.3f ms vs CUDA %.3f ms (%.0f%% faster)",
                 cpu_med, cuda_med, margin)
    else:
        log.info("VAD provider=CPU: median %.3f ms (no CUDA comparison)", cpu_med)
    return [CPU], bench


# --- self-check: mocked probe, no onnxruntime / no model (repo convention) ---
def _selftest():
    class FakeVAD:
        """CUDA build 'fails' (EP registration error); CPU build works and returns a prob."""
        def __init__(self, providers):
            if CUDA in providers:
                raise RuntimeError("simulated CUDA EP registration failure")
        def __call__(self, frame):
            return 0.0

    make = lambda provs: FakeVAD(provs)

    # auto + CUDA advertised but probe raises -> CPU fallback, no exception, error recorded.
    provs, b = select_provider("auto", make_vad=make, available=[CPU, CUDA])
    assert provs == [CPU], provs
    assert b["vad_provider"] == CPU
    assert b["vad_cuda_error"] and "CUDA" in b["vad_cuda_error"], b
    assert b["vad_cpu_median_ms"] is not None

    # override=cpu short-circuits before any benchmark.
    provs, b = select_provider("cpu", make_vad=make, available=[CPU, CUDA])
    assert provs == [CPU] and b["vad_provider"] == CPU and b["vad_cpu_median_ms"] is None

    # override=cuda returns CUDA (with CPU fallback in the list).
    provs, b = select_provider("cuda", make_vad=make, available=[CPU, CUDA])
    assert provs == [CUDA, CPU] and b["vad_provider"] == CUDA

    # auto with no CUDA EP present -> CPU, no error flag.
    provs, b = select_provider("auto", make_vad=make, available=[CPU])
    assert provs == [CPU] and b["vad_cuda_error"] is None

    # auto where CUDA is genuinely faster -> CUDA chosen (inject fast/slow fakes).
    class TimedVAD:
        def __init__(self, providers):
            self.delay = 0.0 if CUDA in providers else 0.001  # CUDA "faster"
        def __call__(self, frame):
            if self.delay:
                time.sleep(self.delay)
            return 0.0
    provs, b = select_provider("auto", make_vad=lambda p: TimedVAD(p), available=[CPU, CUDA])
    assert provs == [CUDA, CPU] and b["vad_provider"] == CUDA, b

    print("provider selftest OK")


if __name__ == "__main__":
    _selftest()
