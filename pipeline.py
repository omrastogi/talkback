"""Offline cascade: wav -> STT -> LLM -> TTS -> wav. Each stage is an isolated subprocess
so the OS reclaims VRAM between stages (never two models resident at once)."""
import argparse
import difflib
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TIMING_RE = re.compile(r"TIMING stage=(\S+) load=([\d.]+) infer=([\d.]+)")


def stage(name):
    return os.path.join(HERE, name)


def run_stage(argv):
    """Run a stage script in a fresh subprocess. Echo its stdout, fail loudly on nonzero exit.
    Returns the parsed (name, load, infer) TIMING tuple, or None if absent."""
    proc = subprocess.run([sys.executable] + argv, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stdout.flush()
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        print(f"\nSTAGE FAILED (exit {proc.returncode}): {' '.join(argv)}", file=sys.stderr)
        sys.exit(proc.returncode)
    timing = None
    for line in proc.stdout.splitlines():
        m = TIMING_RE.search(line)
        if m:
            timing = (m.group(1), float(m.group(2)), float(m.group(3)))
    return timing


def print_table(timings):
    print(f"\n{'stage':<8} {'load(s)':>10} {'infer(s)':>10}")
    total = 0.0
    for name, load, infer in timings:
        print(f"{name:<8} {load:>10.2f} {infer:>10.2f}")
        total += infer
    print(f"{'TOTAL':<8} {'':>10} {total:>10.2f}")


def normalize(s):
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def cascade(wav, out_wav, backend=None, model=None):
    llm_argv = [stage("llm.py"), "--in", "out/transcript.txt", "--out", "out/reply.txt"]
    if backend:
        llm_argv += ["--backend", backend]
    if model:
        llm_argv += ["--model", model]
    timings = [
        run_stage([stage("stt.py"), wav, "--out", "out/transcript.txt"]),
        run_stage(llm_argv),
        run_stage([stage("tts.py"), "--in", "out/reply.txt", "--out", out_wav]),
    ]
    print_table(timings)


def selftest():
    ref = "the quick brown fox jumps over the lazy dog"
    run_stage([stage("tts.py"), "--text", ref, "--out", "out/probe.wav"])
    run_stage([stage("stt.py"), "out/probe.wav", "--out", "out/probe.txt"])
    with open("out/probe.txt", encoding="utf-8") as f:
        hyp = f.read()
    ratio = difflib.SequenceMatcher(None, normalize(ref), normalize(hyp)).ratio()
    print(f"\ntranscript: {hyp!r}")
    print(f"ratio: {ratio:.3f}")
    print("PASS" if ratio >= 0.6 else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--wav")
    g.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="out/reply.wav")
    ap.add_argument("--backend", choices=["openai", "parcs"], default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    os.makedirs("out", exist_ok=True)
    if args.selftest:
        selftest()
    else:
        cascade(args.wav, args.out, args.backend, args.model)


if __name__ == "__main__":
    main()
