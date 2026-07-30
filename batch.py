"""Batch profiler: run a set of audio inputs through STT -> LLM (OpenAI) -> TTS,
save the replies, and write a Markdown profile of the timings.

Two input modes:
  --in-dir DIR : run every audio file in DIR (any format; ffmpeg-decoded to 16 kHz mono)
  (default)    : synthesize the built-in prompts via TTS and run those

Loads STT (Parakeet ~2.4 GB) + TTS (Kokoro 82 M) ONCE — they fit in 4 GB together with the
LLM off-GPU via OpenAI — so one-time load is separated from per-input inference. A warmup
inference per model (--warmup, default on) moves first-call cuDNN/kernel JIT out of input #1.
This is the profiling tool; pipeline.py keeps the strict per-stage subprocess isolation.
"""
import argparse
import glob
import os
import re
import subprocess
import time
from datetime import datetime

import numpy as np
import soundfile as sf
import torch

import config
import llm
import stt
import tts

# Built-in prompts (default mode), spoken via TTS to create audio inputs.
UTTERANCES = [
    "what is the capital of japan?",
    "how many days are in a leap year?",
    "give me one quick tip for staying focused while working.",
    "what is two plus two?",
    "name a planet in our solar system.",
]
AUDIO_EXTS = (".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".webm")


def natkey(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def decode_16k(src, dst):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "16000", dst],
        check=True,
    )
    return dst


def audio_seconds(wav):
    info = sf.info(wav)
    return info.frames / info.samplerate


def warmup_stt(model):
    """One throwaway transcribe so the first real input doesn't pay cuDNN/kernel JIT."""
    p = "out/_warm.wav"
    sf.write(p, (np.random.randn(16000).astype("float32") * 0.01), 16000)  # 1 s quiet noise @16k
    stt.transcribe(model, p, "out/_warm_16k.wav")
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def f(x):
    return f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", help="run every audio file in this dir (else synthesize prompts)")
    ap.add_argument("--backend", choices=["openai", "parcs"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--report", default="out/PROFILE.md")
    ap.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True,
                    help="run one discarded inference per model so input #1 isn't cold")
    args = ap.parse_args()
    os.makedirs("out", exist_ok=True)

    wall0 = time.time()

    # Kokoro loaded once (reply synthesis; also input synthesis in default mode).
    t = time.time(); pipe = tts.load_pipe(); tts_load = time.time() - t
    warm_tts = 0.0
    if args.warmup:
        t = time.time(); tts.synth(pipe, "warming up the model"); warm_tts = time.time() - t

    # ---- gather inputs: names (for filenames), sources (for display), 16 kHz wavs ----
    names, sources, wavs = [], [], []
    t = time.time()
    if args.in_dir:
        files = sorted(
            (p for p in glob.glob(os.path.join(args.in_dir, "*")) if p.lower().endswith(AUDIO_EXTS)),
            key=lambda p: natkey(os.path.basename(p)),
        )
        if not files:
            raise SystemExit(f"no audio files in {args.in_dir}")
        for src in files:
            stem = os.path.splitext(os.path.basename(src))[0]
            dst = f"out/_norm_{stem}.wav"
            decode_16k(src, dst)
            names.append(stem); sources.append(os.path.basename(src)); wavs.append(dst)
        mode = f"files from {args.in_dir}/"
    else:
        for i, u in enumerate(UTTERANCES):
            p = f"out/in_{i:02d}.wav"; sf.write(p, tts.synth(pipe, u), 24000)
            names.append(f"{i:02d}"); sources.append(f'(synth) "{u}"'); wavs.append(p)
        mode = "synthesized prompts"
    prep = time.time() - t
    durations = [audio_seconds(w) for w in wavs]

    # ---- STT (load Parakeet once) ----
    t = time.time(); model = stt.load_model(); stt_load = time.time() - t
    warm_stt = 0.0
    if args.warmup:
        t = time.time(); warmup_stt(model); warm_stt = time.time() - t
    transcripts, stt_infer = [], []
    for i, w in enumerate(wavs):
        t = time.time(); txt = stt.transcribe(model, w, f"out/_16k_{i:02d}.wav")
        stt_infer.append(time.time() - t); transcripts.append(txt)

    # ---- LLM (off-GPU; backend openai or parcs gemma gateway) ----
    t = time.time(); client, model = config.make_client(args.backend, args.model); llm_load = time.time() - t
    warm_llm = 0.0
    if args.warmup:
        t = time.time(); llm.reply(client, "hi", model); warm_llm = time.time() - t
    replies, llm_infer = [], []
    for txt in transcripts:
        t = time.time(); r = llm.reply(client, txt, model)
        llm_infer.append(time.time() - t); replies.append(r)

    # ---- TTS the replies (reuse the loaded Kokoro pipe) ----
    tts_infer, reply_paths = [], []
    for i, r in enumerate(replies):
        t = time.time(); audio = tts.synth(pipe, r); tts_infer.append(time.time() - t)
        rp = f"out/reply_{names[i]}.wav"; sf.write(rp, audio, 24000); reply_paths.append(rp)

    wall = time.time() - wall0

    # ---- report ----
    n = len(wavs)
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    tot = lambda a: sum(a)
    cascade = tot(stt_infer) + tot(llm_infer) + tot(tts_infer)
    stt_rtf = [stt_infer[i] / durations[i] for i in range(n)]

    L = []
    L.append(f"# Cascade profile — {datetime.now():%Y-%m-%d %H:%M}")
    L.append("")
    L.append(f"- Inputs: **{n}** ({mode})")
    backend = args.backend or os.environ.get("LLM_BACKEND", "parcs")
    L.append(f"- Stack: STT Parakeet-tdt-0.6b-v3 (GPU: {device}) · LLM `{model}` ({backend}) · TTS Kokoro-82M")
    L.append(f"- Model load (once each): stt **{f(stt_load)}s** · tts **{f(tts_load)}s** · openai-client **{f(llm_load)}s**")
    if args.warmup:
        L.append(f"- Warmup (discarded, so input #1 isn't cold): stt **{f(warm_stt)}s** · tts **{f(warm_tts)}s** · llm **{f(warm_llm)}s**")
    L.append("")
    L.append("## Transcripts & replies")
    for i in range(n):
        L.append("")
        L.append(f"### {names[i]}  ({f(durations[i])}s audio)")
        L.append(f"- source: {sources[i]}")
        L.append(f"- heard (STT): {transcripts[i]!r}")
        L.append(f"- reply (LLM): {replies[i]!r}")
        L.append(f"- reply audio: `{reply_paths[i]}`")
        L.append(f"- infer: stt={f(stt_infer[i])}s  llm={f(llm_infer[i])}s  tts={f(tts_infer[i])}s  "
                 f"cascade={f(stt_infer[i] + llm_infer[i] + tts_infer[i])}s  (stt RTF {stt_rtf[i]:.2f})")
    L.append("")
    L.append(f"## Timing summary ({n} inputs)")
    L.append("")
    L.append("| stage | total (s) | mean (s) |")
    L.append("|---|---:|---:|")
    for name, a in (("stt", stt_infer), ("llm", llm_infer), ("tts", tts_infer)):
        L.append(f"| {name} | {f(tot(a))} | {f(tot(a) / n)} |")
    L.append(f"| **cascade** | **{f(cascade)}** | **{f(cascade / n)}** |")
    L.append("")
    L.append(f"- input prep ({'ffmpeg decode' if args.in_dir else 'tts synth'}): **{f(prep)}s**")
    L.append(f"- total wall (load + warmup + prep + cascade): **{f(wall)}s**")
    L.append(f"- STT mean real-time factor (infer / audio): **{sum(stt_rtf) / n:.2f}**")
    L.append(f"- throughput: **{n / cascade:.2f}** inputs/sec of cascade inference")
    report = "\n".join(L) + "\n"

    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)
    print(f"[saved report -> {args.report}]")
    print(f"[saved {n} reply wavs -> out/reply_*.wav]")


if __name__ == "__main__":
    main()
