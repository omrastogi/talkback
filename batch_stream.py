"""Streaming (overlapped) profiler: same inputs as batch.py, but each reply is streamed
token-by-token and TTS'd sentence-by-sentence with the stages overlapped (see stream.py).
Reports the realtime latencies (ttft, ttfs, ttfa, total) instead of a sequential cascade sum.

Loads Parakeet + Kokoro once (both resident on 4 GB), warms both, then loops the inputs.

    conda run -n voice python batch_stream.py [--in-dir in] [--report out/PROFILE_stream.md]
"""
import argparse
import glob
import os
import time
from datetime import datetime

import torch

import batch    # decode_16k, natkey, audio_seconds, warmup_stt, AUDIO_EXTS
import config   # make_client
import stream   # load_stt, transcribe, warm_kokoro, run_overlap


def f(x):
    return f"{x:.2f}" if x is not None else "n/a"


def seq_first_audio(r):
    """What a non-overlapped pipeline would spend to first audio: wait for the whole reply,
    then synthesize the first sentence (approximated by this run's first-sentence TTS latency)."""
    if r["ttfa"] is None or r["ttfs"] is None:
        return None
    return r["llm_done"] + (r["ttfa"] - r["ttfs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="in")
    ap.add_argument("--backend", choices=["openai", "parcs"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--report", default="out/PROFILE_stream.md")
    args = ap.parse_args()
    os.makedirs("out", exist_ok=True)

    from kokoro import KPipeline

    wall0 = time.time()

    # ---- load models once ----
    t = time.time(); stt_model = stream.load_stt(); stt_load = time.time() - t
    t = time.time(); pipe = KPipeline(lang_code="a"); tts_load = time.time() - t
    t = time.time(); client, model = config.make_client(args.backend, args.model); llm_load = time.time() - t

    # ---- warm both GPU models so input #1 isn't cold; open audio device once ----
    t = time.time(); batch.warmup_stt(stt_model); warm_stt = time.time() - t
    t = time.time(); stream.warm_kokoro(pipe, args.voice); warm_tts = time.time() - t
    out_stream, playing = stream.open_output()

    # ---- gather inputs: ffmpeg-decode to 16 kHz mono wavs (soundfile can't read mp4) ----
    files = sorted(
        (p for p in glob.glob(os.path.join(args.in_dir, "*")) if p.lower().endswith(batch.AUDIO_EXTS)),
        key=lambda p: batch.natkey(os.path.basename(p)),
    )
    if not files:
        raise SystemExit(f"no audio files in {args.in_dir}")
    t = time.time()
    names, sources, wavs = [], [], []
    for src in files:
        stem = os.path.splitext(os.path.basename(src))[0]
        dst = f"out/_norm_{stem}.wav"
        batch.decode_16k(src, dst)
        names.append(stem); sources.append(os.path.basename(src)); wavs.append(dst)
    prep = time.time() - t
    durations = [batch.audio_seconds(w) for w in wavs]

    # ---- per-input: STT (before t0) then overlapped streaming reply ----
    rows = []
    for i, w in enumerate(wavs):
        t = time.time(); transcript = stream.transcribe(stt_model, w); stt_infer = time.time() - t
        r = stream.run_overlap(client, pipe, transcript, model, args.voice,
                               f"out/stream_reply_{names[i]}.wav", out_stream, playing)
        r["stt_infer"] = stt_infer
        r["transcript"] = transcript
        rows.append(r)

    if out_stream is not None:
        out_stream.stop()
        out_stream.close()
    wall = time.time() - wall0

    # ---- report ----
    n = len(rows)
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    mean = lambda k: sum(r[k] for r in rows) / n
    play = "LIVE" if any(r["play"] for r in rows) else "buffered-only (no audio device)"
    overlapped = sum(1 for r in rows if r["ttfs"] is not None and r["ttfs"] < r["llm_done"])
    mean_seq = sum(seq_first_audio(r) for r in rows) / n

    L = []
    L.append(f"# Streaming (overlapped) profile — {datetime.now():%Y-%m-%d %H:%M}")
    L.append("")
    L.append(f"- Inputs: **{n}** (files from {args.in_dir}/)")
    L.append("- Pipeline: STT (blocking) → LLM token stream → sentence chunks → Kokoro → play/buffer, "
             "stages overlapped via threads + queues (see `stream.py`)")
    backend = args.backend or os.environ.get("LLM_BACKEND", "parcs")
    L.append(f"- Stack: STT Parakeet-tdt-0.6b-v3 (GPU: {device}) · LLM `{model}` ({backend}, streaming) · TTS Kokoro-82M")
    L.append(f"- Model load (once each): stt **{f(stt_load)}s** · tts **{f(tts_load)}s** · openai-client **{f(llm_load)}s**")
    L.append(f"- Warmup (discarded, so input #1 isn't cold): stt **{f(warm_stt)}s** · tts **{f(warm_tts)}s**")
    L.append(f"- Playback: **{play}**")
    L.append("")
    L.append("Latencies are seconds relative to **t0** (set after STT, just before the OpenAI call — "
             "the end-of-utterance analog). `stt_infer` is measured before t0. "
             "`ttft`=first token, `ttfs`=first sentence flushed to TTS, `ttfa`=first audio out, `total`=last audio out.")
    L.append("")
    L.append("## Transcripts & replies")
    for i, r in enumerate(rows):
        seq = seq_first_audio(r)
        saved = (seq - r["ttfa"]) if (seq is not None and r["ttfa"] is not None) else None
        L.append("")
        L.append(f"### {names[i]}  ({f(durations[i])}s audio)")
        L.append(f"- source: {sources[i]}")
        L.append(f"- heard (STT): {r['transcript']!r}")
        L.append(f"- reply (LLM): {r['reply']!r}")
        L.append(f"- reply audio: `out/stream_reply_{names[i]}.wav`")
        L.append(f"- latency: stt_infer={f(r['stt_infer'])}  ttft={f(r['ttft'])}  ttfs={f(r['ttfs'])}  "
                 f"ttfa={f(r['ttfa'])}  total={f(r['total'])}")
        L.append(f"- overlap: first audio {f(r['ttfa'])}s vs ~{f(seq)}s sequential → **{f(saved)}s sooner**"
                 f"{'  (ttfs before llm_done ✓)' if (r['ttfs'] is not None and r['ttfs'] < r['llm_done']) else ''}")
    L.append("")
    L.append(f"## Latency summary ({n} inputs)")
    L.append("")
    L.append("| metric | mean (s) | what it measures |")
    L.append("|---|---:|---|")
    L.append(f"| stt_infer | {f(mean('stt_infer'))} | transcription (before t0) |")
    L.append(f"| ttft | {f(mean('ttft'))} | first LLM token |")
    L.append(f"| ttfs | {f(mean('ttfs'))} | first sentence flushed to TTS |")
    L.append(f"| ttfa | {f(mean('ttfa'))} | first audio chunk out |")
    L.append(f"| total | {f(mean('total'))} | last audio chunk out |")
    L.append("")
    L.append(f"- true overlap (first sentence flushed **before** the LLM finished): **{overlapped}/{n}** inputs")
    L.append(f"- first audio: **{f(mean('ttfa'))}s** overlapped vs **{f(mean_seq)}s** sequential (mean) "
             f"→ **{f(mean_seq - mean('ttfa'))}s** sooner")
    L.append(f"- total wall (load + warmup + prep + {n} inputs): **{f(wall)}s**")
    report = "\n".join(L) + "\n"

    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)
    print(f"[saved report -> {args.report}]")


if __name__ == "__main__":
    main()
