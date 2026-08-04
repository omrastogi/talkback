"""Headless latency benchmark for /ws — no mic, no browser.

Mirrors index.html's playback telemetry: it timestamps each received audio frame and runs a
replica of the browser's Web Audio *gapless scheduler* (startAt = max(now, nextStart)) to derive
the same underrun / total-gap / time-to-first-play numbers a real listener would see. One JSONL
line per turn, in the exact schema scripts/analyze_telemetry.py reads.

Inputs are scripted utterances synthesized once with Kokoro (so runs are repeatable and need no
mic). STT transcribes that speech server-side — transcription need not be perfect, we're timing
the audio round-trip, not accuracy.

    python bench_client.py --synth-only --wavdir /tmp/bench_wavs        # run once, before servers
    python bench_client.py --port 8000 --wavdir /tmp/bench_wavs --out log/bench/openai_robin.jsonl --label openai_robin
"""
import argparse
import asyncio
import io
import json
import os
import time

import soundfile as sf
import websockets

# Conversation scripts — each is one ws connection (fresh per-connection memory). Mixed depth:
# a 1-turn greeting, a 2-turn follow-up, a 3-turn emotional close (robin's CONVERSATION_END), and
# a 2-turn delete flow (robin's DELETE_MESSAGE + affirmation sentinel). Same inputs for every
# persona so with/without-harness is a controlled comparison.
CONVOS = {
    "greet": ["Hello, how are you doing today?"],
    "sleep": ["I have been feeling really tired lately.",
              "What are some things that might help me sleep better?"],
    "memory": ["I want to tell you about my morning walk in the garden.",
               "It reminds me of the walks I used to take with my late husband.",
               "Thank you for listening. I think I will rest now. Goodbye."],
    "delete": ["Actually, please delete my last message.",
               "Yes, go ahead and delete it."],
}

MS = 1000.0


def wav_path(wavdir, convo, i):
    return os.path.join(wavdir, f"{convo}_{i}.wav")


def synth_all(wavdir):
    """Synthesize every scripted utterance to a 24 kHz wav (Kokoro). Run before any server so it
    doesn't fight the server for the 4 GB card."""
    os.makedirs(wavdir, exist_ok=True)
    from tts import load_pipe, synth
    pipe = load_pipe()
    for convo, turns in CONVOS.items():
        for i, text in enumerate(turns):
            sf.write(wav_path(wavdir, convo, i), synth(pipe, text), 24000)
            print("synth", convo, i, repr(text[:48]))


def wav_duration(b):
    with sf.SoundFile(io.BytesIO(b)) as f:
        return len(f) / f.samplerate


async def run_turn(ws, wav_bytes, label, records):
    """Send one utterance, consume frames until 'done', record browser-equivalent telemetry."""
    end_of_speech = time.perf_counter() * MS          # 'PTT release' == finished sending audio
    await ws.send(wav_bytes)

    turn_id, pending_meta = None, None
    first_recv = first_play = None
    next_start = prev_end = None                       # gapless scheduler (audio-clock seconds)
    underruns, total_gap_ms, chunks = 0, 0.0, []

    while True:
        msg = await asyncio.wait_for(ws.recv(), timeout=120)
        if isinstance(msg, (bytes, bytearray)):
            recv = time.perf_counter()
            if first_recv is None:
                first_recv = recv * MS
            dur = wav_duration(bytes(msg))
            start_at = max(recv, next_start if next_start is not None else recv)
            if prev_end is not None and start_at > prev_end + 1e-4:   # buffer drained -> underrun
                underruns += 1
                total_gap_ms += (start_at - prev_end) * MS
            prev_end = next_start = start_at + dur
            if first_play is None:
                first_play = start_at * MS
            chunks.append({
                "idx": pending_meta["idx"] if pending_meta else None,
                "server_send_ts": pending_meta["server_send_ts"] if pending_meta else None,
                "client_recv_ts": recv * MS,
                "decode_done_ts": time.perf_counter() * MS,
                "scheduled_start_ts": start_at,
                "scheduled_end_ts": prev_end,
                "duration_s": dur,
            })
            pending_meta = None
        else:
            m = json.loads(msg)
            if m.get("type") == "turn_start":
                turn_id = m.get("turn_id")
            elif m.get("type") == "audio_meta":
                pending_meta = m
            elif m.get("type") == "done":
                break

    records.append({
        "turn_id": turn_id, "label": label,
        "end_of_speech_ts": end_of_speech,
        "first_audio_recv_ts": first_recv, "first_audio_play_ts": first_play,
        "time_to_first_audio_recv": (first_recv - end_of_speech) if first_recv else None,
        "time_to_first_audio_play": (first_play - end_of_speech) if first_play else None,
        "underrun_count": underruns, "total_gap_ms": total_gap_ms,
        "chunk_count": len(chunks), "chunks": chunks,
    })


async def run_all(port, wavdir, out, label):
    records = []
    for convo, turns in CONVOS.items():
        async with websockets.connect(f"ws://localhost:{port}/ws", max_size=None) as ws:
            for i in range(len(turns)):
                with open(wav_path(wavdir, convo, i), "rb") as fh:
                    await run_turn(ws, fh.read(), f"{label}:{convo}", records)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} turns -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-only", action="store_true")
    ap.add_argument("--wavdir", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    if a.synth_only:
        synth_all(a.wavdir)
        return
    if not a.out:
        ap.error("--out required unless --synth-only")
    asyncio.run(run_all(a.port, a.wavdir, a.out, a.label))


if __name__ == "__main__":
    main()
