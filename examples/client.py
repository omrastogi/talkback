#!/usr/bin/env python3
"""Minimal standalone reference client for the talkback voice service's /ws-stream endpoint
(tap-to-talk mode -- the server's default; see ../API.md for the full protocol).

Deliberately has ZERO dependency on this project's code, so it doubles as an independent
protocol test harness. Only third-party dependency: `websockets` (pip install websockets).

Protocol implemented (see API.md for the authoritative spec):
  -> {"type": "start", "sampleRate": N}      once, declares the input's sample rate
  -> {"type": "turn_start"}                  arms the server's VAD for one turn
  -> <binary PCM16LE mono frames>            the utterance, chunked, then ~1.2s of silence
                                              so the server's trailing-silence VAD reliably
                                              detects end-of-utterance
  <- {"type": "vad_state", ...}              (informational, printed)
  <- {"type": "transcript", "text": ...}
  <- {"type": "turn_start", "turn_id": ...}
  <- {"type": "reply", "text": ...}          one per reply sentence
  <- {"type": "audio_meta", ...}             immediately followed by...
  <- <binary WAV bytes>                      ...one WAV file per reply sentence (PCM16, 24kHz)
  <- {"type": "done"}                        reply finished -- we stop and save

Usage:
    python client.py input.wav -o reply.wav
    python client.py input.wav -o reply.wav --url ws://localhost:9000/ws-stream
    ROBIN_API_KEY=<key> python client.py input.wav -o reply.wav   # if the server requires auth
"""
import argparse
import array
import asyncio
import io
import json
import os
import sys
import wave

import websockets

DEFAULT_URL = "wss://gateway.parcs.northeastern.edu/ai-caring/ca2/ws-stream"
CHUNK_MS = 20            # size of each simulated mic frame sent to the server
SILENCE_TAIL_S = 1.2     # trailing silence appended so the server's VAD (default 600ms
                         # hangover) reliably fires end-of-utterance


def load_pcm16_mono(wav_path):
    """Read a WAV file and return (pcm16le_bytes, sample_rate). Downmixes stereo to mono.
    Only supports 16-bit PCM input (what the server expects on the wire)."""
    with wave.open(wav_path, "rb") as wf:
        channels, sampwidth, rate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sampwidth != 2:
        raise ValueError(f"{wav_path}: only 16-bit PCM WAV is supported, got {sampwidth * 8}-bit")
    samples = array.array("h")
    samples.frombytes(raw)
    if sys.byteorder == "big":            # WAV PCM is little-endian on disk
        samples.byteswap()
    if channels == 2:
        mono = array.array("h", (0,)) * (len(samples) // 2)
        for i in range(len(mono)):
            mono[i] = (samples[2 * i] + samples[2 * i + 1]) // 2
        samples = mono
    elif channels != 1:
        raise ValueError(f"{wav_path}: only mono or stereo WAV is supported, got {channels} channels")
    if sys.byteorder == "big":
        samples.byteswap()                # convert back to little-endian for the wire
    return samples.tobytes(), rate


def chunk_bytes(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


async def send_utterance(ws, pcm_bytes, sample_rate):
    await ws.send(json.dumps({"type": "start", "sampleRate": sample_rate}))
    await ws.send(json.dumps({"type": "turn_start"}))

    bytes_per_sample = 2
    chunk_samples = max(1, int(sample_rate * CHUNK_MS / 1000))
    chunk_size = chunk_samples * bytes_per_sample

    for chunk in chunk_bytes(pcm_bytes, chunk_size):
        await ws.send(chunk)

    silence = b"\x00" * chunk_size
    for _ in range(int(SILENCE_TAIL_S * 1000 / CHUNK_MS)):
        await ws.send(silence)

    print(f"[client] sent {len(pcm_bytes)} bytes @ {sample_rate} Hz + "
          f"{SILENCE_TAIL_S}s silence tail", file=sys.stderr)


async def receive_reply(ws, out_path):
    """Consume server messages for exactly one turn, concatenating reply WAV chunks into
    out_path. Returns once 'done' arrives (or arm_timeout/vad_error, with no audio)."""
    reply_frames = []
    reply_params = None
    reply_text = []

    while True:
        msg = await ws.recv()
        if isinstance(msg, (bytes, bytearray)):
            with wave.open(io.BytesIO(msg), "rb") as wf:
                params = wf.getparams()
                frames = wf.readframes(wf.getnframes())
            if reply_params is None:
                reply_params = params
            reply_frames.append(frames)
            continue

        data = json.loads(msg)
        mtype = data.get("type")
        if mtype == "vad_state":
            print(f"[server] vad_state={data['state']} prob={data['prob']:.2f}", file=sys.stderr)
        elif mtype == "transcript":
            print(f"[server] transcript: {data['text']!r}", file=sys.stderr)
        elif mtype == "turn_start":
            print(f"[server] turn_start turn_id={data['turn_id']}", file=sys.stderr)
        elif mtype == "reply":
            reply_text.append(data["text"])
            print(f"[server] reply sentence: {data['text']!r}", file=sys.stderr)
        elif mtype == "audio_meta":
            pass  # paired with the binary frame that follows; nothing to do here
        elif mtype == "arm_timeout":
            print("[server] arm_timeout -- no speech detected, nothing to save", file=sys.stderr)
            return
        elif mtype == "vad_error":
            print(f"[server] vad_error: {data.get('text')}", file=sys.stderr)
        elif mtype == "done":
            break
        else:
            print(f"[server] (unhandled) {data}", file=sys.stderr)

    if not reply_frames:
        print("[client] no reply audio received", file=sys.stderr)
        return

    with wave.open(out_path, "wb") as out:
        out.setparams(reply_params)
        for frames in reply_frames:
            out.writeframes(frames)
    print(f"[client] wrote {out_path} "
          f"({reply_params.framerate} Hz, {len(reply_frames)} sentence(s))", file=sys.stderr)
    print("full reply text:", " ".join(reply_text))


async def run(url, wav_path, out_path, key):
    pcm_bytes, sample_rate = load_pcm16_mono(wav_path)
    # Auth key (if the server requires one) is sent as a WebSocket subprotocol, not a query
    # param -- a query param ends up verbatim in nginx's/uvicorn's access logs on every
    # connection. A rejected key closes with code 1008 (see API.md).
    subprotocols = [key] if key else None
    try:
        async with websockets.connect(url, subprotocols=subprotocols, max_size=None) as ws:
            await send_utterance(ws, pcm_bytes, sample_rate)
            await receive_reply(ws, out_path)
    except websockets.exceptions.ConnectionClosedError as e:
        if e.code == 1008:
            sys.exit("[client] rejected: bad or missing API key (close code 1008)")
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", help="input WAV file to stream up (16-bit PCM, mono or stereo)")
    ap.add_argument("-o", "--out", default="reply.wav", help="path to save the reply WAV to")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"WebSocket URL (default: {DEFAULT_URL})")
    ap.add_argument("--key", default=os.environ.get("ROBIN_API_KEY"),
                     help="API key (default: $ROBIN_API_KEY). Omit if the server has no "
                          "ROBIN_API_KEY set (auth disabled).")
    args = ap.parse_args()
    asyncio.run(run(args.url, args.wav, args.out, args.key))


if __name__ == "__main__":
    main()
