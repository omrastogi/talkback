"""Headless tap-to-talk gate for /ws-stream (turn_mode=tap). Analog of wstest_stream.py.

Streams a wav as continuous 16 kHz PCM, sends {turn_start} to arm, then trailing silence to
trip the VAD hangover, and asserts the turn completes: VAD reaches SPEECH, then a transcript +
reply + done arrive. Frames are sent back-to-back — VAD timing is audio-frame based, so wall
clock doesn't matter.

    python wstest_tap.py --port 8071 --wav in/inp6.wav
"""
import argparse
import asyncio
import json

import numpy as np
import soundfile as sf
import torch
import torchaudio
import websockets


def load_16k_int16(path):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != 16000:
        data = torchaudio.functional.resample(torch.from_numpy(data), sr, 16000).numpy()
    return (np.clip(data, -1.0, 1.0) * 32767).astype("<i2")


async def main(port, wav):
    pcm = load_16k_int16(wav)
    frames = [pcm[i:i + 512] for i in range(0, len(pcm), 512)]
    silence = np.zeros(512, dtype="<i2")
    states, transcript, reply, done, armed_out = [], "", [], False, False

    async with websockets.connect(f"ws://127.0.0.1:{port}/ws-stream", max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "sampleRate": 16000}))
        await ws.send(json.dumps({"type": "turn_start"}))
        for f in frames:
            await ws.send(f.tobytes())
        for _ in range(70):                          # ~2.2 s silence >> 600 ms hangover
            await ws.send(silence.tobytes())
        try:
            while not done:
                msg = await asyncio.wait_for(ws.recv(), timeout=60)
                if isinstance(msg, (bytes, bytearray)):
                    continue
                m = json.loads(msg)
                t = m.get("type")
                if t == "vad_state":
                    states.append(m["state"])
                elif t == "transcript":
                    transcript = m["text"]
                elif t == "reply":
                    reply.append(m["text"])
                elif t == "arm_timeout":
                    armed_out = True
                    break
                elif t == "done":
                    done = True
        except asyncio.TimeoutError:
            print("TIMEOUT waiting for done")

    collapsed = []
    for s in states:
        if not collapsed or collapsed[-1] != s:
            collapsed.append(s)
    print("VAD state path:", " -> ".join(collapsed) or "(none)")
    print("transcript    :", repr(transcript))
    print("reply         :", repr(" ".join(reply)))
    if armed_out:
        print("RESULT: FAIL (ARM_TIMEOUT — VAD never confirmed speech)")
        return 1
    ok = done and bool(transcript) and ("SPEECH" in states)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--wav", default="in/inp6.wav")
    a = ap.parse_args()
    raise SystemExit(asyncio.run(main(a.port, a.wav)))
