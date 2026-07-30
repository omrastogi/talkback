"""Gate-4 self-test for the streaming server (no browser). Streams out/hello.wav as Int16
PCM frames like the browser worklet does, then asserts we get >=1 partial, a transcript,
>=1 binary audio frame, and done."""
import asyncio
import json

import numpy as np
import soundfile as sf
import torch
import torchaudio
import websockets

FRAME = 3200   # samples per ws frame (~0.2 s at 16 kHz), like the browser's small frames


async def main():
    data, sr = sf.read("out/hello.wav", dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != 16000:
        data = torchaudio.functional.resample(torch.from_numpy(data), sr, 16000).numpy()
    i16 = (np.clip(data, -1.0, 1.0) * 32767).astype("<i2")   # little-endian Int16

    async with websockets.connect("ws://localhost:8000/ws-stream", max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "sampleRate": 16000}))
        for i in range(0, len(i16), FRAME):
            await ws.send(i16[i:i + FRAME].tobytes())
            await asyncio.sleep(0.02)                          # loosely mimic real-time arrival
        await ws.send(json.dumps({"type": "end"}))

        got_partial = got_transcript = got_audio = got_done = False
        while not got_done:
            msg = await asyncio.wait_for(ws.recv(), timeout=90)
            if isinstance(msg, (bytes, bytearray)):
                got_audio = True
            else:
                m = json.loads(msg)
                if m["type"] == "partial":
                    got_partial = True
                    print("partial:", repr(m["text"]))
                elif m["type"] == "transcript":
                    got_transcript = True
                    print("transcript:", repr(m["text"]))
                elif m["type"] == "reply":
                    print("reply:", repr(m["text"]))
                elif m["type"] == "done":
                    got_done = True
        assert got_partial, "no partial frames"
        assert got_transcript, "no transcript frame"
        assert got_audio, "no audio frame"
        print("PASS: partial + transcript + audio + done")


if __name__ == "__main__":
    asyncio.run(main())
