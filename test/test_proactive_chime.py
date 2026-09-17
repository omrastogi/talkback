"""The chime on the wire really is chime.mp3: same samples, same rate, ahead of the speech.

A socket that dies mid-utterance is covered by test_proactive_midsend.py instead -- provoking
it by killing a client is a race the server usually wins, and through a tunnel it never does:
cloudflared keeps the origin socket up after the client's TCP is gone.

    python test/test_proactive_chime.py
    PROACTIVE_BASE=https://<tunnel> PROACTIVE_WS=wss://<tunnel>/ws-stream python test/test_proactive_chime.py
"""
import asyncio
import io
import os
import subprocess
import sys
import tempfile

import numpy as np
import requests
import soundfile as sf
import websockets

from proactive_common import BASE, EXT, HERE, HDRS, KEY, WS, Results, collect, payload

R = Results()
CHIME = os.path.join(HERE, "..", "chime.mp3")


def reference_chime():
    """Decode chime.mp3 the same way the server does, to compare against what it sent."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        tmp = fh.name
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", CHIME, "-ac", "1", "-ar", "24000", tmp],
                   check=True)
    audio, sr = sf.read(tmp, dtype="float32")
    os.remove(tmp)
    return audio, sr


async def main():
    print(f"\ntarget {BASE}  external_id={EXT}")
    ref, ref_sr = reference_chime()

    print("\n[A] the chime frame carries chime.mp3, ahead of the speech")
    async with websockets.connect(WS, subprotocols=[KEY] if KEY else None, max_size=None) as ws:
        await ws.send(f'{{"type": "start", "sampleRate": 16000, "external_id": "{EXT}"}}')
        body = requests.post(f"{BASE}/proactive", json=payload("Checking the chime."),
                             headers=HDRS, timeout=60).json()
        R.check("spoken into the live session", body.get("status") == "spoken", body)
        frames = await collect(ws)
        metas = [f for f in frames if f["type"] == "audio_meta"]
        blobs = [f for f in frames if f["type"] == "<binary>"]
        R.check("chime flagged on the first audio frame only",
                metas[0].get("chime") is True and not metas[1].get("chime"), metas[:2])
        R.check("two audio frames: chime, then speech", len(blobs) == 2, len(blobs))

    # collect() records only binary SIZES, so run the exchange again keeping the bytes

    async with websockets.connect(WS, subprotocols=[KEY] if KEY else None, max_size=None) as ws:
        await ws.send(f'{{"type": "start", "sampleRate": 16000, "external_id": "{EXT}"}}')
        requests.post(f"{BASE}/proactive", json=payload("Checking the chime again."),
                      headers=HDRS, timeout=60)
        raw = []
        while True:
            m = await asyncio.wait_for(ws.recv(), timeout=30)
            if isinstance(m, bytes):
                raw.append(m)
            elif b'"done"' in m.encode():
                break
        got, sr = sf.read(io.BytesIO(raw[0]), dtype="float32")
        print(f"   sent {len(got)/sr:.3f}s @{sr}Hz   vs   chime.mp3 {len(ref)/ref_sr:.3f}s @{ref_sr}Hz")
        R.check("sample rate matches the speech frames", sr == 24000, sr)
        R.check("same length as chime.mp3", abs(len(got) - len(ref)) < 10, (len(got), len(ref)))
        n = min(len(got), len(ref))
        R.check("same samples as chime.mp3", float(np.abs(got[:n] - ref[:n]).max()) < 1e-4)

    return R.report()


sys.exit(asyncio.run(main()))
