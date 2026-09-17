"""The dashboard greet path must be untouched by the proactive work: same frames as before,
no chime, no listen_suppress/listen_resume, and a session that names no external_id.

    python test/test_greet_regression.py
"""
import asyncio
import sys

import requests
import websockets

from proactive_common import BASE, KEY, WS, Results, collect, types

R = Results()


async def main():
    print(f"\ntarget {BASE}")
    async with websockets.connect(WS, subprotocols=[KEY] if KEY else None, max_size=None) as ws:
        await ws.send('{"type": "start", "sampleRate": 16000}')     # no external_id, as today
        await asyncio.sleep(0.5)
        sessions = requests.get(f"{BASE}/api/sessions", headers={"X-Robin-Key": KEY},
                                timeout=15).json()["sessions"]
        mine = sessions[0]
        r = requests.post(f"{BASE}/api/sessions/{mine['id']}/greet",
                          json={"text": "How is your day going?"},
                          headers={"X-Robin-Key": KEY, "Content-Type": "application/json"}, timeout=60)
        R.check("greet accepted", r.status_code == 200, f"{r.status_code} {r.text}")
        frames = await collect(ws)
        print("   frames:", types(frames))
        R.check("frame sequence unchanged",
                types(frames) == ["turn_start", "reply", "audio_meta", "<binary>", "done"], types(frames))
        R.check("no proactive frames leak into a greet",
                not any(f.get("chime") or f["type"].startswith("listen_") for f in frames), types(frames))
        R.check("a session that never named itself has no external_id",
                mine.get("external_id") is None, mine)
    return R.report()


sys.exit(asyncio.run(main()))
