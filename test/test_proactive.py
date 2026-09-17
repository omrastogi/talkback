"""End-to-end test for POST /proactive: auth, validation, dedupe, expiry, queue, drain, fan-out.

Drives fake devices over /ws-stream using the real wire protocol -- the `start` frame may now
carry the device's external_id -- and posts the calling service's exact payload shape.

Run it against a server NO REAL DEVICE is connected to: delivery fans out to every active
session, so a suite run would otherwise speak test utterances in someone's room.

    python test/test_proactive.py                       # local server on :8000
    PROACTIVE_BASE=https://<tunnel> PROACTIVE_WS=wss://<tunnel>/ws-stream python test/test_proactive.py
"""
import asyncio
import sys
from datetime import datetime, timezone, timedelta

import requests
import websockets

from proactive_common import BASE, EXT, HDRS, KEY, WS, Results, collect, payload, types

R = Results()


def post(body, headers=None):
    r = requests.post(f"{BASE}/proactive", json=body,
                      headers=HDRS if headers is None else headers, timeout=60)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {"raw": r.text}


async def main():
    print(f"\ntarget {BASE}  external_id={EXT}")

    print("\n[0] auth gate -- the same X-Robin-Key the dashboard API uses")
    code, body = post(payload("no key at all"), headers={"Content-Type": "application/json"})
    R.check("401 with no key", code == 401, f"{code} {body}")
    code, body = post(payload("wrong key"), headers={"Content-Type": "application/json",
                                                     "X-Robin-Key": "not-the-key"})
    R.check("401 with a wrong key", code == 401, f"{code} {body}")
    R.check("a rejected call is neither spoken nor queued", body.get("status") is None, body)

    print("\n[1] empty utterance -> 422")
    code, body = post(payload(""))
    R.check("422 on empty utterance", code == 422, f"{code} {body}")
    R.check("the caller still gets its message_id back", "message_id" in body, body)

    print("\n[2] delivery_by ten minutes past -> expired, not spoken, not queued")
    stale = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    code, body = post(payload("this one is far too late", delivery_by=stale))
    R.check("status expired", body.get("status") == "expired", body)

    print("\n[3] delivery_by = now (what the caller actually sends) -> inside the grace window")
    queued = payload("Time to take your evening medication.")
    code, body = post(queued)
    R.check("not expired: status queued, no live session", body.get("status") == "queued", body)
    R.check("http 200", code == 200, code)

    print("\n[4] the same message_id again -> the prior verdict, not a second utterance")
    code, body = post(queued)
    R.check("dedupe returns the prior status", body.get("status") == "queued", body)
    R.check("marked as a duplicate", body.get("duplicate") is True, body)

    print("\n[5] a second message queues behind the first")
    code, body = post(payload("Also, your daughter called earlier."))
    R.check("second queued", body.get("status") == "queued", body)

    print("\n[6] device connects and names itself -> the queue drains in order, chime first")
    async with websockets.connect(WS, subprotocols=[KEY] if KEY else None, max_size=None) as ws:
        await ws.send(f'{{"type": "start", "sampleRate": 16000, "external_id": "{EXT}"}}')
        first = await collect(ws)
        print("   frames:", types(first))
        R.check("turn_start opens the turn", first[0]["type"] == "turn_start", first[0])
        R.check("listen_suppress before any audio", "listen_suppress" in types(first), types(first))
        R.check("chime frame present",
                any(f.get("type") == "audio_meta" and f.get("chime") for f in first), types(first))
        chime_i = next(i for i, f in enumerate(first) if f.get("chime"))
        reply_i = next(i for i, f in enumerate(first) if f["type"] == "reply")
        R.check("chime precedes the utterance", chime_i < reply_i, f"chime@{chime_i} reply@{reply_i}")
        R.check("oldest queued message spoken first",
                first[reply_i]["text"].startswith("Time to take"), first[reply_i])
        R.check("binary audio actually sent", any(f["type"] == "<binary>" for f in first), types(first))
        R.check("listen_resume before done",
                types(first).index("listen_resume") < types(first).index("done"), types(first))

        second = await collect(ws)
        print("   frames:", types(second))
        R.check("the second queued message drains too",
                any(f["type"] == "reply" and "daughter" in f["text"] for f in second), types(second))

        print("\n[7] live session -> spoken")
        code, body = post(payload("Your walk is scheduled for three o'clock."))
        R.check("status spoken", body.get("status") == "spoken", body)
        R.check("the sessions that heard it are named", bool(body.get("sessions")), body)
        live = await collect(ws)
        print("   frames:", types(live))
        R.check("it reached the device",
                any(f["type"] == "reply" and "walk" in f["text"] for f in live), types(live))

        print("\n[8] the queue is empty now, so a fresh post is spoken rather than queued")
        code, body = post(payload("One more thing."))
        R.check("still spoken", body.get("status") == "spoken", body)
        await collect(ws)

        print("\n[9] a second device joins -> one post is heard by BOTH (no external_id to pick)")
        async with websockets.connect(WS, subprotocols=[KEY] if KEY else None, max_size=None) as ws2:
            await ws2.send('{"type": "start", "sampleRate": 16000}')      # unnamed, as today's client
            await asyncio.sleep(0.5)
            code, body = post(payload("Everyone should hear this."))
            R.check("two sessions named in the response", len(body.get("sessions") or []) == 2, body)
            a, b = await collect(ws), await collect(ws2)
            R.check("first device heard it",
                    any(f["type"] == "reply" and "Everyone" in f["text"] for f in a), types(a))
            R.check("second device heard it",
                    any(f["type"] == "reply" and "Everyone" in f["text"] for f in b), types(b))

    print("\n[10] every device gone -> back to queueing")
    await asyncio.sleep(1.0)
    code, body = post(payload("Nobody is listening now."))
    R.check("queued after disconnect", body.get("status") == "queued", body)

    return R.report()


sys.exit(asyncio.run(main()))
