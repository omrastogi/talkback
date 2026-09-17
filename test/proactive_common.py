"""Shared setup for the POST /proactive tests: target endpoint, shared key, payload shape.

The payload mirrors the calling service's _post_to_ca() field for field, including its habit
of stamping delivery_by with the CURRENT time -- the case DELIVERY_GRACE_S exists for.

Point the tests at any deployment with env vars (default: a local server on :8000):

    python test/test_proactive.py
    PROACTIVE_BASE=https://<tunnel> PROACTIVE_WS=wss://<tunnel>/ws-stream python test/test_proactive.py
"""
import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("PROACTIVE_BASE", "http://127.0.0.1:8000")
WS = os.environ.get("PROACTIVE_WS", "ws://127.0.0.1:8000/ws-stream")
# Test devices announce an external_id no real device uses, so a run can never speak at one.
EXT = os.environ.get("PROACTIVE_EXT", "test-endpoint-001")


def api_key():
    """ROBIN_API_KEY from .env -- the same key the dashboard API and the WebSocket use."""
    for line in open(os.path.join(HERE, "..", ".env"), encoding="utf-8"):
        if line.startswith("ROBIN_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


KEY = api_key()
HDRS = {"Content-Type": "application/json", "X-Robin-Key": KEY}


def payload(utterance, external_id=EXT, message_type="proactive", delivery_by=None, message_id=None):
    """The calling service's exact body (see _post_to_ca)."""
    now = datetime.now(timezone.utc).isoformat()
    return {"service": "ambient-reminder", "message_id": message_id or str(uuid.uuid4()),
            "severity": 0.9, "message": utterance, "message_type": message_type,
            "external_id": external_id, "occurred_at": now,
            "delivery_by": delivery_by or now, "utterance": utterance,
            "require_affirmation": False}


class Results:
    """Tiny pass/fail tally -- these are end-to-end scripts against a live GPU server, not
    unit tests, so they stay runnable by hand without a test runner."""

    def __init__(self):
        self.passed, self.failed = [], []

    def check(self, name, cond, detail=""):
        (self.passed if cond else self.failed).append(name)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + str(detail)) if detail else ''}")

    def report(self):
        print(f"\n{len(self.passed)} passed, {len(self.failed)} failed")
        if self.failed:
            print("FAILED:", ", ".join(self.failed))
        return 1 if self.failed else 0


async def collect(ws, seconds=25.0):
    """Frames received until `done` or the deadline. Binary audio is recorded as a marker so
    the frame ORDER (chime before speech, suppress before both) stays visible."""
    frames, deadline = [], time.time() + seconds
    while time.time() < deadline:
        try:
            m = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - time.time()))
        except asyncio.TimeoutError:
            break
        if isinstance(m, bytes):
            frames.append({"type": "<binary>", "bytes": len(m)})
        else:
            frames.append(json.loads(m))
            if frames[-1].get("type") == "done":
                break
    return frames


def types(frames):
    return [f["type"] for f in frames]
