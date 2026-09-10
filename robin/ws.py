"""Voice-socket persistence wiring, shared by server.py's two WebSocket endpoints and by
the test suite (which mounts these same functions in a stub app so tests never import
server.py and its model loading).

Session identity: `BoundSession.session_id` is a UUID minted here, at accept time, per
connection. A reconnect is a new session with a new UUID — there is no resume, and no part
of session identity is ever inferred from model output (the RECOVER predecessor cut
sessions on an LLM-emitted keyword; that failure mode is designed out here).
"""
import dataclasses
import logging
import uuid

from fastapi import WebSocket

from robin.auth.deps import WS_CLOSE_UNAUTHORIZED, authenticate_device_ws, ws_presented_token
from robin.db import get_sessionmaker
from robin.db.models import ConversationTurn

log = logging.getLogger("voice")


@dataclasses.dataclass
class BoundSession:
    """Per-connection profile snapshot + turn counter. The profile fields are read once at
    connect and used for the lifetime of the connection (a PATCH from the dashboard takes
    effect on the next connect, not mid-call)."""
    profile_id: int
    display_name: str
    voice: str
    speech_rate: float
    timezone: str
    context: dict
    session_id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)
    turn_index: int = 0

    def next_index(self) -> int:
        i = self.turn_index
        self.turn_index += 1
        return i


async def bind_device_session(ws: WebSocket) -> BoundSession | None:
    """Authenticate the handshake and bind a profile, or close the socket.

    The token travels as the Sec-WebSocket-Protocol value (never a query string — it would
    persist in nginx/uvicorn access logs). Rejection follows the repo's accept-then-close
    pattern (closing pre-accept surfaces as 1006 in browsers) with code 4401. On success
    the socket is accepted echoing the subprotocol, exactly as the shared-key auth did."""
    raw = ws_presented_token(ws)
    profile = await authenticate_device_ws(raw) if raw else None
    if profile is None:
        await ws.accept()
        await ws.close(code=WS_CLOSE_UNAUTHORIZED)
        return None
    await ws.accept(subprotocol=raw)
    return BoundSession(profile_id=profile.id, display_name=profile.display_name,
                       voice=profile.voice, speech_rate=profile.speech_rate,
                       timezone=profile.timezone, context=profile.context)


async def persist_turn(bound: BoundSession | None, *, role: str, content: str,
                       source: str = "voice", latency_ms: int | None = None,
                       meta: dict | None = None) -> None:
    """Write one conversation_turn row, in-path with await. A failed write is logged and
    swallowed: persistence must never kill the connection or block TTS."""
    if bound is None or not content:
        return
    try:
        async with get_sessionmaker()() as session:
            session.add(ConversationTurn(
                profile_id=bound.profile_id, session_id=bound.session_id,
                turn_index=bound.next_index(), role=role, content=content,
                source=source, latency_ms=latency_ms, meta=meta or {}))
            await session.commit()
    except Exception as e:                       # noqa: BLE001 — persistence is best-effort
        log.error("turn write failed (session=%s role=%s): %r", bound.session_id, role, e)
