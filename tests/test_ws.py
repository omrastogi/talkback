"""Voice-socket auth, via the stub app in conftest that mounts the same
bind_device_session/persist_turn wiring server.py uses. The token rides the
Sec-WebSocket-Protocol header (the `subprotocols` argument), exactly as on the real socket.
"""
import asyncio
import uuid

import pytest
from sqlalchemy import select
from starlette.websockets import WebSocketDisconnect

import robin.db as rdb
from robin.db.models import ConversationTurn
from tests.helpers import dashboard_token, device_token, make_account, make_profile


def _run(coro):
    return asyncio.run(coro)


def test_dashboard_token_rejected_with_4401(client):
    aid = _run(make_account())
    raw, _ = _run(dashboard_token(aid))
    with client.websocket_connect("/ws", subprotocols=[raw]) as ws:
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
    assert exc.value.code == 4401


def test_missing_and_bogus_tokens_rejected(client):
    for protocols in ([], ["not-a-real-token"]):
        kwargs = {"subprotocols": protocols} if protocols else {}
        with client.websocket_connect("/ws", **kwargs) as ws:
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
        assert exc.value.code == 4401


def test_device_token_resolves_correct_profile(client):
    pid_a = _run(make_profile("A"))
    pid_b = _run(make_profile("B"))
    raw_b, _ = _run(device_token(pid_b))
    with client.websocket_connect("/ws", subprotocols=[raw_b]) as ws:
        ws.send_json({"type": "start"})
        reply = ws.receive_json()
    assert reply["profile_id"] == pid_b != pid_a
    uuid.UUID(reply["session_id"])                 # server-minted, well-formed


def test_client_sent_profile_id_is_ignored(client):
    pid_a = _run(make_profile("A"))
    pid_b = _run(make_profile("B"))
    raw_a, _ = _run(device_token(pid_a))
    # The client claims to be profile B in its connect payload; binding must stay A.
    with client.websocket_connect("/ws", subprotocols=[raw_a]) as ws:
        ws.send_json({"type": "start", "profile_id": pid_b})
        reply = ws.receive_json()
    assert reply["client_sent_profile_id"] == pid_b     # the claim arrived...
    assert reply["profile_id"] == pid_a                 # ...and changed nothing


def test_reconnect_is_a_new_session(client):
    pid = _run(make_profile())
    raw, _ = _run(device_token(pid))
    sids = []
    for _ in range(2):
        with client.websocket_connect("/ws", subprotocols=[raw]) as ws:
            ws.send_json({"type": "start"})
            sids.append(ws.receive_json()["session_id"])
    assert sids[0] != sids[1]


def test_turn_persisted_under_token_profile(client):
    pid = _run(make_profile())
    raw, _ = _run(device_token(pid))
    with client.websocket_connect("/ws", subprotocols=[raw]) as ws:
        ws.send_json({"type": "start", "speak": "good morning robin"})
        reply = ws.receive_json()
    async def fetch():
        async with rdb._sessionmaker() as s:
            return (await s.execute(select(ConversationTurn))).scalars().all()
    turns = _run(fetch())
    assert len(turns) == 1
    assert turns[0].profile_id == pid
    assert str(turns[0].session_id) == reply["session_id"]
    assert (turns[0].role, turns[0].content, turns[0].turn_index) == \
        ("user", "good morning robin", 0)


def test_revoked_device_token_rejected(client):
    from robin.auth.tokens import revoke_token
    pid = _run(make_profile())
    raw, token_id = _run(device_token(pid))
    async def revoke():
        async with rdb._sessionmaker() as s:
            await revoke_token(s, token_id)
    _run(revoke())
    with client.websocket_connect("/ws", subprotocols=[raw]) as ws:
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
    assert exc.value.code == 4401
