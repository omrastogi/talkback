"""Dashboard HTTP endpoints, including the review-driven invariants: 404 (not 403) for
unlinked profiles, and no response body ever containing 'password' or 'token_hash'."""
import asyncio
import uuid

import pytest

import robin.db as rdb
from robin.db.models import ConversationTurn
from tests.helpers import (PASSWORD, dashboard_token, device_token, link,
                           make_account, make_profile)


def _run(coro):
    return asyncio.run(coro)


def _auth(raw):
    return {"Authorization": f"Bearer {raw}"}


def test_login_success_and_wrong_password(client):
    _run(make_account("cp@example.org"))
    ok = client.post("/auth/login", json={"email": "cp@example.org", "password": PASSWORD})
    assert ok.status_code == 200
    assert ok.json()["token"]
    bad = client.post("/auth/login", json={"email": "cp@example.org", "password": "nope"})
    assert bad.status_code == 401
    unknown = client.post("/auth/login", json={"email": "who@example.org", "password": PASSWORD})
    assert unknown.status_code == 401
    assert bad.json() == unknown.json()            # existence not confirmed either way


def test_login_email_is_case_insensitive(client):
    """citext at work: the same email in different case is the same account."""
    _run(make_account("cp@example.org"))
    r = client.post("/auth/login", json={"email": "CP@Example.ORG", "password": PASSWORD})
    assert r.status_code == 200


def test_logout_revokes_the_presented_token(client):
    aid = _run(make_account())
    raw, _ = _run(dashboard_token(aid))
    assert client.get("/profiles", headers=_auth(raw)).status_code == 200
    assert client.post("/auth/logout", headers=_auth(raw)).status_code == 200
    assert client.get("/profiles", headers=_auth(raw)).status_code == 401


def test_profiles_lists_only_linked(client):
    aid = _run(make_account())
    pid_mine = _run(make_profile("Mine"))
    _run(make_profile("Other"))
    _run(link(aid, pid_mine))
    raw, _ = _run(dashboard_token(aid))
    body = client.get("/profiles", headers=_auth(raw)).json()
    assert [p["id"] for p in body["profiles"]] == [pid_mine]
    assert body["profiles"][0]["role"] == "owner"


def test_unlinked_profile_is_404_not_403(client):
    aid = _run(make_account())
    pid_other = _run(make_profile("Other"))       # exists, but not linked to the caller
    raw, _ = _run(dashboard_token(aid))
    r = client.get(f"/profiles/{pid_other}", headers=_auth(raw))
    assert r.status_code == 404
    missing = client.get("/profiles/999999", headers=_auth(raw))
    assert missing.status_code == 404
    assert r.json() == missing.json()             # indistinguishable from a nonexistent one


def test_patch_profile_owner_only_and_field_allowlist(client):
    aid = _run(make_account())
    pid = _run(make_profile())
    _run(link(aid, pid, role="viewer"))
    raw, _ = _run(dashboard_token(aid))
    assert client.patch(f"/profiles/{pid}", headers=_auth(raw),
                        json={"voice": "am_adam"}).status_code == 403

    aid2 = _run(make_account("owner@example.org"))
    _run(link(aid2, pid, role="owner"))
    raw2, _ = _run(dashboard_token(aid2))
    r = client.patch(f"/profiles/{pid}", headers=_auth(raw2),
                     json={"voice": "am_adam", "speech_rate": 0.9,
                           "timezone": "America/Chicago", "context": {"likes": "gardening"}})
    assert r.status_code == 200
    body = r.json()
    assert (body["voice"], body["timezone"]) == ("am_adam", "America/Chicago")
    assert body["speech_rate"] == pytest.approx(0.9)      # stored as float4 (`real`)
    # Fields outside the allowlist are rejected outright, not silently dropped.
    assert client.patch(f"/profiles/{pid}", headers=_auth(raw2),
                        json={"active": False}).status_code == 422
    assert client.patch(f"/profiles/{pid}", headers=_auth(raw2),
                        json={"timezone": "Mars/Olympus_Mons"}).status_code == 422


def test_turns_pagination_by_session(client):
    aid = _run(make_account())
    pid = _run(make_profile())
    _run(link(aid, pid))
    raw, _ = _run(dashboard_token(aid))
    sid_a, sid_b = uuid.uuid4(), uuid.uuid4()

    async def seed():
        async with rdb._sessionmaker() as s:
            for sid, n in ((sid_a, 3), (sid_b, 2)):
                for i in range(n):
                    s.add(ConversationTurn(profile_id=pid, session_id=sid, turn_index=i,
                                           role="user" if i % 2 == 0 else "assistant",
                                           content=f"turn {i}"))
            await s.commit()
    _run(seed())

    all_turns = client.get(f"/profiles/{pid}/turns", headers=_auth(raw)).json()["turns"]
    assert len(all_turns) == 5
    one = client.get(f"/profiles/{pid}/turns", params={"session_id": str(sid_a)},
                     headers=_auth(raw)).json()["turns"]
    assert len(one) == 3 and all(t["session_id"] == str(sid_a) for t in one)
    limited = client.get(f"/profiles/{pid}/turns", params={"limit": 2},
                         headers=_auth(raw)).json()["turns"]
    assert len(limited) == 2
    before = client.get(f"/profiles/{pid}/turns",
                        params={"before": all_turns[-1]["created_at"]},
                        headers=_auth(raw)).json()["turns"]
    assert all(t["created_at"] < all_turns[-1]["created_at"] for t in before)


def test_no_response_contains_password_or_token_hash(client):
    """The single most important constraint (see the RECOVER as_dict() finding): sweep the
    serialized output of every endpoint for the forbidden substrings."""
    aid = _run(make_account())
    pid = _run(make_profile())
    _run(link(aid, pid))
    _run(device_token(pid))                        # ensure token rows exist too
    raw, _ = _run(dashboard_token(aid))

    responses = [
        client.post("/auth/login", json={"email": "cp@example.org", "password": PASSWORD}),
        client.post("/auth/login", json={"email": "cp@example.org", "password": "wrong"}),
        client.get("/profiles", headers=_auth(raw)),
        client.get(f"/profiles/{pid}", headers=_auth(raw)),
        client.patch(f"/profiles/{pid}", headers=_auth(raw), json={"display_name": "M"}),
        client.get(f"/profiles/{pid}/turns", headers=_auth(raw)),
        client.get("/profiles/999999", headers=_auth(raw)),
        client.post("/auth/logout", headers=_auth(raw)),
    ]
    for r in responses:
        body = r.text.lower()
        assert "password" not in body, f"{r.request.method} {r.url}: 'password' in body"
        assert "token_hash" not in body, f"{r.request.method} {r.url}: 'token_hash' in body"


def test_me_returns_the_token_account(client):
    aid = _run(make_account("me@example.org", is_admin=True))
    raw, _ = _run(dashboard_token(aid))
    body = client.get("/auth/me", headers=_auth(raw)).json()
    assert body == {"account_id": aid, "email": "me@example.org",
                    "display_name": "Care Partner", "is_admin": True}
    assert client.get("/auth/me").status_code == 401


def test_admin_sees_all_profiles_and_can_patch_unlinked(client):
    admin = _run(make_account("admin@example.org", is_admin=True))
    pid_linked = _run(make_profile("Linked"))
    pid_other = _run(make_profile("Other"))
    _run(link(admin, pid_linked, role="viewer"))
    raw, _ = _run(dashboard_token(admin))

    profiles = client.get("/profiles", headers=_auth(raw)).json()["profiles"]
    roles = {p["id"]: p["role"] for p in profiles}
    assert roles == {pid_linked: "viewer", pid_other: "admin"}

    # Unlinked profile: visible, patchable (admin outranks the owner requirement) —
    # but a truly nonexistent id is still a 404.
    got = client.get(f"/profiles/{pid_other}", headers=_auth(raw))
    assert got.status_code == 200 and got.json()["role"] == "admin"
    patched = client.patch(f"/profiles/{pid_other}", headers=_auth(raw),
                           json={"display_name": "Renamed"})
    assert patched.status_code == 200 and patched.json()["display_name"] == "Renamed"
    assert client.get("/profiles/999999", headers=_auth(raw)).status_code == 404


def test_sessions_grouping_and_cursor(client):
    import datetime as dt
    aid = _run(make_account())
    pid = _run(make_profile())
    _run(link(aid, pid))
    raw, _ = _run(dashboard_token(aid))
    sid_a, sid_b = uuid.uuid4(), uuid.uuid4()
    t0 = dt.datetime(2026, 6, 10, 12, 0, tzinfo=dt.timezone.utc)

    async def seed():
        async with rdb._sessionmaker() as s:
            # Session A: 3 turns ending 12:02; session B: 2 turns ending 13:01.
            for i in range(3):
                s.add(ConversationTurn(profile_id=pid, session_id=sid_a, turn_index=i,
                                       role="user" if i % 2 == 0 else "assistant",
                                       content=f"a{i}", source="voice",
                                       created_at=t0 + dt.timedelta(minutes=i)))
            for i in range(2):
                s.add(ConversationTurn(profile_id=pid, session_id=sid_b, turn_index=i,
                                       role="assistant", content=f"b{i}",
                                       source="proactive",
                                       created_at=t0 + dt.timedelta(hours=1, minutes=i)))
            await s.commit()
    _run(seed())

    sessions = client.get(f"/profiles/{pid}/sessions", headers=_auth(raw)).json()["sessions"]
    assert [s["session_id"] for s in sessions] == [str(sid_b), str(sid_a)]  # newest first
    assert sessions[0]["turn_count"] == 2 and sessions[0]["sources"] == ["proactive"]
    assert sessions[1]["turn_count"] == 3 and sessions[1]["sources"] == ["voice"]
    assert sessions[1]["started_at"] < sessions[1]["last_at"]

    older = client.get(f"/profiles/{pid}/sessions",
                       params={"before": sessions[0]["last_at"]},
                       headers=_auth(raw)).json()["sessions"]
    assert [s["session_id"] for s in older] == [str(sid_a)]

    # Same visibility rule as every other profile endpoint.
    other = _run(make_account("other@example.org"))
    raw2, _ = _run(dashboard_token(other))
    assert client.get(f"/profiles/{pid}/sessions", headers=_auth(raw2)).status_code == 404


def test_activity_buckets_days_in_profile_timezone(client):
    import datetime as dt
    aid = _run(make_account())
    pid = _run(make_profile())            # default timezone America/New_York (UTC-4 in June)
    _run(link(aid, pid))
    raw, _ = _run(dashboard_token(aid))
    sid = uuid.uuid4()

    async def seed():
        async with rdb._sessionmaker() as s:
            # 03:30 UTC on June 10 = 23:30 EDT on June 9: must land in the June 9 bucket.
            s.add(ConversationTurn(profile_id=pid, session_id=sid, turn_index=0,
                                   role="user", content="late night", source="voice",
                                   created_at=dt.datetime(2026, 6, 10, 3, 30,
                                                          tzinfo=dt.timezone.utc)))
            s.add(ConversationTurn(profile_id=pid, session_id=sid, turn_index=1,
                                   role="assistant", content="reply", source="voice",
                                   latency_ms=1500,
                                   created_at=dt.datetime(2026, 6, 10, 3, 31,
                                                          tzinfo=dt.timezone.utc)))
            s.add(ConversationTurn(profile_id=pid, session_id=uuid.uuid4(), turn_index=0,
                                   role="assistant", content="greet", source="proactive",
                                   created_at=dt.datetime(2026, 6, 10, 15, 0,
                                                          tzinfo=dt.timezone.utc)))
            await s.commit()
    _run(seed())

    body = client.get(f"/profiles/{pid}/activity",
                      params={"date_from": "2026-06-01", "date_to": "2026-06-30"},
                      headers=_auth(raw)).json()
    assert body["timezone"] == "America/New_York"
    assert body["last_active_at"] is not None
    days = {d["day"]: d for d in body["days"]}
    assert set(days) == {"2026-06-09", "2026-06-10"}
    june9 = days["2026-06-09"]
    assert june9["sessions"] == 1 and june9["user_turns"] == 1
    assert june9["assistant_turns"] == 1 and june9["voice_turns"] == 2
    assert june9["proactive_turns"] == 0 and june9["avg_latency_ms"] == 1500.0
    june10 = days["2026-06-10"]
    assert june10["sessions"] == 1 and june10["proactive_turns"] == 1
    assert june10["user_turns"] == 0 and june10["voice_turns"] == 0

    # Range excluding both days -> empty, but last_active_at is unbounded.
    empty = client.get(f"/profiles/{pid}/activity",
                       params={"date_from": "2026-07-01", "date_to": "2026-07-31"},
                       headers=_auth(raw)).json()
    assert empty["days"] == [] and empty["last_active_at"] is not None
    assert client.get(f"/profiles/{pid}/activity",
                      params={"date_from": "2026-07-02", "date_to": "2026-07-01"},
                      headers=_auth(raw)).status_code == 422


def test_diagnostic_sessions_hidden_by_default(client):
    """Turns from a device token labeled 'diagnostic…' (the dashboard Live page) stay out
    of sessions/turns/activity unless include_diagnostic=true. Turns with no token
    provenance (pre-migration rows) count as real data."""
    aid = _run(make_account())
    pid = _run(make_profile())
    _run(link(aid, pid))
    raw, _ = _run(dashboard_token(aid))
    _, tablet_tid = _run(device_token(pid, label="Tab A9 living room"))
    _, diag_tid = _run(device_token(pid, label="diagnostic: browser — admin@example.org"))
    sid_real, sid_diag, sid_legacy = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

    async def seed():
        async with rdb._sessionmaker() as s:
            for sid, token_id in ((sid_real, tablet_tid), (sid_diag, diag_tid),
                                  (sid_legacy, None)):
                s.add(ConversationTurn(profile_id=pid, session_id=sid, turn_index=0,
                                       role="user", content="hi", auth_token_id=token_id))
            await s.commit()
    _run(seed())

    sessions = client.get(f"/profiles/{pid}/sessions",
                          headers=_auth(raw)).json()["sessions"]
    assert {s["session_id"] for s in sessions} == {str(sid_real), str(sid_legacy)}
    sessions = client.get(f"/profiles/{pid}/sessions",
                          params={"include_diagnostic": "true"},
                          headers=_auth(raw)).json()["sessions"]
    assert len(sessions) == 3

    turns = client.get(f"/profiles/{pid}/turns", headers=_auth(raw)).json()["turns"]
    assert {t["session_id"] for t in turns} == {str(sid_real), str(sid_legacy)}
    turns = client.get(f"/profiles/{pid}/turns", params={"include_diagnostic": "true"},
                       headers=_auth(raw)).json()["turns"]
    assert len(turns) == 3

    activity = client.get(f"/profiles/{pid}/activity", headers=_auth(raw)).json()
    assert sum(d["sessions"] for d in activity["days"]) == 2
    activity = client.get(f"/profiles/{pid}/activity",
                          params={"include_diagnostic": "true"},
                          headers=_auth(raw)).json()
    assert sum(d["sessions"] for d in activity["days"]) == 3
