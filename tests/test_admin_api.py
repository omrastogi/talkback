"""Admin provisioning endpoints: every route requires is_admin, the full CLI-equivalent
account -> profile -> link -> device-token flow works over HTTP, and conflicts are 409s."""
import asyncio

import pytest

import robin.db as rdb
from robin.auth.tokens import verify_token
from tests.helpers import (PASSWORD, dashboard_token, device_token, link,
                           make_account, make_profile)


def _run(coro):
    return asyncio.run(coro)


def _auth(raw):
    return {"Authorization": f"Bearer {raw}"}


def _admin(client):
    aid = _run(make_account("admin@example.org", is_admin=True))
    raw, _ = _run(dashboard_token(aid))
    return aid, raw


ADMIN_CALLS = [
    ("post", "/accounts", {"json": {"email": "x@example.org", "password": "longenough",
                                    "display_name": "X"}}),
    ("get", "/accounts", {}),
    ("post", "/profiles", {"json": {"display_name": "X"}}),
    ("get", "/profiles/1/links", {}),
    ("post", "/profiles/1/links", {"json": {"account_id": 1}}),
    ("delete", "/profiles/1/links/1", {}),
    ("post", "/profiles/1/device-tokens", {"json": {"label": "tab"}}),
    ("get", "/tokens", {}),
    ("post", "/tokens/1/revoke", {}),
]


def test_non_admin_gets_403_on_every_admin_route(client):
    aid = _run(make_account())
    raw, _ = _run(dashboard_token(aid))
    for method, path, kwargs in ADMIN_CALLS:
        r = getattr(client, method)(path, headers=_auth(raw), **kwargs)
        assert r.status_code == 403, f"{method.upper()} {path}: {r.status_code}"


def test_provisioning_flow(client):
    """The CLI's typical new-household sequence, over HTTP: create account, create
    profile, link, issue a device token, and prove each piece actually works."""
    _, admin_raw = _admin(client)

    r = client.post("/accounts", headers=_auth(admin_raw),
                    json={"email": "new-cp@example.org", "password": PASSWORD,
                          "display_name": "New Care Partner"})
    assert r.status_code == 201
    account = r.json()
    assert account["is_admin"] is False

    r = client.post("/profiles", headers=_auth(admin_raw),
                    json={"display_name": "Margaret", "timezone": "America/Chicago",
                          "speech_rate": 0.9})
    assert r.status_code == 201
    profile = r.json()
    assert profile["timezone"] == "America/Chicago"
    assert profile["voice"] == "af_heart"          # server default filled in

    r = client.post(f"/profiles/{profile['id']}/links", headers=_auth(admin_raw),
                    json={"account_id": account["id"], "role": "owner"})
    assert r.status_code == 201

    # The new (non-admin) account logs in and sees exactly its linked profile.
    login = client.post("/auth/login", json={"email": "new-cp@example.org",
                                             "password": PASSWORD})
    assert login.status_code == 200
    cp_raw = login.json()["token"]
    profiles = client.get("/profiles", headers=_auth(cp_raw)).json()["profiles"]
    assert [p["id"] for p in profiles] == [profile["id"]]
    assert profiles[0]["role"] == "owner"

    # Device token: issuance returns the raw value once, and it verifies as kind=device.
    r = client.post(f"/profiles/{profile['id']}/device-tokens", headers=_auth(admin_raw),
                    json={"label": "Tab A9 living room"})
    assert r.status_code == 201
    issued = r.json()

    async def check():
        async with rdb._sessionmaker() as s:
            row = await verify_token(s, issued["token"], kind="device")
            assert row is not None and row.profile_id == profile["id"]
    _run(check())

    links = client.get(f"/profiles/{profile['id']}/links",
                       headers=_auth(admin_raw)).json()["links"]
    assert links == [{"account_id": account["id"], "profile_id": profile["id"],
                      "role": "owner", "created_at": links[0]["created_at"]}]


def test_duplicate_email_is_409(client):
    _, admin_raw = _admin(client)
    body = {"email": "dup@example.org", "password": PASSWORD, "display_name": "D"}
    assert client.post("/accounts", headers=_auth(admin_raw), json=body).status_code == 201
    body["email"] = "DUP@Example.org"              # citext: same address
    assert client.post("/accounts", headers=_auth(admin_raw), json=body).status_code == 409


def test_link_conflicts_and_missing(client):
    _, admin_raw = _admin(client)
    aid = _run(make_account("cp2@example.org"))
    pid = _run(make_profile())

    assert client.post("/profiles/999999/links", headers=_auth(admin_raw),
                       json={"account_id": aid}).status_code == 404
    assert client.post(f"/profiles/{pid}/links", headers=_auth(admin_raw),
                       json={"account_id": 999999}).status_code == 404
    assert client.post(f"/profiles/{pid}/links", headers=_auth(admin_raw),
                       json={"account_id": aid}).status_code == 201
    assert client.post(f"/profiles/{pid}/links", headers=_auth(admin_raw),
                       json={"account_id": aid}).status_code == 409
    assert client.post(f"/profiles/{pid}/links", headers=_auth(admin_raw),
                       json={"account_id": aid, "role": "boss"}).status_code == 422

    assert client.delete(f"/profiles/{pid}/links/{aid}",
                         headers=_auth(admin_raw)).json() == {"deleted": True}
    assert client.delete(f"/profiles/{pid}/links/{aid}",
                         headers=_auth(admin_raw)).status_code == 404


def test_token_list_and_revoke_lifecycle(client):
    _, admin_raw = _admin(client)
    pid = _run(make_profile())

    issued = client.post(f"/profiles/{pid}/device-tokens", headers=_auth(admin_raw),
                         json={"label": "kitchen tablet"}).json()
    tid = issued["token_id"]

    tokens = client.get("/tokens", params={"profile_id": pid},
                        headers=_auth(admin_raw)).json()["tokens"]
    assert [t["id"] for t in tokens] == [tid]
    assert tokens[0]["revoked_at"] is None and tokens[0]["label"] == "kitchen tablet"

    assert client.post(f"/tokens/{tid}/revoke",
                       headers=_auth(admin_raw)).json() == {"revoked": True}
    assert client.post("/tokens/999999/revoke",
                       headers=_auth(admin_raw)).status_code == 404

    # Revoked tokens are hidden by default, shown with include_revoked, and dead for auth.
    assert client.get("/tokens", params={"profile_id": pid},
                      headers=_auth(admin_raw)).json()["tokens"] == []
    revoked = client.get("/tokens", params={"profile_id": pid, "include_revoked": True},
                         headers=_auth(admin_raw)).json()["tokens"]
    assert revoked[0]["revoked_at"] is not None

    async def check_dead():
        async with rdb._sessionmaker() as s:
            assert await verify_token(s, issued["token"], kind="device") is None
    _run(check_dead())


def test_invalid_profile_create_bodies(client):
    _, admin_raw = _admin(client)
    assert client.post("/profiles", headers=_auth(admin_raw),
                       json={"display_name": ""}).status_code == 422
    assert client.post("/profiles", headers=_auth(admin_raw),
                       json={"display_name": "M",
                             "timezone": "Mars/Olympus_Mons"}).status_code == 422
    assert client.post("/profiles", headers=_auth(admin_raw),
                       json={"display_name": "M", "speech_rate": 9}).status_code == 422
    assert client.post("/profiles", headers=_auth(admin_raw),
                       json={"display_name": "M", "active": False}).status_code == 422


def test_no_admin_response_contains_password_or_token_hash(client):
    """Extend the RECOVER-derived sweep to every admin endpoint. The token issuance
    response legitimately contains "token" but never the hash or a password."""
    _, admin_raw = _admin(client)
    pid = _run(make_profile())
    aid = _run(make_account("cp3@example.org"))

    responses = [
        client.post("/accounts", headers=_auth(admin_raw),
                    json={"email": "sweep@example.org", "password": PASSWORD,
                          "display_name": "S"}),
        client.get("/accounts", headers=_auth(admin_raw)),
        client.post("/profiles", headers=_auth(admin_raw), json={"display_name": "S"}),
        client.post(f"/profiles/{pid}/links", headers=_auth(admin_raw),
                    json={"account_id": aid}),
        client.get(f"/profiles/{pid}/links", headers=_auth(admin_raw)),
        client.post(f"/profiles/{pid}/device-tokens", headers=_auth(admin_raw),
                    json={"label": "sweep tablet"}),
        client.get("/tokens", headers=_auth(admin_raw)),
        client.get("/tokens", params={"include_revoked": True}, headers=_auth(admin_raw)),
        client.delete(f"/profiles/{pid}/links/{aid}", headers=_auth(admin_raw)),
    ]
    for r in responses:
        body = r.text.lower()
        assert "password" not in body, f"{r.request.method} {r.url}: 'password' in body"
        assert "token_hash" not in body, f"{r.request.method} {r.url}: 'token_hash' in body"
