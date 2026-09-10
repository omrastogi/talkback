import datetime

from sqlalchemy import select

import robin.db as rdb
from robin.auth.tokens import hash_token, revoke_token, verify_token
from robin.db.models import AuthToken
from tests.helpers import device_token, make_profile


async def test_token_round_trip():
    pid = await make_profile()
    raw, _ = await device_token(pid)
    async with rdb._sessionmaker() as s:
        row = await verify_token(s, raw)
    assert row is not None
    assert row.kind == "device"
    assert row.profile_id == pid
    assert row.last_used_at is not None            # touched on first successful verify


async def test_raw_token_not_stored_anywhere_in_row():
    pid = await make_profile()
    raw, token_id = await device_token(pid)
    async with rdb._sessionmaker() as s:
        row = await s.get(AuthToken, token_id)
    assert row.token_hash == hash_token(raw)
    for col in ("kind", "label"):
        assert raw not in (getattr(row, col) or "")
    assert raw.encode() != row.token_hash          # not the raw bytes either


async def test_unknown_token_fails():
    async with rdb._sessionmaker() as s:
        assert await verify_token(s, "no-such-token") is None


async def test_revoked_token_fails():
    pid = await make_profile()
    raw, token_id = await device_token(pid)
    async with rdb._sessionmaker() as s:
        assert await revoke_token(s, token_id)
    async with rdb._sessionmaker() as s:
        assert await verify_token(s, raw) is None


async def test_wrong_kind_rejected():
    pid = await make_profile()
    raw, _ = await device_token(pid)
    async with rdb._sessionmaker() as s:
        assert await verify_token(s, raw, kind="dashboard") is None


async def test_last_used_write_is_throttled():
    pid = await make_profile()
    raw, token_id = await device_token(pid)
    async with rdb._sessionmaker() as s:
        await verify_token(s, raw)
    async with rdb._sessionmaker() as s:
        first = (await s.get(AuthToken, token_id)).last_used_at
    async with rdb._sessionmaker() as s:
        await verify_token(s, raw)                 # within 5 minutes: no second write
    async with rdb._sessionmaker() as s:
        assert (await s.get(AuthToken, token_id)).last_used_at == first
