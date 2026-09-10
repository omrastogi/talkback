"""Test setup: a real Postgres database (TEST_DATABASE_URL), never SQLite — the prior
system was ported off SQLite and hit sequence and constraint problems that SQLite had
silently absorbed. The suite never imports server.py (which loads STT/TTS models onto the
GPU at import); WebSocket behavior is tested through a stub app that mounts the very same
robin.ws / robin.api functions the real endpoints call.

The engine is rebuilt with NullPool so connections never cross event loops: pytest-asyncio
tests, TestClient's portal thread, and asyncio.run() fixtures each get fresh connections.
"""
import asyncio
import os

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

import config

config.load_env()
TEST_URL = os.environ.get("TEST_DATABASE_URL")
if not TEST_URL:
    raise SystemExit("TEST_DATABASE_URL is not set (e.g. "
                     "postgresql+asyncpg://user@127.0.0.1:5433/robin_test)")
os.environ["DATABASE_URL"] = TEST_URL          # everything under robin/ sees the test DB

import robin.db as rdb                                             # noqa: E402
from robin.db.models import Base                                   # noqa: E402

rdb._engine = create_async_engine(TEST_URL, poolclass=NullPool)
rdb._sessionmaker = async_sessionmaker(rdb._engine, expire_on_commit=False)

TABLES = "conversation_turn, auth_token, account_profile, account, profile"


@pytest.fixture(scope="session", autouse=True)
def _schema():
    async def setup():
        async with rdb._engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS citext"))
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
    asyncio.run(setup())
    yield


@pytest.fixture(autouse=True)
def _clean():
    yield
    async def truncate():
        async with rdb._engine.begin() as conn:
            await conn.execute(text(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE"))
    asyncio.run(truncate())


@pytest.fixture
def db():
    """Per-test AsyncSession factory (call it, use as async context manager)."""
    return rdb._sessionmaker


@pytest.fixture(scope="session")
def app():
    """A minimal app mounting the real routers and a WS endpoint that uses the same
    bind_device_session helper server.py uses. The WS endpoint echoes what the server
    resolved — profile_id from the token, session_id minted at accept — after reading one
    client frame, so tests can prove a client-sent profile_id is ignored."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

    from robin.api.admin import router as admin_router
    from robin.api.auth import router as auth_router
    from robin.api.profiles import router as profiles_router
    from robin.ws import bind_device_session, persist_turn

    a = FastAPI()
    a.include_router(auth_router)
    a.include_router(profiles_router)
    a.include_router(admin_router)

    @a.websocket("/ws")
    async def ws_ep(ws: WebSocket):
        bound = await bind_device_session(ws)
        if bound is None:
            return
        try:
            payload = await ws.receive_json()      # e.g. a start frame naming a profile_id
        except WebSocketDisconnect:
            return
        if payload.get("speak"):                   # persist BEFORE replying, so the reply
            await persist_turn(bound, role="user", content=payload["speak"])
        await ws.send_json({"profile_id": bound.profile_id,   # frame proves the write landed
                            "session_id": str(bound.session_id),
                            "voice": bound.voice,
                            "client_sent_profile_id": payload.get("profile_id")})
        await ws.close()

    return a


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c
