"""Async engine + session factory, configured solely from DATABASE_URL.

No default URL: pointing code at a real host by accident is exactly the kind of quiet
misconfiguration this refuses to allow. `config.load_env()` runs first so `.env` works the
same here as everywhere else in the repo.
"""
import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import config

_engine = None
_sessionmaker = None


def database_url() -> str:
    config.load_env()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Set it in .env or the environment, e.g. "
            "postgresql+asyncpg://user@127.0.0.1:5433/robin"
        )
    return url


def get_engine():
    global _engine, _sessionmaker
    if _engine is None:
        _engine = create_async_engine(database_url(), pool_pre_ping=True)
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    get_engine()
    return _sessionmaker


async def get_session():
    """FastAPI dependency: one AsyncSession per request, committed by the endpoint."""
    async with get_sessionmaker()() as session:
        yield session
