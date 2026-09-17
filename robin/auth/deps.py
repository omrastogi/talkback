"""FastAPI dependencies and the WebSocket device-auth helper.

This module must stay importable without server.py (which loads the STT/TTS models at
import): the WebSocket endpoints call authenticate_device_ws() from server.py, and the
test suite mounts the very same helper in a stub app.
"""
from fastapi import Depends, HTTPException, Request, WebSocket
from sqlalchemy.ext.asyncio import AsyncSession

from robin.auth.tokens import verify_token
from robin.db import get_session, get_sessionmaker
from robin.db.models import Account, AccountProfile, AuthToken, Profile

WS_CLOSE_UNAUTHORIZED = 4401


def _bearer(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return ""
    return auth[7:].strip()


async def require_dashboard(request: Request,
                            session: AsyncSession = Depends(get_session)) -> Account:
    """Authorization: Bearer <dashboard token> -> the Account, or 401."""
    token = await verify_token(session, _bearer(request), kind="dashboard")
    if token is None:
        raise HTTPException(status_code=401, detail="invalid or revoked token")
    account = await session.get(Account, token.account_id)
    if account is None:
        raise HTTPException(status_code=401, detail="invalid or revoked token")
    # Stash the token row so /auth/logout can revoke the exact credential presented.
    request.state.auth_token = token
    return account


async def require_device_profile(request: Request,
                                 session: AsyncSession = Depends(get_session)) -> Profile:
    """Authorization: Bearer <device token> -> the bound active Profile, or 401.

    The HTTP twin of authenticate_device_ws, for the tablet's own enrollment calls: the
    token's binding fixes the profile — nothing the client sends can select another."""
    token = await verify_token(session, _bearer(request), kind="device")
    if token is None:
        raise HTTPException(status_code=401, detail="invalid or revoked token")
    profile = await session.get(Profile, token.profile_id)
    if profile is None or not profile.active:
        raise HTTPException(status_code=401, detail="invalid or revoked token")
    return profile


async def require_admin(account: Account = Depends(require_dashboard)) -> Account:
    """require_dashboard plus is_admin. 403, not 404: admin routes' existence is public
    (the 404-not-403 rule protects profile existence, not route existence)."""
    if not account.is_admin:
        raise HTTPException(status_code=403, detail="admin required")
    return account


async def account_profile_role(session: AsyncSession, account_id: int,
                               profile_id: int) -> str | None:
    """The caller's role on a profile, or None when unlinked. Callers turn None into 404,
    not 403: an unlinked profile's existence is not confirmed."""
    link = await session.get(AccountProfile, (account_id, profile_id))
    return link.role if link else None


def ws_presented_token(ws: WebSocket) -> str:
    """The device token as presented on the WebSocket handshake.

    Sec-WebSocket-Protocol only — the same transport (and the same reasoning) as the
    shared-key auth this replaces: a query string lands verbatim in nginx's and uvicorn's
    access logs on every connection, permanently persisting the credential on disk, and a
    device token is a longer-lived secret than the old shared key ever was."""
    return ws.headers.get("sec-websocket-protocol", "").strip()


async def authenticate_device_ws(raw_token: str) -> tuple[Profile, AuthToken] | None:
    """Voice-socket auth: raw token -> (bound active Profile, the token row), else None.

    Rules, in order: the token must exist and be unrevoked; it must be kind='device' (a
    dashboard token is not valid for the voice socket); its profile must exist and be
    active. profile_id comes from the token row alone — nothing the client sends can
    select a profile. The token row is returned so turns can record which device
    produced them (conversation_turn.auth_token_id)."""
    async with get_sessionmaker()() as session:
        token = await verify_token(session, raw_token, kind="device")
        if token is None:
            return None
        profile = await session.get(Profile, token.profile_id)
        if profile is None or not profile.active:
            return None
        return profile, token
