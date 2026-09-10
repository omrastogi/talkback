"""Opaque token lifecycle: issue, verify, revoke.

The raw token exists in exactly two places, ever: the issuing response and the client's
storage. The database holds only its SHA-256. A lost device token is revoked and reissued,
not recovered.
"""
import datetime
import hashlib
import secrets

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from robin.db.models import AuthToken

# verify() updates last_used_at, but not more than once per this window: the voice socket
# would otherwise pay a write per connection health-check for a column that only needs to
# answer "has this tablet been alive lately".
LAST_USED_THROTTLE = datetime.timedelta(minutes=5)


def hash_token(raw: str) -> bytes:
    return hashlib.sha256(raw.encode()).digest()


async def issue_token(session: AsyncSession, *, kind: str,
                      account_id: int | None = None, profile_id: int | None = None,
                      label: str | None = None) -> tuple[str, AuthToken]:
    """Create a token row and return (raw_token, row). The caller shows raw_token once."""
    raw = secrets.token_urlsafe(32)
    row = AuthToken(token_hash=hash_token(raw), kind=kind,
                    account_id=account_id, profile_id=profile_id, label=label)
    session.add(row)
    await session.flush()
    return raw, row


async def verify_token(session: AsyncSession, raw: str, *,
                       kind: str | None = None) -> AuthToken | None:
    """Hash the presented token and look it up. None when unknown, revoked, or (if `kind`
    is given) of the wrong kind. Touches last_used_at, throttled to LAST_USED_THROTTLE."""
    if not raw:
        return None
    row = (await session.execute(
        select(AuthToken).where(AuthToken.token_hash == hash_token(raw))
    )).scalar_one_or_none()
    if row is None or row.revoked_at is not None:
        return None
    if kind is not None and row.kind != kind:
        return None
    now = datetime.datetime.now(datetime.timezone.utc)
    if row.last_used_at is None or now - row.last_used_at > LAST_USED_THROTTLE:
        row.last_used_at = now
        await session.commit()
    return row


async def revoke_token(session: AsyncSession, token_id: int) -> bool:
    row = await session.get(AuthToken, token_id)
    if row is None:
        return False
    if row.revoked_at is None:
        row.revoked_at = datetime.datetime.now(datetime.timezone.utc)
        await session.commit()
    return True
