"""Admin provisioning over HTTP: accounts, profiles, links, and device tokens — the same
operations as the `python -m robin.admin` CLI (which remains the bootstrap path for the
first admin account). Every route requires an is_admin dashboard token.

Same rules as everywhere else in robin/api: explicit Pydantic response models, request
bodies with extra='forbid', and no response shape ever carries password material or a
token hash. The raw device token appears exactly once, in the issuance response."""
import datetime
import zoneinfo

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from robin.auth.deps import require_admin
from robin.auth.passwords import hash_password
from robin.auth.tokens import issue_token, revoke_token
from robin.db import get_session
from robin.db.models import Account, AccountProfile, AuthToken, Profile

router = APIRouter(tags=["admin"], dependencies=[Depends(require_admin)])


# ---------------------------------------------------------------------------
# Accounts

class AccountCreate(BaseModel):
    model_config = {"extra": "forbid"}

    email: str = Field(min_length=3)
    password: str = Field(min_length=8)
    display_name: str = Field(min_length=1)
    is_admin: bool = False


class AccountResponse(BaseModel):
    id: int
    email: str
    display_name: str
    is_admin: bool
    created_at: datetime.datetime


class AccountListResponse(BaseModel):
    accounts: list[AccountResponse]


def _account_response(a: Account) -> AccountResponse:
    return AccountResponse(id=a.id, email=a.email, display_name=a.display_name,
                           is_admin=a.is_admin, created_at=a.created_at)


@router.post("/accounts", response_model=AccountResponse, status_code=201)
async def create_account(body: AccountCreate,
                         session: AsyncSession = Depends(get_session)):
    account = Account(email=body.email, password_hash=hash_password(body.password),
                      display_name=body.display_name, is_admin=body.is_admin)
    session.add(account)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()    # rollback before raising, or the session is poisoned
        raise HTTPException(status_code=409, detail="email already registered")
    return _account_response(account)


@router.get("/accounts", response_model=AccountListResponse)
async def list_accounts(session: AsyncSession = Depends(get_session)):
    rows = (await session.execute(select(Account).order_by(Account.id))).scalars().all()
    return AccountListResponse(accounts=[_account_response(a) for a in rows])


# ---------------------------------------------------------------------------
# Profiles

class ProfileCreate(BaseModel):
    model_config = {"extra": "forbid"}

    display_name: str = Field(min_length=1)
    timezone: str | None = None
    voice: str | None = Field(default=None, min_length=1)
    speech_rate: float | None = Field(default=None, gt=0.0, le=3.0)
    context: dict | None = None

    @field_validator("timezone")
    @classmethod
    def _iana(cls, v):
        if v is not None:
            try:
                zoneinfo.ZoneInfo(v)
            except (zoneinfo.ZoneInfoNotFoundError, ValueError):
                raise ValueError(f"not an IANA timezone name: {v!r}")
        return v


class AdminProfileResponse(BaseModel):
    """A profile as the admin created it — no `role` field: provisioning is not a link."""
    id: int
    display_name: str
    timezone: str
    voice: str
    speech_rate: float
    context: dict
    active: bool
    created_at: datetime.datetime
    updated_at: datetime.datetime


@router.post("/profiles", response_model=AdminProfileResponse, status_code=201)
async def create_profile(body: ProfileCreate,
                         session: AsyncSession = Depends(get_session)):
    profile = Profile(display_name=body.display_name)
    for field in ("timezone", "voice", "speech_rate", "context"):
        value = getattr(body, field)
        if value is not None:
            setattr(profile, field, value)
    session.add(profile)
    await session.commit()
    await session.refresh(profile)      # server defaults for the fields not provided
    return AdminProfileResponse(
        id=profile.id, display_name=profile.display_name, timezone=profile.timezone,
        voice=profile.voice, speech_rate=profile.speech_rate, context=profile.context,
        active=profile.active, created_at=profile.created_at, updated_at=profile.updated_at)


# ---------------------------------------------------------------------------
# Account <-> profile links

class LinkCreate(BaseModel):
    model_config = {"extra": "forbid"}

    account_id: int
    role: str = "owner"

    @field_validator("role")
    @classmethod
    def _role(cls, v):
        if v not in ("owner", "viewer"):
            raise ValueError("role must be 'owner' or 'viewer'")
        return v


class LinkResponse(BaseModel):
    account_id: int
    profile_id: int
    role: str
    created_at: datetime.datetime


class LinkListResponse(BaseModel):
    links: list[LinkResponse]


class UnlinkResponse(BaseModel):
    deleted: bool


@router.get("/profiles/{profile_id}/links", response_model=LinkListResponse)
async def list_links(profile_id: int, session: AsyncSession = Depends(get_session)):
    if await session.get(Profile, profile_id) is None:
        raise HTTPException(status_code=404, detail="profile not found")
    rows = (await session.execute(
        select(AccountProfile).where(AccountProfile.profile_id == profile_id)
        .order_by(AccountProfile.account_id)
    )).scalars().all()
    return LinkListResponse(links=[
        LinkResponse(account_id=l.account_id, profile_id=l.profile_id, role=l.role,
                     created_at=l.created_at) for l in rows])


@router.post("/profiles/{profile_id}/links", response_model=LinkResponse, status_code=201)
async def create_link(profile_id: int, body: LinkCreate,
                      session: AsyncSession = Depends(get_session)):
    if await session.get(Profile, profile_id) is None:
        raise HTTPException(status_code=404, detail="profile not found")
    if await session.get(Account, body.account_id) is None:
        raise HTTPException(status_code=404, detail="account not found")
    link = AccountProfile(account_id=body.account_id, profile_id=profile_id, role=body.role)
    session.add(link)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status_code=409, detail="already linked")
    await session.refresh(link)
    return LinkResponse(account_id=link.account_id, profile_id=link.profile_id,
                        role=link.role, created_at=link.created_at)


@router.delete("/profiles/{profile_id}/links/{account_id}", response_model=UnlinkResponse)
async def delete_link(profile_id: int, account_id: int,
                      session: AsyncSession = Depends(get_session)):
    link = await session.get(AccountProfile, (account_id, profile_id))
    if link is None:
        raise HTTPException(status_code=404, detail="link not found")
    await session.delete(link)
    await session.commit()
    return UnlinkResponse(deleted=True)


# ---------------------------------------------------------------------------
# Device tokens

class DeviceTokenCreate(BaseModel):
    model_config = {"extra": "forbid"}

    label: str = Field(min_length=1)


class DeviceTokenIssueResponse(BaseModel):
    token: str                      # the raw device token; shown once, not recoverable
    token_id: int
    profile_id: int
    label: str
    created_at: datetime.datetime


class TokenInfo(BaseModel):
    id: int
    kind: str
    account_id: int | None
    profile_id: int | None
    label: str | None
    last_used_at: datetime.datetime | None
    revoked_at: datetime.datetime | None
    created_at: datetime.datetime


class TokenListResponse(BaseModel):
    tokens: list[TokenInfo]


class RevokeResponse(BaseModel):
    revoked: bool


@router.post("/profiles/{profile_id}/device-tokens",
             response_model=DeviceTokenIssueResponse, status_code=201)
async def issue_device_token(profile_id: int, body: DeviceTokenCreate,
                             session: AsyncSession = Depends(get_session)):
    if await session.get(Profile, profile_id) is None:
        raise HTTPException(status_code=404, detail="profile not found")
    raw, row = await issue_token(session, kind="device", profile_id=profile_id,
                                 label=body.label)
    await session.commit()
    return DeviceTokenIssueResponse(token=raw, token_id=row.id, profile_id=profile_id,
                                    label=row.label, created_at=row.created_at)


@router.get("/tokens", response_model=TokenListResponse)
async def list_tokens(profile_id: int | None = Query(default=None),
                      include_revoked: bool = Query(default=False),
                      session: AsyncSession = Depends(get_session)):
    q = select(AuthToken).order_by(AuthToken.id)
    if profile_id is not None:
        q = q.where(AuthToken.profile_id == profile_id)
    if not include_revoked:
        q = q.where(AuthToken.revoked_at.is_(None))
    rows = (await session.execute(q)).scalars().all()
    return TokenListResponse(tokens=[
        TokenInfo(id=t.id, kind=t.kind, account_id=t.account_id, profile_id=t.profile_id,
                  label=t.label, last_used_at=t.last_used_at, revoked_at=t.revoked_at,
                  created_at=t.created_at) for t in rows])


@router.post("/tokens/{token_id}/revoke", response_model=RevokeResponse)
async def revoke_token_endpoint(token_id: int,
                                session: AsyncSession = Depends(get_session)):
    if not await revoke_token(session, token_id):
        raise HTTPException(status_code=404, detail="token not found")
    return RevokeResponse(revoked=True)
