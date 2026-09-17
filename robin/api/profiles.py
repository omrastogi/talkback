"""Profile endpoints for the dashboard. Every response body is an explicit Pydantic model
listing its fields — there is deliberately no generic SQLAlchemy-to-dict helper anywhere in
this codebase (RECOVER's as_dict() leaked password hashes through exactly that shortcut),
and no schema here expands a relationship.

Unlinked profiles return 404, not 403: the endpoint must not confirm that a profile the
caller cannot see exists at all.
"""
import datetime
import uuid
import zoneinfo

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Date, cast, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from robin.auth.deps import account_profile_role, require_dashboard
from robin.db import get_session
from robin.db.models import Account, AccountProfile, AuthToken, ConversationTurn, Profile

router = APIRouter(prefix="/profiles", tags=["profiles"])

# Device tokens whose label starts with this mark diagnostic sessions (the dashboard's
# Live page issues them as "diagnostic: browser — <admin email>"). Their turns persist
# like any other — same protocol as the tablet — but the history/activity endpoints
# exclude them unless include_diagnostic=true, so test sessions don't pollute real data.
DIAGNOSTIC_LABEL_PREFIX = "diagnostic"


def _exclude_diagnostic_clause():
    """WHERE clause keeping only non-diagnostic turns. Rows with no token provenance
    (pre-migration history, proactive paths) are real data and are kept."""
    diagnostic_token_ids = select(AuthToken.id).where(
        AuthToken.label.ilike(f"{DIAGNOSTIC_LABEL_PREFIX}%"))
    return or_(ConversationTurn.auth_token_id.is_(None),
               ConversationTurn.auth_token_id.not_in(diagnostic_token_ids))


class ProfileResponse(BaseModel):
    id: int
    display_name: str
    timezone: str
    voice: str
    speech_rate: float
    context: dict
    active: bool
    role: str                       # the CALLER's role: "owner" | "viewer", or "admin"
                                    # when an admin reaches a profile they are not linked to
    created_at: datetime.datetime
    updated_at: datetime.datetime


class ProfileListResponse(BaseModel):
    profiles: list[ProfileResponse]


class ProfilePatch(BaseModel):
    """The five mutable fields, all optional; anything else is rejected by extra='forbid'."""
    model_config = {"extra": "forbid"}

    display_name: str | None = Field(default=None, min_length=1)
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


class SessionSummary(BaseModel):
    session_id: uuid.UUID
    started_at: datetime.datetime
    last_at: datetime.datetime
    turn_count: int
    sources: list[str]


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


class ActivityDay(BaseModel):
    day: datetime.date
    sessions: int
    user_turns: int
    assistant_turns: int
    proactive_turns: int
    voice_turns: int
    avg_latency_ms: float | None


class ActivityResponse(BaseModel):
    timezone: str                   # days are bucketed in the profile's timezone
    days: list[ActivityDay]
    last_active_at: datetime.datetime | None


class TurnResponse(BaseModel):
    id: int
    session_id: uuid.UUID
    turn_index: int
    role: str
    content: str
    source: str
    latency_ms: int | None
    meta: dict
    created_at: datetime.datetime


class TurnListResponse(BaseModel):
    turns: list[TurnResponse]


def _profile_response(p: Profile, role: str) -> ProfileResponse:
    return ProfileResponse(id=p.id, display_name=p.display_name, timezone=p.timezone,
                           voice=p.voice, speech_rate=p.speech_rate, context=p.context,
                           active=p.active, role=role,
                           created_at=p.created_at, updated_at=p.updated_at)


async def _linked_profile_or_404(session: AsyncSession, account: Account,
                                 profile_id: int) -> tuple[Profile, str]:
    role = await account_profile_role(session, account.id, profile_id)
    if role is None:
        if not account.is_admin:
            raise HTTPException(status_code=404, detail="profile not found")
        role = "admin"              # admins see every profile; only true absence is a 404
    profile = await session.get(Profile, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="profile not found")
    return profile, role


@router.get("", response_model=ProfileListResponse)
async def list_profiles(account: Account = Depends(require_dashboard),
                        session: AsyncSession = Depends(get_session)):
    if account.is_admin:
        # Admins list every profile; role is their link's role where one exists.
        rows = (await session.execute(
            select(Profile, AccountProfile.role)
            .join(AccountProfile,
                  (AccountProfile.profile_id == Profile.id)
                  & (AccountProfile.account_id == account.id),
                  isouter=True)
            .order_by(Profile.id)
        )).all()
        return ProfileListResponse(
            profiles=[_profile_response(p, role or "admin") for p, role in rows])
    rows = (await session.execute(
        select(Profile, AccountProfile.role)
        .join(AccountProfile, AccountProfile.profile_id == Profile.id)
        .where(AccountProfile.account_id == account.id)
        .order_by(Profile.id)
    )).all()
    return ProfileListResponse(profiles=[_profile_response(p, role) for p, role in rows])


@router.get("/{profile_id}", response_model=ProfileResponse)
async def get_profile(profile_id: int, account: Account = Depends(require_dashboard),
                      session: AsyncSession = Depends(get_session)):
    profile, role = await _linked_profile_or_404(session, account, profile_id)
    return _profile_response(profile, role)


@router.patch("/{profile_id}", response_model=ProfileResponse)
async def patch_profile(profile_id: int, body: ProfilePatch,
                        account: Account = Depends(require_dashboard),
                        session: AsyncSession = Depends(get_session)):
    profile, role = await _linked_profile_or_404(session, account, profile_id)
    if role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="owner role required")
    changes = body.model_dump(exclude_unset=True, exclude_none=True)
    for field, value in changes.items():
        setattr(profile, field, value)
    if changes:
        profile.updated_at = datetime.datetime.now(datetime.timezone.utc)
        await session.commit()
        await session.refresh(profile)
    return _profile_response(profile, role)


@router.get("/{profile_id}/sessions", response_model=SessionListResponse)
async def list_sessions(profile_id: int,
                        before: datetime.datetime | None = Query(default=None),
                        limit: int = Query(default=50, ge=1, le=200),
                        include_diagnostic: bool = Query(default=False),
                        account: Account = Depends(require_dashboard),
                        session: AsyncSession = Depends(get_session)):
    """Conversations for one profile as (session_id, timing, turn count) summaries,
    most recently active first; page with `before` (a last_at cursor)."""
    await _linked_profile_or_404(session, account, profile_id)
    last_at = func.max(ConversationTurn.created_at)
    q = (select(ConversationTurn.session_id,
                func.min(ConversationTurn.created_at).label("started_at"),
                last_at.label("last_at"),
                func.count().label("turn_count"),
                func.array_agg(ConversationTurn.source.distinct()).label("sources"))
         .where(ConversationTurn.profile_id == profile_id)
         .group_by(ConversationTurn.session_id))
    if not include_diagnostic:
        q = q.where(_exclude_diagnostic_clause())
    if before is not None:
        q = q.having(last_at < before)
    q = q.order_by(last_at.desc()).limit(limit)
    rows = (await session.execute(q)).all()
    return SessionListResponse(sessions=[
        SessionSummary(session_id=r.session_id, started_at=r.started_at, last_at=r.last_at,
                       turn_count=r.turn_count, sources=sorted(r.sources))
        for r in rows
    ])


@router.get("/{profile_id}/activity", response_model=ActivityResponse)
async def activity(profile_id: int,
                   date_from: datetime.date | None = Query(default=None),
                   date_to: datetime.date | None = Query(default=None),
                   include_diagnostic: bool = Query(default=False),
                   account: Account = Depends(require_dashboard),
                   session: AsyncSession = Depends(get_session)):
    """Per-day usage derived from conversation_turn, bucketed in the profile's timezone
    (a "day" is only meaningful as the person's local day). Defaults to the last 30 days.
    last_active_at is unbounded by the date range."""
    profile, _ = await _linked_profile_or_404(session, account, profile_id)
    tz = zoneinfo.ZoneInfo(profile.timezone)
    if date_to is None:
        date_to = datetime.datetime.now(tz).date()
    if date_from is None:
        date_from = date_to - datetime.timedelta(days=29)
    if date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from is after date_to")

    day = cast(func.timezone(profile.timezone, ConversationTurn.created_at), Date)
    q = (select(day.label("day"),
                func.count(ConversationTurn.session_id.distinct()).label("sessions"),
                func.count().filter(ConversationTurn.role == "user").label("user_turns"),
                func.count().filter(ConversationTurn.role == "assistant")
                    .label("assistant_turns"),
                func.count().filter(ConversationTurn.source == "proactive")
                    .label("proactive_turns"),
                func.count().filter(ConversationTurn.source == "voice").label("voice_turns"),
                func.avg(ConversationTurn.latency_ms)
                    .filter(ConversationTurn.role == "assistant").label("avg_latency_ms"))
         .where(ConversationTurn.profile_id == profile_id,
                day.between(date_from, date_to))
         .group_by(day).order_by(day))
    last_active_q = (select(func.max(ConversationTurn.created_at))
                     .where(ConversationTurn.profile_id == profile_id))
    if not include_diagnostic:
        q = q.where(_exclude_diagnostic_clause())
        last_active_q = last_active_q.where(_exclude_diagnostic_clause())
    rows = (await session.execute(q)).all()
    last_active_at = (await session.execute(last_active_q)).scalar_one_or_none()
    return ActivityResponse(timezone=profile.timezone, days=[
        ActivityDay(day=r.day, sessions=r.sessions, user_turns=r.user_turns,
                    assistant_turns=r.assistant_turns, proactive_turns=r.proactive_turns,
                    voice_turns=r.voice_turns,
                    avg_latency_ms=float(r.avg_latency_ms) if r.avg_latency_ms is not None
                    else None)
        for r in rows
    ], last_active_at=last_active_at)


@router.get("/{profile_id}/turns", response_model=TurnListResponse)
async def list_turns(profile_id: int,
                     session_id: uuid.UUID | None = Query(default=None),
                     before: datetime.datetime | None = Query(default=None),
                     limit: int = Query(default=100, ge=1, le=500),
                     include_diagnostic: bool = Query(default=False),
                     account: Account = Depends(require_dashboard),
                     session: AsyncSession = Depends(get_session)):
    """Turns for one profile, newest first; page with `before` (a created_at cursor) or
    narrow to one conversation with `session_id`."""
    await _linked_profile_or_404(session, account, profile_id)
    q = select(ConversationTurn).where(ConversationTurn.profile_id == profile_id)
    if not include_diagnostic:
        q = q.where(_exclude_diagnostic_clause())
    if session_id is not None:
        q = q.where(ConversationTurn.session_id == session_id)
    if before is not None:
        q = q.where(ConversationTurn.created_at < before)
    q = q.order_by(ConversationTurn.created_at.desc(),
                   ConversationTurn.turn_index.desc()).limit(limit)
    turns = (await session.execute(q)).scalars().all()
    return TurnListResponse(turns=[
        TurnResponse(id=t.id, session_id=t.session_id, turn_index=t.turn_index, role=t.role,
                     content=t.content, source=t.source, latency_ms=t.latency_ms,
                     meta=t.meta, created_at=t.created_at)
        for t in turns
    ])
