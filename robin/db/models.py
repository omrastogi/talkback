"""The five tables. SQLAlchemy 2.0 declarative with Mapped[...] / mapped_column only.

Two deliberate absences, both traceable to the RECOVER review:
  - No relationship() attributes. Nothing here can be "expanded" into a response the way
    RECOVER's as_dict() expanded patient.users into password hashes. Endpoints query what
    they return, explicitly.
  - No serializer of any kind on these classes. Response shapes live in the Pydantic
    models next to each endpoint.

There is no session table: session_id on conversation_turn is a UUID the server mints at
WebSocket accept. Session boundaries are a server decision, never inferred from model
output (RECOVER cut sessions on a CONVERSATION_END keyword the LLM sometimes forgot).
"""
import datetime
import uuid

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    REAL,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import CITEXT, JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _ts():
    return mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=text("now()"))


class Profile(Base):
    """One row per person Robin talks to; the unit a device token is bound to."""
    __tablename__ = "profile"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    timezone: Mapped[str] = mapped_column(Text, nullable=False,
                                          server_default=text("'America/New_York'"))
    voice: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'af_heart'"))
    speech_rate: Mapped[float] = mapped_column(REAL, nullable=False, server_default=text("0.85"))
    context: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    created_at: Mapped[datetime.datetime] = _ts()
    updated_at: Mapped[datetime.datetime] = _ts()


class Account(Base):
    """A human who logs in: a care partner or a member of the research team."""
    __tablename__ = "account"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    email: Mapped[str] = mapped_column(CITEXT, nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    created_at: Mapped[datetime.datetime] = _ts()
    updated_at: Mapped[datetime.datetime] = _ts()


class AccountProfile(Base):
    """Many-to-many: a care partner may cover several profiles in a household."""
    __tablename__ = "account_profile"

    account_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("account.id", ondelete="CASCADE"), primary_key=True)
    profile_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("profile.id", ondelete="CASCADE"), primary_key=True)
    role: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'owner'"))
    created_at: Mapped[datetime.datetime] = _ts()


class AuthToken(Base):
    """Opaque random tokens, not JWTs: device tokens never expire and revocation needs a
    DB lookup per request regardless, so a signed self-describing token buys nothing.
    Only the SHA-256 of the raw token is stored; the raw value is shown once at issuance."""
    __tablename__ = "auth_token"
    __table_args__ = (
        CheckConstraint(
            "(kind = 'device'    AND profile_id IS NOT NULL AND account_id IS NULL) OR "
            "(kind = 'dashboard' AND account_id IS NOT NULL AND profile_id IS NULL)",
            name="auth_token_one_subject",
        ),
        Index("ix_auth_token_profile_active", "profile_id",
              postgresql_where=text("revoked_at IS NULL")),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    token_hash: Mapped[bytes] = mapped_column(LargeBinary, nullable=False, unique=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    account_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("account.id", ondelete="CASCADE"), nullable=True)
    profile_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("profile.id", ondelete="CASCADE"), nullable=True)
    label: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_used_at: Mapped[datetime.datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True)
    revoked_at: Mapped[datetime.datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True)
    created_at: Mapped[datetime.datetime] = _ts()


class ConversationTurn(Base):
    """One row per utterance. `content` is display text ("8:00 PM"); the spoken form from
    server.for_speech() is derived at synthesis time and never persisted."""
    __tablename__ = "conversation_turn"
    __table_args__ = (
        UniqueConstraint("session_id", "turn_index", name="uq_conversation_turn_session_index"),
        Index("ix_conversation_turn_profile_created",
              "profile_id", text("created_at DESC")),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    profile_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("profile.id", ondelete="CASCADE"), nullable=False)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)          # user | assistant | system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False,        # voice | proactive | tool
                                        server_default=text("'voice'"))
    speaker_id: Mapped[str | None] = mapped_column(Text, nullable=True)   # reserved, unpopulated
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    meta: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime.datetime] = _ts()
