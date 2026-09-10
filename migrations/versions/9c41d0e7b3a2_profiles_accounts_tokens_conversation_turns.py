"""profiles, accounts, tokens, conversation turns

Revision ID: 9c41d0e7b3a2
Revises:
Create Date: 2026-09-08
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "9c41d0e7b3a2"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS citext")

    op.create_table(
        "profile",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("display_name", sa.Text, nullable=False),
        sa.Column("timezone", sa.Text, nullable=False, server_default="America/New_York"),
        sa.Column("voice", sa.Text, nullable=False, server_default="af_heart"),
        sa.Column("speech_rate", sa.REAL, nullable=False, server_default=sa.text("0.85")),
        sa.Column("context", postgresql.JSONB, nullable=False,
                  server_default=sa.text("'{}'::jsonb")),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )

    op.create_table(
        "account",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("email", postgresql.CITEXT, nullable=False, unique=True),
        sa.Column("password_hash", sa.Text, nullable=False),
        sa.Column("display_name", sa.Text, nullable=False),
        sa.Column("is_admin", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )

    op.create_table(
        "account_profile",
        sa.Column("account_id", sa.BigInteger,
                  sa.ForeignKey("account.id", ondelete="CASCADE"),
                  primary_key=True, nullable=False),
        sa.Column("profile_id", sa.BigInteger,
                  sa.ForeignKey("profile.id", ondelete="CASCADE"),
                  primary_key=True, nullable=False),
        sa.Column("role", sa.Text, nullable=False, server_default="owner"),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )

    op.create_table(
        "auth_token",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("token_hash", sa.LargeBinary, nullable=False, unique=True),
        sa.Column("kind", sa.Text, nullable=False),
        sa.Column("account_id", sa.BigInteger,
                  sa.ForeignKey("account.id", ondelete="CASCADE"), nullable=True),
        sa.Column("profile_id", sa.BigInteger,
                  sa.ForeignKey("profile.id", ondelete="CASCADE"), nullable=True),
        sa.Column("label", sa.Text, nullable=True),
        sa.Column("last_used_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.CheckConstraint(
            "(kind = 'device'    AND profile_id IS NOT NULL AND account_id IS NULL) OR "
            "(kind = 'dashboard' AND account_id IS NOT NULL AND profile_id IS NULL)",
            name="auth_token_one_subject",
        ),
    )
    op.create_index("ix_auth_token_profile_active", "auth_token", ["profile_id"],
                    postgresql_where=sa.text("revoked_at IS NULL"))

    op.create_table(
        "conversation_turn",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("profile_id", sa.BigInteger,
                  sa.ForeignKey("profile.id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("turn_index", sa.Integer, nullable=False),
        sa.Column("role", sa.Text, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("source", sa.Text, nullable=False, server_default="voice"),
        sa.Column("speaker_id", sa.Text, nullable=True),
        sa.Column("latency_ms", sa.Integer, nullable=True),
        sa.Column("meta", postgresql.JSONB, nullable=False,
                  server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.UniqueConstraint("session_id", "turn_index",
                            name="uq_conversation_turn_session_index"),
    )
    op.create_index("ix_conversation_turn_profile_created", "conversation_turn",
                    ["profile_id", sa.text("created_at DESC")])


def downgrade() -> None:
    op.drop_table("conversation_turn")
    op.drop_table("auth_token")
    op.drop_table("account_profile")
    op.drop_table("account")
    op.drop_table("profile")
    # The citext extension is left installed: dropping it would break anything else in the
    # database that adopted it after this migration ran.
