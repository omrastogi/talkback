"""wake_model + wake_clip: per-profile wake-word personalization

wake_model: one row per trained head (merged LoRA on the shared base, plain ONNX blob,
~205 KB — small enough that bytea beats an object store here). At most one active per
profile via a partial unique index; retraining inserts and flips `active`, so lineage is
queryable. `threshold` rides with the blob because the two are one artifact: the head is
calibrated to an FA budget at that exact value (see oww-train LORA.md), and the WS
delivery sends them in the same meta frame.

wake_clip: the enrollment recordings those heads are trained from (dashboard-recorded
16 kHz mono WAVs). Kept, not cached: a head is welded to one base version, so a base
bump means retraining every profile from its stored clips.

Revision ID: c8b2e64a90f3
Revises: e4a8f17c25d9
Create Date: 2026-09-16
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP

revision = "c8b2e64a90f3"
down_revision = "e4a8f17c25d9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "wake_model",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("profile_id", sa.BigInteger,
                  sa.ForeignKey("profile.id", ondelete="CASCADE"), nullable=False),
        sa.Column("base_version", sa.Text, nullable=False),
        sa.Column("sha256", sa.Text, nullable=False),
        sa.Column("onnx", sa.LargeBinary, nullable=False),
        sa.Column("threshold", sa.REAL, nullable=False),
        sa.Column("manifest", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )
    op.create_index("uq_wake_model_profile_active", "wake_model", ["profile_id"],
                    unique=True, postgresql_where=sa.text("active"))
    op.create_table(
        "wake_clip",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("profile_id", sa.BigInteger,
                  sa.ForeignKey("profile.id", ondelete="CASCADE"), nullable=False),
        sa.Column("label", sa.Text, nullable=False),
        sa.Column("wav", sa.LargeBinary, nullable=False),
        sa.Column("duration_s", sa.REAL, nullable=False),
        sa.Column("sha256", sa.Text, nullable=False),
        sa.Column("created_at", TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.CheckConstraint("label IN ('positive', 'negative')", name="wake_clip_label"),
        sa.UniqueConstraint("profile_id", "sha256", name="uq_wake_clip_profile_sha"),
    )


def downgrade() -> None:
    op.drop_table("wake_clip")
    op.drop_table("wake_model")
