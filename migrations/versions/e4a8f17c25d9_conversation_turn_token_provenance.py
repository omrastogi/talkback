"""conversation turn token provenance

Which device token's connection produced each turn: nullable FK, SET NULL on token
delete so removing a credential never removes history. Nullable also covers every
pre-existing row and any future write path without a socket. Diagnostic (browser)
sessions are distinguished by their token's label prefix, and the dashboard queries
exclude them by default — see robin/api/profiles.py.

Revision ID: e4a8f17c25d9
Revises: 9c41d0e7b3a2
Create Date: 2026-09-10
"""
import sqlalchemy as sa
from alembic import op

revision = "e4a8f17c25d9"
down_revision = "9c41d0e7b3a2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("conversation_turn",
                  sa.Column("auth_token_id", sa.BigInteger, nullable=True))
    op.create_foreign_key("fk_conversation_turn_auth_token", "conversation_turn",
                          "auth_token", ["auth_token_id"], ["id"], ondelete="SET NULL")


def downgrade() -> None:
    op.drop_constraint("fk_conversation_turn_auth_token", "conversation_turn",
                       type_="foreignkey")
    op.drop_column("conversation_turn", "auth_token_id")
