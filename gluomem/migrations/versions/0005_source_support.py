"""Revocable exact source-support witnesses; no plaintext or proof backfill."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005_source_support"
down_revision = "0004_source_attribution"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memory_source_support",
        sa.Column("provenance_id", sa.BigInteger(), nullable=False, autoincrement=False),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("principal_id", sa.Text(), nullable=False),
        sa.Column("namespace", sa.Text(), nullable=False),
        sa.Column("event_payload_hash", sa.Text(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("message_time", sa.Text(), nullable=True),
        sa.Column("binding_hash", sa.Text(), nullable=False),
        sa.CheckConstraint("char_start >= 0"),
        sa.CheckConstraint("char_end > char_start"),
        sa.ForeignKeyConstraint(["provenance_id"], ["memory_provenance.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("provenance_id"),
    )
    op.create_index(
        "idx_source_support_scope",
        "memory_source_support",
        ["tenant_id", "principal_id", "namespace"],
        unique=False,
    )


def downgrade() -> None:
    op.execute("LOCK TABLE memory_source_support IN ACCESS EXCLUSIVE MODE")
    op.execute(
        "DO $$ BEGIN IF EXISTS (SELECT 1 FROM memory_source_support) THEN "
        "RAISE EXCEPTION 'Cannot downgrade while retained source support exists'; "
        "END IF; END $$"
    )
    op.drop_index("idx_source_support_scope", table_name="memory_source_support")
    op.drop_table("memory_source_support")
