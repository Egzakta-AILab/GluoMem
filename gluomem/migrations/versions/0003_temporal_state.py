"""Metadata-only ordinary state chronology (not correction lineage).

Table/index operations generated against live 7e2b094eb309 metadata.
Existing event_time/created_at values are deliberately not backfilled.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_temporal_state"
down_revision = "7e2b094eb309"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memory_frame_temporal",
        sa.Column("frame_id", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("principal_id", sa.Text(), nullable=False),
        sa.Column("namespace", sa.Text(), nullable=False),
        sa.Column("operation", sa.Text(), nullable=False),
        sa.Column("previous_frame_id", sa.BigInteger(), nullable=True),
        sa.Column("effective_at", sa.Text(), nullable=True),
        sa.Column("observed_at", sa.Text(), nullable=False),
        sa.Column("recorded_at", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "(operation = 'observation' AND previous_frame_id IS NULL) OR "
            "(operation = 'transition' AND previous_frame_id IS NOT NULL "
            "AND effective_at IS NOT NULL)"
        ),
        sa.CheckConstraint("frame_id != previous_frame_id"),
        sa.ForeignKeyConstraint(["frame_id"], ["memory_frames.id"]),
        sa.ForeignKeyConstraint(["previous_frame_id"], ["memory_frames.id"]),
        sa.PrimaryKeyConstraint("frame_id"),
        sa.UniqueConstraint("previous_frame_id"),
    )
    op.create_index(
        "idx_temporal_scope",
        "memory_frame_temporal",
        ["tenant_id", "principal_id", "namespace"],
        unique=False,
    )


def downgrade() -> None:
    op.execute("LOCK TABLE memory_frame_temporal IN ACCESS EXCLUSIVE MODE")
    op.execute(
        "DO $$ BEGIN IF EXISTS (SELECT 1 FROM memory_frame_temporal) THEN "
        "RAISE EXCEPTION 'Cannot downgrade while temporal state authority exists'; "
        "END IF; END $$"
    )
    op.drop_index("idx_temporal_scope", table_name="memory_frame_temporal")
    op.drop_table("memory_frame_temporal")
