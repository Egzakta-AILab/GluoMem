"""Optional source-bound semantic proposals after source-support schema 0005.

No historical conversion or authority backfill.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006_semantic_bindings"
down_revision = "0005_source_support"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memory_claim_semantics",
        sa.Column(
            "provenance_id",
            sa.BigInteger(),
            sa.ForeignKey("memory_provenance.id", ondelete="CASCADE"),
            primary_key=True,
            autoincrement=False,
        ),
        sa.Column(
            "frame_id",
            sa.BigInteger(),
            sa.ForeignKey("memory_frames.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("principal_id", sa.Text(), nullable=False),
        sa.Column("namespace", sa.Text(), nullable=False),
        sa.Column("contract", sa.Text(), nullable=False),
        sa.Column("meaning_json", sa.Text(), nullable=False),
        sa.Column("binding_token", sa.Text(), nullable=False),
        sa.CheckConstraint("contract = 'semantic-state-writer-v4'"),
    )
    op.create_index(
        "idx_claim_semantics_scope_frame",
        "memory_claim_semantics",
        ["tenant_id", "principal_id", "namespace", "frame_id"],
    )


def downgrade() -> None:
    op.execute("LOCK TABLE memory_claim_semantics IN ACCESS EXCLUSIVE MODE")
    op.execute(
        "DO $$ BEGIN IF EXISTS (SELECT 1 FROM memory_claim_semantics) THEN "
        "RAISE EXCEPTION 'Cannot downgrade while semantic source bindings exist'; END IF; END $$"
    )
    op.drop_index("idx_claim_semantics_scope_frame", table_name="memory_claim_semantics")
    op.drop_table("memory_claim_semantics")
