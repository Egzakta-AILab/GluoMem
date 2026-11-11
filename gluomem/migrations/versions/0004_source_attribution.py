"""Nullable trusted source attribution; no inferred role/time backfill.

Column operations generated against live 0003_temporal_state metadata.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_source_attribution"
down_revision = "0003_temporal_state"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("memory_provenance", sa.Column("message_role", sa.Text(), nullable=True))
    op.add_column("memory_provenance", sa.Column("observed_at", sa.Text(), nullable=True))


def downgrade() -> None:
    op.execute("LOCK TABLE memory_provenance IN ACCESS EXCLUSIVE MODE")
    op.execute(
        "DO $$ BEGIN IF EXISTS (SELECT 1 FROM memory_provenance "
        "WHERE message_role IS NOT NULL OR observed_at IS NOT NULL) THEN "
        "RAISE EXCEPTION 'Cannot downgrade while trusted source attribution exists'; "
        "END IF; END $$"
    )
    op.drop_column("memory_provenance", "observed_at")
    op.drop_column("memory_provenance", "message_role")
