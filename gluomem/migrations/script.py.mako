"""${message}."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    """Apply the generated schema changes."""
    vector_dim = int(op.get_context().config.attributes.get("vector_dim", 384))
    op.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    """Reverse the generated schema changes."""
    ${downgrades if downgrades else "pass"}
