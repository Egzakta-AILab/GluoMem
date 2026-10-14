"""Initial scoped GluoMem PostgreSQL schema with pgvector HNSW indexes."""

from __future__ import annotations

from pathlib import Path

from alembic import context, op

from gluomem.migrations.runner import validate_schema_name

revision: str = "0001_initial_scoped_schema"
down_revision: str | None = None
branch_labels: tuple[str, ...] | None = None
depends_on: str | None = None

_SCHEMA_POSTGRES_PATH = (
    Path(__file__).resolve().parents[2] / "storage" / "schema_postgres.sql"
)


def _get_params() -> tuple[str, int]:
    cfg = context.config
    schema = validate_schema_name(str(cfg.attributes.get("schema", "gluomem")))
    vector_dim = int(cfg.attributes.get("vector_dim", 384))
    if vector_dim <= 0:
        raise ValueError(f"vector_dim must be positive, got {vector_dim}")
    return schema, vector_dim


def upgrade() -> None:
    """Create the initial scoped GluoMem tables, FTS GIN index, and pgvector HNSW indexes."""
    schema, vector_dim = _get_params()
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    op.execute(f"SET search_path TO {schema}, public")
    ddl = (
        _SCHEMA_POSTGRES_PATH.read_text(encoding="utf-8")
        .replace("__SCHEMA__", schema)
        .replace("__VECTOR_DIM__", str(vector_dim))
        .replace("__DIM__", str(vector_dim))
    )
    op.execute(ddl)


def downgrade() -> None:
    """Drop all GluoMem tables in reverse dependency order."""
    schema, _ = _get_params()
    tables = (
        "memory_jobs",
        "scope_generations",
        "memory_tombstones",
        "memory_provenance",
        "source_events",
        "kg_entity_frames",
        "knowledge_relation_history",
        "knowledge_relations",
        "knowledge_entities",
        "memory_frame_chunks_vec",
        "memory_frames_vec",
        "memory_frames_fts",
        "memory_frame_chunks",
        "memory_frame_lineage",
        "memory_frames",
        "sessions",
        "meta",
    )
    for table in tables:
        op.execute(f"DROP TABLE IF EXISTS {schema}.{table} CASCADE")
