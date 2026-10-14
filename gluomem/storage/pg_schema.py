"""Dimensioned PostgreSQL Core metadata for Alembic."""

from __future__ import annotations

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TSVECTOR

_NOW = sa.text("""to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS')""")
_SCOPE = ("tenant_id", "principal_id", "namespace")


def build_metadata(dim: int = 384) -> sa.MetaData:
    """Build schema-independent metadata with ``dim``-length vectors."""
    if dim <= 0:
        raise ValueError("vector dim must be positive")
    metadata = sa.MetaData()

    def scope() -> list[sa.Column]:
        return [
            sa.Column(name, sa.Text, nullable=False, server_default=default)
            for name, default in zip(_SCOPE, ("local", "default", "personal"), strict=True)
        ]

    def ident() -> sa.Column:
        return sa.Column("id", sa.BigInteger, primary_key=True)

    def stamp(name: str) -> sa.Column:
        return sa.Column(name, sa.Text, nullable=False, server_default=_NOW)

    def ref(name: str, target: str, *, nullable: bool = True, ondelete=None) -> sa.Column:
        return sa.Column(
            name, sa.BigInteger, sa.ForeignKey(target, ondelete=ondelete), nullable=nullable
        )

    def text(name: str, default: str | None = None, *, nullable: bool = True) -> sa.Column:
        return sa.Column(name, sa.Text, nullable=nullable, server_default=default)

    def integer(name: str, default: str = "0") -> sa.Column:
        return sa.Column(name, sa.Integer, nullable=False, server_default=default)

    def table(name: str, *columns) -> sa.Table:
        return sa.Table(name, metadata, *columns)

    def index(name: str, tbl: sa.Table, *columns: str, **options) -> None:
        sa.Index(name, *(tbl.c[c] for c in columns), **options)

    table("meta", sa.Column("key", sa.Text, primary_key=True), text("value", nullable=False))
    table(
        "sessions",
        ident(),
        *scope(),
        text("gop_id", nullable=False),
        text("project_id"),
        text("status", "active"),
        stamp("started_at"),
        text("ended_at"),
        text("summary"),
        sa.UniqueConstraint(*_SCOPE, "gop_id"),
    )
    frames = table(
        "memory_frames",
        ident(),
        *scope(),
        text("frame_type", nullable=False),
        text("gop_id", nullable=False),
        integer("t"),
        ref("base_frame_id", "memory_frames.id"),
        text("content", nullable=False),
        text("folded_content", "", nullable=False),
        text("kind", "fact", nullable=False),
        text("status", "active", nullable=False),
        text("importance", "normal", nullable=False),
        text("source", "user_stated", nullable=False),
        text("trust", "normal", nullable=False),
        sa.Column("confidence", sa.Double, nullable=False, server_default="1.0"),
        ref("superseded_by", "memory_frames.id"),
        ref("supersedes", "memory_frames.id"),
        text("expires_at"),
        integer("pinned"),
        integer("verified_by_user"),
        integer("access_count"),
        stamp("created_at"),
        text("event_time"),
        stamp("last_accessed"),
        text("content_hash"),
        sa.ForeignKeyConstraint(
            [*_SCOPE, "gop_id"], [f"sessions.{c}" for c in (*_SCOPE, "gop_id")]
        ),
        sa.CheckConstraint("frame_type IN ('I','P','B')"),
        sa.CheckConstraint(
            "kind IN ('fact','profile','preference','decision','procedure','episode',"
            "'correction','summary')"
        ),
        sa.CheckConstraint(
            "status IN ('candidate','active','superseded','expired','forgotten','tombstoned')"
        ),
        sa.CheckConstraint(
            "importance IN ('critical','important','normal','temporary','deprecated')"
        ),
        sa.CheckConstraint(
            "source IN ('user_stated','tool_verified','agent_inferred','import','system',"
            "'personal','workspace','team_sync')"
        ),
        sa.CheckConstraint("trust IN ('trusted','normal','low','quarantined')"),
    )
    index("idx_frames_base", frames, "base_frame_id")
    index("idx_frames_scope_status", frames, *_SCOPE, "status", "trust")
    index("idx_frames_scope_gop_t", frames, *_SCOPE, "gop_id", "t")
    index("idx_frames_scope_hash", frames, *_SCOPE, "content_hash")
    index("idx_frames_scope_event", frames, *_SCOPE, "event_time")
    fts = table(
        "memory_frames_fts",
        sa.Column(
            "rowid",
            sa.BigInteger,
            sa.ForeignKey("memory_frames.id", ondelete="CASCADE"),
            primary_key=True,
            autoincrement=False,
        ),
        text("content", nullable=False),
        text("folded_content", "", nullable=False),
        sa.Column(
            "tsv",
            TSVECTOR,
            sa.Computed(
                "to_tsvector('simple', coalesce(folded_content, '')) "
                "|| to_tsvector('english', coalesce(content, ''))",
                persisted=True,
            ),
        ),
    )
    index("idx_frames_fts_gin", fts, "tsv", postgresql_using="gin")
    table(
        "memory_frame_chunks",
        ident(),
        ref("frame_id", "memory_frames.id", nullable=False, ondelete="CASCADE"),
        sa.Column("chunk_idx", sa.Integer, nullable=False),
        text("content", nullable=False),
        sa.Column("char_start", sa.Integer),
        sa.Column("char_end", sa.Integer),
        sa.UniqueConstraint("frame_id", "chunk_idx"),
    )
    for name, target, index_name in (
        ("memory_frames_vec", "memory_frames.id", "idx_frames_vec_hnsw"),
        ("memory_frame_chunks_vec", "memory_frame_chunks.id", "idx_chunks_vec_hnsw"),
    ):
        vec = table(
            name,
            sa.Column(
                "rowid",
                sa.BigInteger,
                sa.ForeignKey(target, ondelete="CASCADE"),
                primary_key=True,
                autoincrement=False,
            ),
            sa.Column("embedding", Vector(dim), nullable=False),
        )
        index(
            index_name,
            vec,
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_l2_ops"},
        )
    lineage = table(
        "memory_frame_lineage",
        ident(),
        ref("derived_frame_id", "memory_frames.id", nullable=False, ondelete="CASCADE"),
        sa.Column("source_frame_id", sa.BigInteger, nullable=False),
        text("source_frame_type", nullable=False),
        text("source_content_hash"),
        text("source_source", nullable=False),
        text("source_trust", nullable=False),
        text("source_created_at", nullable=False),
        text("source_event_time"),
        text("operation", nullable=False),
        stamp("recorded_at"),
    )
    index("idx_lineage_derived", lineage, "derived_frame_id")
    index("idx_lineage_source", lineage, "source_frame_id")
    entities = table(
        "knowledge_entities",
        ident(),
        *scope(),
        text("entity_type"),
        text("name"),
        text("canonical_name"),
        text("properties", "{}"),
        stamp("valid_from"),
        text("valid_to"),
        stamp("recorded_at"),
    )
    index("idx_kg_entities_scope_name", entities, *_SCOPE, "entity_type", "canonical_name")
    relations = table(
        "knowledge_relations",
        ident(),
        *scope(),
        ref("source_id", "knowledge_entities.id"),
        ref("target_id", "knowledge_entities.id"),
        text("relation_type"),
        sa.Column("confidence", sa.Double, server_default="1.0"),
        text("properties", "{}"),
        stamp("valid_from"),
        text("valid_to"),
        stamp("recorded_at"),
        sa.UniqueConstraint(*_SCOPE, "source_id", "target_id", "relation_type"),
    )
    index("idx_kg_rel_scope_tgt", relations, *_SCOPE, "target_id")
    history = table(
        "knowledge_relation_history",
        ident(),
        ref("relation_id", "knowledge_relations.id", nullable=False, ondelete="CASCADE"),
        text("valid_from", nullable=False),
        text("valid_to", nullable=False),
        text("recorded_at", nullable=False),
        stamp("closed_recorded_at"),
    )
    index("idx_kg_rel_history_rel", history, "relation_id")
    links = table(
        "kg_entity_frames",
        sa.Column(
            "entity_id",
            sa.BigInteger,
            sa.ForeignKey("knowledge_entities.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "frame_id",
            sa.BigInteger,
            sa.ForeignKey("memory_frames.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    index("idx_kg_entity_frames_frame", links, "frame_id")
    events = table(
        "source_events",
        ident(),
        *scope(),
        text("event_id", nullable=False),
        integer("revision", "1"),
        text("source_type", nullable=False),
        text("source_ref", nullable=False),
        text("payload_json", nullable=False),
        text("payload_hash", nullable=False),
        stamp("created_at"),
        sa.UniqueConstraint(*_SCOPE, "event_id", "revision"),
    )
    index("idx_source_events_scope_ref", events, *_SCOPE, "source_ref")
    provenance = table(
        "memory_provenance",
        ident(),
        *scope(),
        ref("frame_id", "memory_frames.id", nullable=False, ondelete="CASCADE"),
        ref("source_event_id", "source_events.id", ondelete="SET NULL"),
        text("source_type", nullable=False),
        text("source_ref", nullable=False),
        text("message_id"),
        text("verbatim_span"),
        stamp("created_at"),
    )
    index("idx_provenance_frame", provenance, "frame_id")
    index("idx_provenance_scope_ref", provenance, *_SCOPE, "source_ref")
    tombstones = table(
        "memory_tombstones",
        ident(),
        *scope(),
        text("content_hash", nullable=False),
        text("reason", nullable=False),
        text("source_event_id"),
        text("source_ref"),
        sa.Column("frame_id", sa.BigInteger),
        stamp("created_at"),
        sa.CheckConstraint("reason IN ('dismissed','forgotten','retracted')"),
    )
    index("idx_tombstones_scope_hash", tombstones, *_SCOPE, "content_hash")
    index("idx_tombstones_scope_event", tombstones, *_SCOPE, "source_event_id")
    table(
        "scope_generations",
        *(sa.Column(name, sa.Text, primary_key=True) for name in _SCOPE),
        integer("generation"),
        stamp("updated_at"),
    )
    jobs = table(
        "memory_jobs",
        ident(),
        *scope(),
        text("job_type", "ingest_event", nullable=False),
        text("dedup_key", nullable=False),
        text("payload", "{}", nullable=False),
        text("status", "queued", nullable=False),
        integer("attempt_count"),
        integer("max_attempts", "3"),
        integer("scope_generation"),
        text("worker_id"),
        text("lease_expires_at"),
        stamp("available_at"),
        text("last_error"),
        stamp("created_at"),
        stamp("updated_at"),
        sa.UniqueConstraint(*_SCOPE, "job_type", "dedup_key"),
        sa.CheckConstraint("status IN ('queued','leased','succeeded','dead_letter')"),
    )
    index("idx_memory_jobs_status_lease", jobs, "status", "lease_expires_at")
    index("idx_memory_jobs_scope", jobs, *_SCOPE, "status")
    return metadata
