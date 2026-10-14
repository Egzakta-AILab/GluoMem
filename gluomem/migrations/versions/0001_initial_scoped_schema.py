"""initial scoped schema."""

from __future__ import annotations

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply the generated schema changes."""
    vector_dim = int(op.get_context().config.attributes.get("vector_dim", 384))
    op.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
    op.create_table(
        "knowledge_entities",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("tenant_id", sa.Text(), server_default="local", nullable=False),
        sa.Column("principal_id", sa.Text(), server_default="default", nullable=False),
        sa.Column("namespace", sa.Text(), server_default="personal", nullable=False),
        sa.Column("entity_type", sa.Text(), nullable=True),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("canonical_name", sa.Text(), nullable=True),
        sa.Column("properties", sa.Text(), server_default="{}", nullable=True),
        sa.Column(
            "valid_from",
            sa.Text(),
            server_default=sa.text(
                "to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS')"
            ),
            nullable=False,
        ),
        sa.Column("valid_to", sa.Text(), nullable=True),
        sa.Column(
            "recorded_at",
            sa.Text(),
            server_default=sa.text(
                "to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS')"
            ),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_kg_entities_scope_name",
        "knowledge_entities",
        ["tenant_id", "principal_id", "namespace", "entity_type", "canonical_name"],
        unique=False,
    )
    op.create_table(
        "memory_jobs",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("tenant_id", sa.Text(), server_default="local", nullable=False),
        sa.Column("principal_id", sa.Text(), server_default="default", nullable=False),
        sa.Column("namespace", sa.Text(), server_default="personal", nullable=False),
        sa.Column("job_type", sa.Text(), server_default="ingest_event", nullable=False),
        sa.Column("dedup_key", sa.Text(), nullable=False),
        sa.Column("payload", sa.Text(), server_default="{}", nullable=False),
        sa.Column("status", sa.Text(), server_default="queued", nullable=False),
        sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("max_attempts", sa.Integer(), server_default="3", nullable=False),
        sa.Column("scope_generation", sa.Integer(), server_default="0", nullable=False),
        sa.Column("worker_id", sa.Text(), nullable=True),
        sa.Column("lease_expires_at", sa.Text(), nullable=True),
        sa.Column(
            "available_at",
            sa.Text(),
            server_default=sa.text(
                "to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS')"
            ),
            nullable=False,
        ),
        sa.Column("llm_calls_used", sa.Integer(), server_default="0", nullable=False),
        sa.Column("max_llm_calls", sa.Integer(), server_default="6", nullable=False),
        sa.Column("tokens_used", sa.Integer(), server_default="0", nullable=False),
        sa.Column("max_tokens", sa.Integer(), server_default="8000", nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.Text(),
            server_default=sa.text(
                "to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS')"
            ),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.Text(),
            server_default=sa.text(
                "to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS')"
            ),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('queued','leased','succeeded','failed','dead_letter',"
            "'aborted_stale','fenced')"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "principal_id", "namespace", "job_type", "dedup_key"),
    )
    op.create_index(
        "idx_memory_jobs_scope",
        "memory_jobs",
        ["tenant_id", "principal_id", "namespace", "status"],
        unique=False,
    )
    op.create_index(
        "idx_memory_jobs_status_lease", "memory_jobs", ["status", "lease_expires_at"], unique=False
    )
    op.create_table(
        "memory_tombstones",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("tenant_id", sa.Text(), server_default="local", nullable=False),
        sa.Column("principal_id", sa.Text(), server_default="default", nullable=False),
        sa.Column("namespace", sa.Text(), server_default="personal", nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("source_event_id", sa.Text(), nullable=True),
        sa.Column("source_ref", sa.Text(), nullable=True),
        sa.Column("frame_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.Text(),
            server_default=sa.text(
                "to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS')"
            ),
            nullable=False,
        ),
        sa.CheckConstraint("reason IN ('dismissed','forgotten','retracted')"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_tombstones_scope_event",
        "memory_tombstones",
        ["tenant_id", "principal_id", "namespace", "source_event_id"],
        unique=False,
    )
    op.create_index(
        "idx_tombstones_scope_hash",
        "memory_tombstones",
        ["tenant_id", "principal_id", "namespace", "content_hash"],
        unique=False,
    )
    op.create_table(
        "meta",
        sa.Column("key", sa.Text(), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("key"),
    )
    op.create_table(
        "scope_generations",
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("principal_id", sa.Text(), nullable=False),
        sa.Column("namespace", sa.Text(), nullable=False),
        sa.Column("generation", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "updated_at",
            sa.Text(),
            server_default=sa.text(
                "to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS')"
            ),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("tenant_id", "principal_id", "namespace"),
    )
    op.create_table(
        "sessions",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("tenant_id", sa.Text(), server_default="local", nullable=False),
        sa.Column("principal_id", sa.Text(), server_default="default", nullable=False),
        sa.Column("namespace", sa.Text(), server_default="personal", nullable=False),
        sa.Column("gop_id", sa.Text(), nullable=False),
        sa.Column("project_id", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), server_default="active", nullable=True),
        sa.Column(
            "started_at",
            sa.Text(),
            server_default=sa.text(
                "to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS')"
            ),
            nullable=False,
        ),
        sa.Column("ended_at", sa.Text(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "principal_id", "namespace", "gop_id"),
    )
    op.create_table(
        "source_events",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("tenant_id", sa.Text(), server_default="local", nullable=False),
        sa.Column("principal_id", sa.Text(), server_default="default", nullable=False),
        sa.Column("namespace", sa.Text(), server_default="personal", nullable=False),
        sa.Column("event_id", sa.Text(), nullable=False),
        sa.Column("revision", sa.Integer(), server_default="1", nullable=False),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("source_ref", sa.Text(), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("payload_hash", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.Text(),
            server_default=sa.text(
                "to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS')"
            ),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "principal_id", "namespace", "event_id", "revision"),
    )
    op.create_index(
        "idx_source_events_scope_ref",
        "source_events",
        ["tenant_id", "principal_id", "namespace", "source_ref"],
        unique=False,
    )
    op.create_table(
        "knowledge_relations",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("tenant_id", sa.Text(), server_default="local", nullable=False),
        sa.Column("principal_id", sa.Text(), server_default="default", nullable=False),
        sa.Column("namespace", sa.Text(), server_default="personal", nullable=False),
        sa.Column("source_id", sa.BigInteger(), nullable=True),
        sa.Column("target_id", sa.BigInteger(), nullable=True),
        sa.Column("relation_type", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Double(), server_default="1.0", nullable=True),
        sa.Column("properties", sa.Text(), server_default="{}", nullable=True),
        sa.Column(
            "valid_from",
            sa.Text(),
            server_default=sa.text(
                "to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS')"
            ),
            nullable=False,
        ),
        sa.Column("valid_to", sa.Text(), nullable=True),
        sa.Column(
            "recorded_at",
            sa.Text(),
            server_default=sa.text(
                "to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS')"
            ),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["knowledge_entities.id"],
        ),
        sa.ForeignKeyConstraint(
            ["target_id"],
            ["knowledge_entities.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id", "principal_id", "namespace", "source_id", "target_id", "relation_type"
        ),
    )
    op.create_index(
        "idx_kg_rel_scope_tgt",
        "knowledge_relations",
        ["tenant_id", "principal_id", "namespace", "target_id"],
        unique=False,
    )
    op.create_table(
        "memory_frames",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("tenant_id", sa.Text(), server_default="local", nullable=False),
        sa.Column("principal_id", sa.Text(), server_default="default", nullable=False),
        sa.Column("namespace", sa.Text(), server_default="personal", nullable=False),
        sa.Column("frame_type", sa.Text(), nullable=False),
        sa.Column("gop_id", sa.Text(), nullable=False),
        sa.Column("t", sa.Integer(), server_default="0", nullable=False),
        sa.Column("base_frame_id", sa.BigInteger(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("folded_content", sa.Text(), server_default="", nullable=False),
        sa.Column("kind", sa.Text(), server_default="fact", nullable=False),
        sa.Column("status", sa.Text(), server_default="active", nullable=False),
        sa.Column("importance", sa.Text(), server_default="normal", nullable=False),
        sa.Column("source", sa.Text(), server_default="user_stated", nullable=False),
        sa.Column("trust", sa.Text(), server_default="normal", nullable=False),
        sa.Column("confidence", sa.Double(), server_default="1.0", nullable=False),
        sa.Column("superseded_by", sa.BigInteger(), nullable=True),
        sa.Column("supersedes", sa.BigInteger(), nullable=True),
        sa.Column("expires_at", sa.Text(), nullable=True),
        sa.Column("pinned", sa.Integer(), server_default="0", nullable=False),
        sa.Column("verified_by_user", sa.Integer(), server_default="0", nullable=False),
        sa.Column("access_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "created_at",
            sa.Text(),
            server_default=sa.text(
                "to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS')"
            ),
            nullable=False,
        ),
        sa.Column("event_time", sa.Text(), nullable=True),
        sa.Column(
            "last_accessed",
            sa.Text(),
            server_default=sa.text(
                "to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS')"
            ),
            nullable=False,
        ),
        sa.Column("content_hash", sa.Text(), nullable=True),
        sa.CheckConstraint("frame_type IN ('I','P','B')"),
        sa.CheckConstraint(
            "importance IN ('critical','important','normal','temporary','deprecated')"
        ),
        sa.CheckConstraint(
            "kind IN ('fact','profile','preference','decision','procedure','episode',"
            "'correction','summary')"
        ),
        sa.CheckConstraint(
            "source IN ('user_stated','tool_verified','agent_inferred','import','system',"
            "'personal','workspace','team_sync')"
        ),
        sa.CheckConstraint(
            "status IN ('candidate','active','superseded','expired','forgotten','tombstoned')"
        ),
        sa.CheckConstraint("trust IN ('trusted','normal','low','quarantined')"),
        sa.ForeignKeyConstraint(
            ["base_frame_id"],
            ["memory_frames.id"],
        ),
        sa.ForeignKeyConstraint(
            ["superseded_by"],
            ["memory_frames.id"],
        ),
        sa.ForeignKeyConstraint(
            ["supersedes"],
            ["memory_frames.id"],
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "principal_id", "namespace", "gop_id"],
            [
                "sessions.tenant_id",
                "sessions.principal_id",
                "sessions.namespace",
                "sessions.gop_id",
            ],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_frames_base", "memory_frames", ["base_frame_id"], unique=False)
    op.create_index(
        "idx_frames_scope_event",
        "memory_frames",
        ["tenant_id", "principal_id", "namespace", "event_time"],
        unique=False,
    )
    op.create_index(
        "idx_frames_scope_gop_t",
        "memory_frames",
        ["tenant_id", "principal_id", "namespace", "gop_id", "t"],
        unique=False,
    )
    op.create_index(
        "idx_frames_scope_hash",
        "memory_frames",
        ["tenant_id", "principal_id", "namespace", "content_hash"],
        unique=False,
    )
    op.create_index(
        "idx_frames_scope_status",
        "memory_frames",
        ["tenant_id", "principal_id", "namespace", "status", "trust"],
        unique=False,
    )
    op.create_table(
        "kg_entity_frames",
        sa.Column("entity_id", sa.BigInteger(), nullable=False),
        sa.Column("frame_id", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(["entity_id"], ["knowledge_entities.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["frame_id"], ["memory_frames.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("entity_id", "frame_id"),
    )
    op.create_index("idx_kg_entity_frames_frame", "kg_entity_frames", ["frame_id"], unique=False)
    op.create_table(
        "knowledge_relation_history",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("relation_id", sa.BigInteger(), nullable=False),
        sa.Column("valid_from", sa.Text(), nullable=False),
        sa.Column("valid_to", sa.Text(), nullable=False),
        sa.Column("recorded_at", sa.Text(), nullable=False),
        sa.Column(
            "closed_recorded_at",
            sa.Text(),
            server_default=sa.text(
                "to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS')"
            ),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["relation_id"], ["knowledge_relations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_kg_rel_history_rel", "knowledge_relation_history", ["relation_id"], unique=False
    )
    op.create_table(
        "memory_frame_chunks",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("frame_id", sa.BigInteger(), nullable=False),
        sa.Column("chunk_idx", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=True),
        sa.Column("char_end", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["frame_id"], ["memory_frames.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("frame_id", "chunk_idx"),
    )
    op.create_table(
        "memory_frame_lineage",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("derived_frame_id", sa.BigInteger(), nullable=False),
        sa.Column("source_frame_id", sa.BigInteger(), nullable=False),
        sa.Column("source_frame_type", sa.Text(), nullable=False),
        sa.Column("source_content_hash", sa.Text(), nullable=True),
        sa.Column("source_source", sa.Text(), nullable=False),
        sa.Column("source_trust", sa.Text(), nullable=False),
        sa.Column("source_created_at", sa.Text(), nullable=False),
        sa.Column("source_event_time", sa.Text(), nullable=True),
        sa.Column("operation", sa.Text(), nullable=False),
        sa.Column(
            "recorded_at",
            sa.Text(),
            server_default=sa.text(
                "to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS')"
            ),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["derived_frame_id"], ["memory_frames.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_lineage_derived", "memory_frame_lineage", ["derived_frame_id"], unique=False
    )
    op.create_index("idx_lineage_source", "memory_frame_lineage", ["source_frame_id"], unique=False)
    op.create_table(
        "memory_frames_fts",
        sa.Column("rowid", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("folded_content", sa.Text(), server_default="", nullable=False),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('simple', coalesce(folded_content, '')) "
                "|| to_tsvector('english', coalesce(content, ''))",
                persisted=True,
            ),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["rowid"], ["memory_frames.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("rowid"),
    )
    op.create_index(
        "idx_frames_fts_gin", "memory_frames_fts", ["tsv"], unique=False, postgresql_using="gin"
    )
    op.create_table(
        "memory_frames_vec",
        sa.Column("rowid", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=vector_dim), nullable=False),
        sa.ForeignKeyConstraint(["rowid"], ["memory_frames.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("rowid"),
    )
    op.create_index(
        "idx_frames_vec_hnsw",
        "memory_frames_vec",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_l2_ops"},
    )
    op.create_table(
        "memory_provenance",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("tenant_id", sa.Text(), server_default="local", nullable=False),
        sa.Column("principal_id", sa.Text(), server_default="default", nullable=False),
        sa.Column("namespace", sa.Text(), server_default="personal", nullable=False),
        sa.Column("frame_id", sa.BigInteger(), nullable=False),
        sa.Column("source_event_id", sa.BigInteger(), nullable=True),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("source_ref", sa.Text(), nullable=False),
        sa.Column("message_id", sa.Text(), nullable=True),
        sa.Column("verbatim_span", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.Text(),
            server_default=sa.text(
                "to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS')"
            ),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["frame_id"], ["memory_frames.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_event_id"], ["source_events.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_provenance_frame", "memory_provenance", ["frame_id"], unique=False)
    op.create_index(
        "idx_provenance_scope_ref",
        "memory_provenance",
        ["tenant_id", "principal_id", "namespace", "source_ref"],
        unique=False,
    )
    op.create_table(
        "memory_frame_chunks_vec",
        sa.Column("rowid", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=vector_dim), nullable=False),
        sa.ForeignKeyConstraint(["rowid"], ["memory_frame_chunks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("rowid"),
    )
    op.create_index(
        "idx_chunks_vec_hnsw",
        "memory_frame_chunks_vec",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_l2_ops"},
    )


def downgrade() -> None:
    """Reverse the generated schema changes."""
    op.drop_index(
        "idx_chunks_vec_hnsw",
        table_name="memory_frame_chunks_vec",
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_l2_ops"},
    )
    op.drop_table("memory_frame_chunks_vec")
    op.drop_index("idx_provenance_scope_ref", table_name="memory_provenance")
    op.drop_index("idx_provenance_frame", table_name="memory_provenance")
    op.drop_table("memory_provenance")
    op.drop_index(
        "idx_frames_vec_hnsw",
        table_name="memory_frames_vec",
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_l2_ops"},
    )
    op.drop_table("memory_frames_vec")
    op.drop_index("idx_frames_fts_gin", table_name="memory_frames_fts", postgresql_using="gin")
    op.drop_table("memory_frames_fts")
    op.drop_index("idx_lineage_source", table_name="memory_frame_lineage")
    op.drop_index("idx_lineage_derived", table_name="memory_frame_lineage")
    op.drop_table("memory_frame_lineage")
    op.drop_table("memory_frame_chunks")
    op.drop_index("idx_kg_rel_history_rel", table_name="knowledge_relation_history")
    op.drop_table("knowledge_relation_history")
    op.drop_index("idx_kg_entity_frames_frame", table_name="kg_entity_frames")
    op.drop_table("kg_entity_frames")
    op.drop_index("idx_frames_scope_status", table_name="memory_frames")
    op.drop_index("idx_frames_scope_hash", table_name="memory_frames")
    op.drop_index("idx_frames_scope_gop_t", table_name="memory_frames")
    op.drop_index("idx_frames_scope_event", table_name="memory_frames")
    op.drop_index("idx_frames_base", table_name="memory_frames")
    op.drop_table("memory_frames")
    op.drop_index("idx_kg_rel_scope_tgt", table_name="knowledge_relations")
    op.drop_table("knowledge_relations")
    op.drop_index("idx_source_events_scope_ref", table_name="source_events")
    op.drop_table("source_events")
    op.drop_table("sessions")
    op.drop_table("scope_generations")
    op.drop_table("meta")
    op.drop_index("idx_tombstones_scope_hash", table_name="memory_tombstones")
    op.drop_index("idx_tombstones_scope_event", table_name="memory_tombstones")
    op.drop_table("memory_tombstones")
    op.drop_index("idx_memory_jobs_status_lease", table_name="memory_jobs")
    op.drop_index("idx_memory_jobs_scope", table_name="memory_jobs")
    op.drop_table("memory_jobs")
    op.drop_index("idx_kg_entities_scope_name", table_name="knowledge_entities")
    op.drop_table("knowledge_entities")
