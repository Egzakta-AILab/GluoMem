-- gluomem PostgreSQL + pgvector schema.
-- Applied idempotently by PostgresBackend on open; __VECTOR_DIM__ is
-- substituted with the configured embedding dimension.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id           BIGSERIAL PRIMARY KEY,
    tenant_id    TEXT NOT NULL DEFAULT 'local',
    principal_id TEXT NOT NULL DEFAULT 'default',
    namespace    TEXT NOT NULL DEFAULT 'personal',
    gop_id       TEXT NOT NULL,
    project_id   TEXT,
    status       TEXT DEFAULT 'active',
    started_at   TEXT NOT NULL DEFAULT (to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS')),
    ended_at     TEXT,
    summary      TEXT,
    UNIQUE(tenant_id, principal_id, namespace, gop_id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_scope_gop
    ON sessions(tenant_id, principal_id, namespace, gop_id);

CREATE TABLE IF NOT EXISTS memory_frames (
    id               BIGSERIAL PRIMARY KEY,
    tenant_id        TEXT NOT NULL DEFAULT 'local',
    principal_id     TEXT NOT NULL DEFAULT 'default',
    namespace        TEXT NOT NULL DEFAULT 'personal',
    frame_type       TEXT NOT NULL CHECK(frame_type IN ('I','P','B')),
    gop_id           TEXT NOT NULL,
    t                INTEGER NOT NULL DEFAULT 0,
    base_frame_id    BIGINT REFERENCES memory_frames(id),
    content          TEXT NOT NULL,
    folded_content   TEXT NOT NULL DEFAULT '',
    kind             TEXT NOT NULL DEFAULT 'fact'
        CHECK(kind IN ('fact','profile','preference','decision','procedure','episode','correction','summary')),
    status           TEXT NOT NULL DEFAULT 'active'
        CHECK(status IN ('candidate','active','superseded','expired','forgotten','tombstoned')),
    importance       TEXT NOT NULL DEFAULT 'normal'
        CHECK(importance IN ('critical','important','normal','temporary','deprecated')),
    source           TEXT NOT NULL DEFAULT 'user_stated'
        CHECK(source IN ('user_stated','tool_verified','agent_inferred','import','system','personal','workspace','team_sync')),
    trust            TEXT NOT NULL DEFAULT 'normal'
        CHECK(trust IN ('trusted','normal','low','quarantined')),
    confidence       DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    superseded_by    BIGINT REFERENCES memory_frames(id),
    supersedes       BIGINT REFERENCES memory_frames(id),
    expires_at       TEXT,
    pinned           INTEGER NOT NULL DEFAULT 0,
    verified_by_user INTEGER NOT NULL DEFAULT 0,
    access_count     INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL DEFAULT (to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS')),
    event_time       TEXT,
    last_accessed    TEXT NOT NULL DEFAULT (to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS')),
    content_hash     TEXT,
    FOREIGN KEY (tenant_id, principal_id, namespace, gop_id)
        REFERENCES sessions(tenant_id, principal_id, namespace, gop_id)
);

CREATE INDEX IF NOT EXISTS idx_frames_gop_t ON memory_frames(gop_id, t);
CREATE INDEX IF NOT EXISTS idx_frames_type  ON memory_frames(frame_type, gop_id);
CREATE INDEX IF NOT EXISTS idx_frames_base  ON memory_frames(base_frame_id);
CREATE INDEX IF NOT EXISTS idx_frames_hash  ON memory_frames(content_hash);
CREATE INDEX IF NOT EXISTS idx_frames_event ON memory_frames(event_time);
CREATE INDEX IF NOT EXISTS idx_frames_scope_status
    ON memory_frames(tenant_id, principal_id, namespace, status, trust);
CREATE INDEX IF NOT EXISTS idx_frames_scope_gop_t
    ON memory_frames(tenant_id, principal_id, namespace, gop_id, t);
CREATE INDEX IF NOT EXISTS idx_frames_scope_hash
    ON memory_frames(tenant_id, principal_id, namespace, content_hash);
CREATE INDEX IF NOT EXISTS idx_frames_scope_event
    ON memory_frames(tenant_id, principal_id, namespace, event_time);

CREATE TABLE IF NOT EXISTS memory_frames_fts (
    rowid          BIGINT PRIMARY KEY REFERENCES memory_frames(id) ON DELETE CASCADE,
    content        TEXT NOT NULL,
    folded_content TEXT NOT NULL DEFAULT '',
    tsv            tsvector GENERATED ALWAYS AS (
                       to_tsvector('simple', coalesce(folded_content, ''))
                       || to_tsvector('english', coalesce(content, ''))
                   ) STORED
);

CREATE INDEX IF NOT EXISTS idx_frames_fts_gin ON memory_frames_fts USING GIN (tsv);

CREATE TABLE IF NOT EXISTS memory_frame_chunks (
    id         BIGSERIAL PRIMARY KEY,
    frame_id   BIGINT NOT NULL REFERENCES memory_frames(id) ON DELETE CASCADE,
    chunk_idx  INTEGER NOT NULL,
    content    TEXT NOT NULL,
    char_start INTEGER,
    char_end   INTEGER,
    UNIQUE(frame_id, chunk_idx)
);

CREATE TABLE IF NOT EXISTS memory_frames_vec (
    rowid     BIGINT PRIMARY KEY REFERENCES memory_frames(id) ON DELETE CASCADE,
    embedding vector(__VECTOR_DIM__) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_frames_vec_hnsw
    ON memory_frames_vec USING hnsw (embedding vector_l2_ops);

CREATE TABLE IF NOT EXISTS memory_frame_chunks_vec (
    rowid     BIGINT PRIMARY KEY REFERENCES memory_frame_chunks(id) ON DELETE CASCADE,
    embedding vector(__VECTOR_DIM__) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_vec_hnsw
    ON memory_frame_chunks_vec USING hnsw (embedding vector_l2_ops);

CREATE TABLE IF NOT EXISTS memory_frame_lineage (
    id                  BIGSERIAL PRIMARY KEY,
    derived_frame_id    BIGINT NOT NULL REFERENCES memory_frames(id) ON DELETE CASCADE,
    source_frame_id     BIGINT NOT NULL,
    source_frame_type   TEXT NOT NULL,
    source_content_hash TEXT,
    source_source       TEXT NOT NULL,
    source_trust        TEXT NOT NULL,
    source_created_at   TEXT NOT NULL,
    source_event_time   TEXT,
    operation           TEXT NOT NULL,
    recorded_at         TEXT NOT NULL DEFAULT (to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS'))
);

CREATE INDEX IF NOT EXISTS idx_lineage_derived ON memory_frame_lineage(derived_frame_id);
CREATE INDEX IF NOT EXISTS idx_lineage_source  ON memory_frame_lineage(source_frame_id);

CREATE TABLE IF NOT EXISTS knowledge_entities (
    id             BIGSERIAL PRIMARY KEY,
    tenant_id      TEXT NOT NULL DEFAULT 'local',
    principal_id   TEXT NOT NULL DEFAULT 'default',
    namespace      TEXT NOT NULL DEFAULT 'personal',
    entity_type    TEXT,
    name           TEXT,
    canonical_name TEXT,
    properties     TEXT DEFAULT '{}',
    valid_from     TEXT NOT NULL DEFAULT (to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS')),
    valid_to       TEXT,
    recorded_at    TEXT NOT NULL DEFAULT (to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS'))
);

CREATE INDEX IF NOT EXISTS idx_kg_entities_canonical
    ON knowledge_entities(canonical_name);
CREATE INDEX IF NOT EXISTS idx_kg_entities_scope_name
    ON knowledge_entities(tenant_id, principal_id, namespace, entity_type, canonical_name);

CREATE TABLE IF NOT EXISTS knowledge_relations (
    id            BIGSERIAL PRIMARY KEY,
    tenant_id     TEXT NOT NULL DEFAULT 'local',
    principal_id  TEXT NOT NULL DEFAULT 'default',
    namespace     TEXT NOT NULL DEFAULT 'personal',
    source_id     BIGINT REFERENCES knowledge_entities(id),
    target_id     BIGINT REFERENCES knowledge_entities(id),
    relation_type TEXT,
    confidence    DOUBLE PRECISION DEFAULT 1.0,
    properties    TEXT DEFAULT '{}',
    valid_from    TEXT NOT NULL DEFAULT (to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS')),
    valid_to      TEXT,
    recorded_at   TEXT NOT NULL DEFAULT (to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS')),
    UNIQUE(tenant_id, principal_id, namespace, source_id, target_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_kg_relations_target
    ON knowledge_relations(target_id);
CREATE INDEX IF NOT EXISTS idx_kg_rel_scope_src
    ON knowledge_relations(tenant_id, principal_id, namespace, source_id);
CREATE INDEX IF NOT EXISTS idx_kg_rel_scope_tgt
    ON knowledge_relations(tenant_id, principal_id, namespace, target_id);

CREATE TABLE IF NOT EXISTS knowledge_relation_history (
    id                 BIGSERIAL PRIMARY KEY,
    relation_id        BIGINT NOT NULL REFERENCES knowledge_relations(id) ON DELETE CASCADE,
    valid_from         TEXT NOT NULL,
    valid_to           TEXT NOT NULL,
    recorded_at        TEXT NOT NULL,
    closed_recorded_at TEXT NOT NULL DEFAULT (to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS'))
);

CREATE INDEX IF NOT EXISTS idx_kg_rel_history_rel
    ON knowledge_relation_history(relation_id);

CREATE TABLE IF NOT EXISTS kg_entity_frames (
    entity_id BIGINT REFERENCES knowledge_entities(id) ON DELETE CASCADE,
    frame_id  BIGINT REFERENCES memory_frames(id) ON DELETE CASCADE,
    PRIMARY KEY(entity_id, frame_id)
);

CREATE INDEX IF NOT EXISTS idx_kg_entity_frames_frame
    ON kg_entity_frames(frame_id);

CREATE TABLE IF NOT EXISTS source_events (
    id           BIGSERIAL PRIMARY KEY,
    tenant_id    TEXT NOT NULL DEFAULT 'local',
    principal_id TEXT NOT NULL DEFAULT 'default',
    namespace    TEXT NOT NULL DEFAULT 'personal',
    event_id     TEXT NOT NULL,
    revision     INTEGER NOT NULL DEFAULT 1,
    source_type  TEXT NOT NULL,
    source_ref   TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS')),
    UNIQUE(tenant_id, principal_id, namespace, event_id, revision)
);

CREATE INDEX IF NOT EXISTS idx_source_events_scope_ref
    ON source_events(tenant_id, principal_id, namespace, source_ref);

CREATE TABLE IF NOT EXISTS memory_provenance (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       TEXT NOT NULL DEFAULT 'local',
    principal_id    TEXT NOT NULL DEFAULT 'default',
    namespace       TEXT NOT NULL DEFAULT 'personal',
    frame_id        BIGINT NOT NULL REFERENCES memory_frames(id) ON DELETE CASCADE,
    source_event_id BIGINT REFERENCES source_events(id) ON DELETE SET NULL,
    source_type     TEXT NOT NULL,
    source_ref      TEXT NOT NULL,
    message_id      TEXT,
    verbatim_span   TEXT,
    created_at      TEXT NOT NULL DEFAULT (to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS'))
);

CREATE INDEX IF NOT EXISTS idx_provenance_frame
    ON memory_provenance(frame_id);
CREATE INDEX IF NOT EXISTS idx_provenance_scope_ref
    ON memory_provenance(tenant_id, principal_id, namespace, source_ref);

CREATE TABLE IF NOT EXISTS memory_tombstones (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       TEXT NOT NULL DEFAULT 'local',
    principal_id    TEXT NOT NULL DEFAULT 'default',
    namespace       TEXT NOT NULL DEFAULT 'personal',
    content_hash    TEXT NOT NULL,
    reason          TEXT NOT NULL CHECK(reason IN ('dismissed','forgotten','retracted')),
    source_event_id TEXT,
    source_ref      TEXT,
    frame_id        BIGINT,
    created_at      TEXT NOT NULL DEFAULT (to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS'))
);

CREATE INDEX IF NOT EXISTS idx_tombstones_scope_hash
    ON memory_tombstones(tenant_id, principal_id, namespace, content_hash);
CREATE INDEX IF NOT EXISTS idx_tombstones_scope_event
    ON memory_tombstones(tenant_id, principal_id, namespace, source_event_id);

CREATE TABLE IF NOT EXISTS scope_generations (
    tenant_id    TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    namespace    TEXT NOT NULL,
    generation   INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT NOT NULL DEFAULT (to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS')),
    PRIMARY KEY (tenant_id, principal_id, namespace)
);

CREATE TABLE IF NOT EXISTS memory_jobs (
    id                BIGSERIAL PRIMARY KEY,
    tenant_id         TEXT NOT NULL DEFAULT 'local',
    principal_id      TEXT NOT NULL DEFAULT 'default',
    namespace         TEXT NOT NULL DEFAULT 'personal',
    job_type          TEXT NOT NULL DEFAULT 'ingest_event',
    dedup_key         TEXT NOT NULL,
    payload           TEXT NOT NULL DEFAULT '{}',
    status            TEXT NOT NULL DEFAULT 'queued'
        CHECK(status IN ('queued','leased','succeeded','failed','dead_letter','aborted_stale','fenced')),
    attempt_count     INTEGER NOT NULL DEFAULT 0,
    max_attempts      INTEGER NOT NULL DEFAULT 3,
    scope_generation  INTEGER NOT NULL DEFAULT 0,
    worker_id         TEXT,
    lease_expires_at  TEXT,
    available_at      TEXT NOT NULL DEFAULT (to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS')),
    llm_calls_used    INTEGER NOT NULL DEFAULT 0,
    max_llm_calls     INTEGER NOT NULL DEFAULT 6,
    tokens_used       INTEGER NOT NULL DEFAULT 0,
    max_tokens        INTEGER NOT NULL DEFAULT 8000,
    last_error        TEXT,
    created_at        TEXT NOT NULL DEFAULT (to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS')),
    updated_at        TEXT NOT NULL DEFAULT (to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS')),
    UNIQUE(tenant_id, principal_id, namespace, job_type, dedup_key)
);

CREATE INDEX IF NOT EXISTS idx_memory_jobs_status_lease
    ON memory_jobs(status, lease_expires_at);
CREATE INDEX IF NOT EXISTS idx_memory_jobs_scope
    ON memory_jobs(tenant_id, principal_id, namespace, status);
