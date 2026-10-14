-- gluomem PostgreSQL + pgvector schema.
-- Applied idempotently by PostgresBackend on open; __VECTOR_DIM__ is
-- substituted with the configured embedding dimension.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id         BIGSERIAL PRIMARY KEY,
    gop_id     TEXT UNIQUE NOT NULL,
    project_id TEXT,
    status     TEXT DEFAULT 'active',
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    summary    TEXT
);

CREATE TABLE IF NOT EXISTS memory_frames (
    id            BIGSERIAL PRIMARY KEY,
    frame_type    TEXT NOT NULL CHECK(frame_type IN ('I','P','B')),
    gop_id        TEXT NOT NULL REFERENCES sessions(gop_id),
    t             INTEGER NOT NULL DEFAULT 0,
    base_frame_id BIGINT REFERENCES memory_frames(id),
    content       TEXT NOT NULL,
    importance    TEXT NOT NULL DEFAULT 'normal'
        CHECK(importance IN ('critical','important','normal','temporary','deprecated')),
    source        TEXT NOT NULL DEFAULT 'user_stated'
        CHECK(source IN ('user_stated','tool_verified','agent_inferred','import','system','personal','workspace','team_sync')),
    trust         TEXT NOT NULL DEFAULT 'normal'
        CHECK(trust IN ('trusted','normal','low','quarantined')),
    access_count  INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    event_time    TEXT,
    last_accessed TEXT NOT NULL,
    content_hash  TEXT
);

CREATE INDEX IF NOT EXISTS idx_frames_gop_t ON memory_frames(gop_id, t);
CREATE INDEX IF NOT EXISTS idx_frames_type  ON memory_frames(frame_type, gop_id);
CREATE INDEX IF NOT EXISTS idx_frames_base  ON memory_frames(base_frame_id);
CREATE INDEX IF NOT EXISTS idx_frames_hash  ON memory_frames(content_hash);
CREATE INDEX IF NOT EXISTS idx_frames_event ON memory_frames(event_time);

CREATE TABLE IF NOT EXISTS memory_frames_fts (
    rowid   BIGINT PRIMARY KEY REFERENCES memory_frames(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    tsv     tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
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

CREATE TABLE IF NOT EXISTS memory_frame_chunks_vec (
    rowid     BIGINT PRIMARY KEY REFERENCES memory_frame_chunks(id) ON DELETE CASCADE,
    embedding vector(__VECTOR_DIM__) NOT NULL
);

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
    entity_type    TEXT,
    name           TEXT,
    canonical_name TEXT,
    properties     TEXT DEFAULT '{}',
    valid_from     TEXT NOT NULL,
    valid_to       TEXT,
    recorded_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kg_entities_canonical
    ON knowledge_entities(canonical_name);

CREATE TABLE IF NOT EXISTS knowledge_relations (
    id            BIGSERIAL PRIMARY KEY,
    source_id     BIGINT REFERENCES knowledge_entities(id),
    target_id     BIGINT REFERENCES knowledge_entities(id),
    relation_type TEXT,
    confidence    DOUBLE PRECISION DEFAULT 1.0,
    properties    TEXT DEFAULT '{}',
    valid_from    TEXT NOT NULL,
    valid_to      TEXT,
    recorded_at   TEXT NOT NULL,
    UNIQUE(source_id, target_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_kg_relations_target
    ON knowledge_relations(target_id);

CREATE TABLE IF NOT EXISTS knowledge_relation_history (
    id                 BIGSERIAL PRIMARY KEY,
    relation_id        BIGINT NOT NULL REFERENCES knowledge_relations(id) ON DELETE CASCADE,
    valid_from         TEXT NOT NULL,
    valid_to           TEXT NOT NULL,
    recorded_at        TEXT NOT NULL,
    closed_recorded_at TEXT NOT NULL
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
