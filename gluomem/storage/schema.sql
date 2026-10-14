-- gluomem SQLite schema.
-- Applied idempotently (IF NOT EXISTS) by MindDB on every open.
-- The two vec0 virtual tables (memory_frames_vec, memory_frame_chunks_vec) are
-- created by SqliteBackend after loading this file because their float[<dim>]
-- column dimension is parameterized at runtime from config.

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    gop_id     TEXT UNIQUE NOT NULL,
    project_id TEXT,
    status     TEXT DEFAULT 'active',
    started_at TEXT DEFAULT (datetime('now')),
    ended_at   TEXT,
    summary    TEXT
);

CREATE TABLE IF NOT EXISTS memory_frames (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_type    TEXT NOT NULL CHECK(frame_type IN ('I','P','B')),
    gop_id        TEXT NOT NULL REFERENCES sessions(gop_id),
    t             INTEGER NOT NULL DEFAULT 0,
    base_frame_id INTEGER REFERENCES memory_frames(id),
    content       TEXT NOT NULL,
    importance    TEXT NOT NULL DEFAULT 'normal'
        CHECK(importance IN ('critical','important','normal','temporary','deprecated')),
    source        TEXT NOT NULL DEFAULT 'user_stated'
        CHECK(source IN ('user_stated','tool_verified','agent_inferred','import','system','personal','workspace','team_sync')),
    trust         TEXT NOT NULL DEFAULT 'normal'
        CHECK(trust IN ('trusted','normal','low','quarantined')),
    access_count  INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    event_time    TEXT,
    last_accessed TEXT NOT NULL DEFAULT (datetime('now')),
    content_hash  TEXT
);

CREATE INDEX IF NOT EXISTS idx_frames_gop_t ON memory_frames(gop_id, t);
CREATE INDEX IF NOT EXISTS idx_frames_type  ON memory_frames(frame_type, gop_id);
CREATE INDEX IF NOT EXISTS idx_frames_base  ON memory_frames(base_frame_id);
CREATE INDEX IF NOT EXISTS idx_frames_hash  ON memory_frames(content_hash);
CREATE INDEX IF NOT EXISTS idx_frames_event ON memory_frames(event_time);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_frames_fts USING fts5(
    content,
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS memory_frame_chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id   INTEGER NOT NULL REFERENCES memory_frames(id) ON DELETE CASCADE,
    chunk_idx  INTEGER NOT NULL,
    content    TEXT NOT NULL,
    char_start INTEGER,
    char_end   INTEGER,
    UNIQUE(frame_id, chunk_idx)
);

CREATE TABLE IF NOT EXISTS memory_frame_lineage (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    derived_frame_id    INTEGER NOT NULL REFERENCES memory_frames(id) ON DELETE CASCADE,
    source_frame_id     INTEGER NOT NULL,
    source_frame_type   TEXT NOT NULL,
    source_content_hash TEXT,
    source_source       TEXT NOT NULL,
    source_trust        TEXT NOT NULL,
    source_created_at   TEXT NOT NULL,
    source_event_time   TEXT,
    operation           TEXT NOT NULL,
    recorded_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_lineage_derived ON memory_frame_lineage(derived_frame_id);
CREATE INDEX IF NOT EXISTS idx_lineage_source  ON memory_frame_lineage(source_frame_id);

CREATE TABLE IF NOT EXISTS knowledge_entities (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type    TEXT,
    name           TEXT,
    canonical_name TEXT,
    properties     TEXT DEFAULT '{}',
    valid_from     TEXT DEFAULT (datetime('now')),
    valid_to       TEXT,
    recorded_at    TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_kg_entities_canonical
    ON knowledge_entities(canonical_name);

CREATE TABLE IF NOT EXISTS knowledge_relations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id     INTEGER REFERENCES knowledge_entities(id),
    target_id     INTEGER REFERENCES knowledge_entities(id),
    relation_type TEXT,
    confidence    REAL DEFAULT 1.0,
    properties    TEXT DEFAULT '{}',
    valid_from    TEXT DEFAULT (datetime('now')),
    valid_to      TEXT,
    recorded_at   TEXT DEFAULT (datetime('now')),
    UNIQUE(source_id, target_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_kg_relations_target
    ON knowledge_relations(target_id);

CREATE TABLE IF NOT EXISTS knowledge_relation_history (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    relation_id        INTEGER NOT NULL REFERENCES knowledge_relations(id) ON DELETE CASCADE,
    valid_from         TEXT NOT NULL,
    valid_to           TEXT NOT NULL,
    recorded_at        TEXT NOT NULL,
    closed_recorded_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_kg_rel_history_rel
    ON knowledge_relation_history(relation_id);

CREATE TABLE IF NOT EXISTS kg_entity_frames (
    entity_id INTEGER REFERENCES knowledge_entities(id) ON DELETE CASCADE,
    frame_id  INTEGER REFERENCES memory_frames(id) ON DELETE CASCADE,
    PRIMARY KEY(entity_id, frame_id)
);

CREATE INDEX IF NOT EXISTS idx_kg_entity_frames_frame
    ON kg_entity_frames(frame_id);
