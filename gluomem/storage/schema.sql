-- gluomem — SQLite schema.
--
-- One *.mind SQLite file (WAL, FK on).
--
-- The two vec0 virtual tables (memory_frames_vec, memory_frame_chunks_vec) are
-- NOT declared here: they are created PROGRAMMATICALLY at runtime by MindDB
-- with dim = active embedder dim (never hardcoded in this .sql).

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY AUTOINCREMENT,
  gop_id TEXT UNIQUE NOT NULL, project_id TEXT, status TEXT DEFAULT 'active',
  started_at TEXT DEFAULT (datetime('now')), ended_at TEXT, summary TEXT);

CREATE TABLE IF NOT EXISTS memory_frames (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  frame_type TEXT NOT NULL CHECK(frame_type IN('I','P','B')),
  gop_id TEXT NOT NULL REFERENCES sessions(gop_id),
  t INTEGER NOT NULL DEFAULT 0,
  base_frame_id INTEGER REFERENCES memory_frames(id),         -- P/B → base I
  content TEXT NOT NULL,
  importance TEXT NOT NULL DEFAULT 'normal'
    CHECK(importance IN('critical','important','normal','temporary','deprecated')),
  source TEXT NOT NULL DEFAULT 'user_stated'
    CHECK(source IN('user_stated','tool_verified','agent_inferred','import','system','personal','workspace','team_sync')),
  trust  TEXT NOT NULL DEFAULT 'normal' CHECK(trust IN('trusted','normal','low','quarantined')),
  access_count INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),   -- INGEST time = recency-decay anchor
  event_time TEXT,                                       -- WORLD/event time = range-query anchor
  last_accessed TEXT NOT NULL DEFAULT (datetime('now')),
  content_hash TEXT);
CREATE INDEX IF NOT EXISTS idx_frames_gop_t ON memory_frames(gop_id,t);
CREATE INDEX IF NOT EXISTS idx_frames_type  ON memory_frames(frame_type,gop_id);
CREATE INDEX IF NOT EXISTS idx_frames_base  ON memory_frames(base_frame_id);
CREATE INDEX IF NOT EXISTS idx_frames_hash  ON memory_frames(content_hash);
CREATE INDEX IF NOT EXISTS idx_frames_event ON memory_frames(event_time);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_frames_fts USING fts5(content, content_rowid='id', tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS memory_frame_chunks (id INTEGER PRIMARY KEY AUTOINCREMENT,
  frame_id INTEGER NOT NULL REFERENCES memory_frames(id) ON DELETE CASCADE,
  chunk_idx INTEGER NOT NULL, content TEXT NOT NULL, char_start INTEGER, char_end INTEGER,
  UNIQUE(frame_id, chunk_idx));
-- Constituents of derived frames (compaction). source_frame_id has no FK: the
-- merged source row is deleted; this keeps its identity + provenance, not text.
-- Added after 0.0.1; IF NOT EXISTS upgrades older files on open.
CREATE TABLE IF NOT EXISTS memory_frame_lineage (id INTEGER PRIMARY KEY AUTOINCREMENT,
  derived_frame_id INTEGER NOT NULL REFERENCES memory_frames(id) ON DELETE CASCADE,
  source_frame_id INTEGER NOT NULL, source_frame_type TEXT NOT NULL,
  source_content_hash TEXT, source_source TEXT NOT NULL, source_trust TEXT NOT NULL,
  source_created_at TEXT NOT NULL, source_event_time TEXT,
  operation TEXT NOT NULL, recorded_at TEXT NOT NULL DEFAULT (datetime('now')));
CREATE INDEX IF NOT EXISTS idx_lineage_derived ON memory_frame_lineage(derived_frame_id);
CREATE INDEX IF NOT EXISTS idx_lineage_source  ON memory_frame_lineage(source_frame_id);
-- vec0 tables created with dim = active embedder dim (never hardcoded):
--   memory_frames_vec(embedding float[<dim>]) ; memory_frame_chunks_vec(embedding float[<dim>])

CREATE TABLE IF NOT EXISTS knowledge_entities (id INTEGER PRIMARY KEY AUTOINCREMENT, entity_type TEXT, name TEXT,
  properties TEXT DEFAULT '{}', valid_from TEXT DEFAULT (datetime('now')), valid_to TEXT,
  recorded_at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS knowledge_relations (id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id INTEGER REFERENCES knowledge_entities(id), target_id INTEGER REFERENCES knowledge_entities(id),
  relation_type TEXT, confidence REAL DEFAULT 1.0, properties TEXT DEFAULT '{}',
  valid_from TEXT DEFAULT (datetime('now')), valid_to TEXT, recorded_at TEXT DEFAULT (datetime('now')),
  UNIQUE(source_id, target_id, relation_type));
-- Closed validity intervals of a relation that was retired and later
-- re-asserted after a gap. knowledge_relations holds the current interval;
-- this table keeps earlier ones so as_of(T) inside the gap stays false.
-- Added after 0.0.1; IF NOT EXISTS upgrades older files on open.
CREATE TABLE IF NOT EXISTS knowledge_relation_history (id INTEGER PRIMARY KEY AUTOINCREMENT,
  relation_id INTEGER NOT NULL REFERENCES knowledge_relations(id) ON DELETE CASCADE,
  valid_from TEXT NOT NULL, valid_to TEXT NOT NULL, recorded_at TEXT NOT NULL,
  closed_recorded_at TEXT NOT NULL DEFAULT (datetime('now')));
CREATE INDEX IF NOT EXISTS idx_rel_history_rel ON knowledge_relation_history(relation_id);
CREATE TABLE IF NOT EXISTS kg_entity_frames (entity_id INTEGER REFERENCES knowledge_entities(id) ON DELETE CASCADE,
  frame_id INTEGER REFERENCES memory_frames(id) ON DELETE CASCADE, PRIMARY KEY(entity_id, frame_id));
