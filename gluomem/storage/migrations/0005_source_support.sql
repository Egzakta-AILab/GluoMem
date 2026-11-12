-- Exact revocable source support; no plaintext and no historical backfill.
CREATE TABLE memory_source_support (
    provenance_id INTEGER PRIMARY KEY REFERENCES memory_provenance(id) ON DELETE CASCADE,
    tenant_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    event_payload_hash TEXT NOT NULL,
    char_start INTEGER NOT NULL CHECK(char_start >= 0),
    char_end INTEGER NOT NULL CHECK(char_end > char_start),
    message_time TEXT,
    binding_hash TEXT NOT NULL
);
CREATE INDEX idx_source_support_scope ON memory_source_support(tenant_id, principal_id, namespace);
