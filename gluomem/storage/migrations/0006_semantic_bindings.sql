-- No historical conversion: legacy rows keep unknown semantics.
CREATE TABLE memory_claim_semantics (
    provenance_id INTEGER PRIMARY KEY REFERENCES memory_provenance(id) ON DELETE CASCADE,
    frame_id INTEGER NOT NULL REFERENCES memory_frames(id) ON DELETE CASCADE,
    tenant_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    contract TEXT NOT NULL CHECK (contract = 'semantic-state-writer-v4'),
    meaning_json TEXT NOT NULL,
    binding_token TEXT NOT NULL
);
CREATE INDEX idx_claim_semantics_scope_frame ON memory_claim_semantics
    (tenant_id, principal_id, namespace, frame_id);
