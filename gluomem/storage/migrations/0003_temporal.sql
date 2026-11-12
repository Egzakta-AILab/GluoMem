-- SQLite v2 -> v3. Existing rows stay undated: event_time is not learned time.
CREATE TABLE IF NOT EXISTS memory_frame_temporal (
    frame_id          INTEGER PRIMARY KEY REFERENCES memory_frames(id),
    tenant_id         TEXT NOT NULL,
    principal_id      TEXT NOT NULL,
    namespace         TEXT NOT NULL,
    operation         TEXT NOT NULL,
    previous_frame_id INTEGER UNIQUE REFERENCES memory_frames(id),
    effective_at      TEXT,
    observed_at       TEXT NOT NULL,
    recorded_at       TEXT NOT NULL,
    CHECK (frame_id != previous_frame_id),
    CHECK ((operation = 'observation' AND previous_frame_id IS NULL)
        OR (operation = 'transition' AND previous_frame_id IS NOT NULL AND effective_at IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS idx_temporal_scope
    ON memory_frame_temporal(tenant_id, principal_id, namespace);
