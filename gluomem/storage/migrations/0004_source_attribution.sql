-- SQLite v3 -> v4. Trusted attribution is unknown for all prior receipts.
ALTER TABLE memory_provenance ADD COLUMN message_role TEXT;
ALTER TABLE memory_provenance ADD COLUMN observed_at TEXT;
