# Lifecycle/write/KG fixes

Observable: scoped, crash-safe SourceEvent delivery; forgotten, corrected, retracted and dismissed plaintext cannot be recalled or exported, and suppression survives case/punctuation replay.

Owned files: frames (except eligibility), write/events.py, admit path in write/gate.py, write/classify.py, write/cognify.py, kg/graph.py, lifecycle configuration additions, SQLite/Postgres bridge and suppression DDL, tests/conftest.py and lifecycle contract tests. Preserve service import paths by re-exporting frame-layer helpers.

Delete first: unused SuppressionTombstone, classifier duplication, impossible decoder fallbacks and indexable-text wrapper. Retain set_status and importance alias if outside callers need them. Replace manually seeded shared-provenance test with actual ingestion. Add one shared no-recall assertion covering database projections, payloads, KG, reconstruction, list/export and lineage; parameterized regression contracts cover lifecycle transitions and crash/replay.

Implementation: frame-layer provenance/review/generation modules, prepared embeddings passed into nested transactional writes, conflict-safe event insertion, scope-wide event-claim dedup, collision-free source GOPs, normalized suppression keys, set-based KG orphan removal and batched list provenance. Payload redaction conservatively redacts the whole affected envelope, preserving hashes and replay receipts.

Verification: uv run ruff check . && uv run pyrefly check && uv run pytest -q. Report schema/API deltas, commits and any unproven acceptance target. No push or PR (parent owns integration).

Compatibility decision: normalized keys live in new `memory_frame_keys` and `memory_tombstone_keys` tables rather than new columns on existing tables. This keeps pre-existing SQLite files readable without editing the storage-backend migration code owned by another branch. Tombstone keys are FK-attached to tombstones; frame keys retain only normalized hashes so forgetting corrected ancestors can suppress variants without keeping their text. Existing hash-only tombstones cannot acquire a normalized key after their text was already erased.

Contract consolidation: delete the four lifecycle contract files (`test_correction_lineage`, `test_deletion_revocation`, `test_review_queue`, `test_source_events`) and replace them with `tests/contracts/test_lifecycle.py`, sharing the root no-recall fixture. Preserve `test_job_recovery.py` untouched.

Measurement: `org dev census --diff 30b7d4a --repo ...` reports zeros because it is KVARK-specific; `org dev perf ingest --repo ...` has no GluoMem Locust scenario. Use a local baseline archive, deterministic SQLite query counts and timings, plus a Python AST LOC/test-function census instead. PostgreSQL runtime and physical disk/WAL/backup erasure remain unverified.

Worker seam: `ingest_source_event(write_guard=...)` invokes the optional guard after the receipt INSERT and before claim writes, inside the same transaction. Workers can re-check/lock the generation here without wrapping embedding computation in a transaction; guard failures roll back the receipt too. Pure receipt hits do not write and need no guard.

Measured on 100 SQLite memories, median of 30 repetitions against archived base `30b7d4a`: list SQL calls 102 → 3 and 1.842 → 1.794 ms; export SQL calls 503 → 205 and 4.794 → 4.254 ms. Physical LOC: source 22,521 → 22,277 (-244), tests 9,461 → 9,315 (-146). Test functions: 218 → 228 (+10), needed for previously uncovered crash rollback, invalid review transitions, scoped delivery, variant suppression, shared-source ingestion and KG retirement.
