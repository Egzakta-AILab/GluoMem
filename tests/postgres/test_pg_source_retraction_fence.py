"""Run the same SQL source-removal contracts against live PostgreSQL."""

from __future__ import annotations

import pytest

from tests.contracts.test_source_retraction_fence import (
    test_frame_erasure_clears_affected_session_summary_only,
    test_removed_source_without_receipt_cannot_be_reasserted,
    test_repeated_retraction_and_scope_forget_preserve_fence,
    test_retraction_between_prepare_and_commit_blocks_stale_work,
    test_retraction_clears_summary_when_all_frames_have_shared_support,
    test_retraction_scrubs_pending_jobs_without_source_event_receipt,
    test_scope_forget_scrubs_session_summaries_without_cross_scope_damage,
    test_source_retraction_is_exactly_scoped,
    test_source_retraction_precedes_extractor_and_embedder_calls,
)

pytestmark = pytest.mark.postgres

__all__ = [
    "test_frame_erasure_clears_affected_session_summary_only",
    "test_removed_source_without_receipt_cannot_be_reasserted",
    "test_repeated_retraction_and_scope_forget_preserve_fence",
    "test_retraction_between_prepare_and_commit_blocks_stale_work",
    "test_retraction_clears_summary_when_all_frames_have_shared_support",
    "test_retraction_scrubs_pending_jobs_without_source_event_receipt",
    "test_scope_forget_scrubs_session_summaries_without_cross_scope_damage",
    "test_source_retraction_is_exactly_scoped",
    "test_source_retraction_precedes_extractor_and_embedder_calls",
]


def test_old_postgres_retractions_backfilled_on_upgrade(pg_dsn, pg_schema_factory):
    import psycopg
    from psycopg import sql

    from gluomem.migrations import run_downgrade, run_upgrade

    schema = pg_schema_factory(384)
    run_downgrade(pg_dsn, "0001", schema=schema)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
        conn.execute(
            "INSERT INTO memory_tombstones "
            "(tenant_id, principal_id, namespace, content_hash, reason, source_ref) "
            "VALUES ('local', 'default', 'personal', 'legacy-hash', 'retracted', 'legacy-doc')"
        )
    run_upgrade(pg_dsn, schema=schema)
    for _ in range(2):
        with psycopg.connect(pg_dsn) as conn:
            row = conn.execute(
                sql.SQL("SELECT source_ref FROM {}.source_retractions").format(
                    sql.Identifier(schema)
                )
            ).fetchone()
            assert row == ("legacy-doc",)


def test_worker_scope_lock_and_source_retraction_do_not_deadlock(
    mind_db,
    frame_store,
    pg_dsn,
    cfg,
):
    import time
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from gluomem.frames.generation import bump_scope_generation
    from gluomem.frames.lifecycle import retract_source
    from gluomem.frames.source_fences import lock_source
    from gluomem.frames.store import FrameStore
    from gluomem.storage.db import MindDB

    other = MindDB(
        backend_type="postgres", dsn=pg_dsn, postgres_schema=mind_db.backend.postgres_schema
    )
    other_store = FrameStore(other, frame_store.embedder, chunking=cfg.chunking)
    pid = other.scalar("SELECT pg_backend_pid()")
    entered = Event()

    def retract():
        entered.set()
        return retract_source(other_store, "doc:concurrent")

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with mind_db.transaction():
                mind_db.execute("SET LOCAL lock_timeout = '4s'")
                bump_scope_generation(mind_db.backend, frame_store.scope)
                future = pool.submit(retract)
                assert entered.wait(3)
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    blocked = mind_db.scalar(
                        "SELECT wait_event_type FROM pg_stat_activity WHERE pid = ?",
                        (pid,),
                    )
                    if blocked == "Lock":
                        break
                    time.sleep(0.01)
                    mind_db.execute("SELECT pg_stat_clear_snapshot()")
                else:
                    pytest.fail("Concurrent retraction never reached the expected lock wait")
                # Matches Phase2 after the worker has locked its generation row.
                lock_source(mind_db.backend, frame_store.scope, "doc:concurrent")
            assert future.result(timeout=5).source_ref == "doc:concurrent"
        assert mind_db.scalar("SELECT COUNT(*) FROM source_retractions") == 1
    finally:
        other.close()
