"""Controlled generation/session lock inversion regressions on real PostgreSQL."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from gluomem.frames.generation import current_scope_generation
from gluomem.frames.lifecycle import correct_memory, forget_memory, forget_scope, retract_source
from gluomem.frames.source_fences import SourceRetractedError
from gluomem.frames.store import FrameStore
from gluomem.storage.db import MindDB
from gluomem.write.events import SourceEvent, ingest_source_event

pytestmark = pytest.mark.postgres


@pytest.mark.parametrize(
    "operation",
    [
        "forget",
        "correct",
        "forget_scope",
        "accept",
        "edit",
        "dismiss",
        "pin",
        "promote",
        "sweep",
        "admit_verified",
    ],
)
def test_lifecycle_then_retraction_uses_one_lock_order(
    mind_db, frame_store, cfg, pg_dsn, monkeypatch, operation
):
    """Pause A holding a session/frame row; observe B blocked by A, then resume.

    Before the fix B owns generation while waiting for A's session. A then
    requests generation and PostgreSQL detects the cycle. After the fix B waits
    on generation itself, so both privacy operations commit. No sleep establishes
    the ordering: events and pg_blocking_pids are the synchronization barriers.
    """
    import psycopg

    event = SourceEvent(
        event_id="race",
        source_ref="doc:lock-order",
        content="Private original",
        require_review=operation in ("accept", "edit", "dismiss"),
        kind="episode" if operation in ("promote", "sweep") else "fact",
    )
    frame = ingest_source_event(event, store=frame_store, cfg=cfg).frames[0]
    before = current_scope_generation(mind_db.backend, frame_store.scope)
    other = MindDB(
        backend_type="postgres", dsn=pg_dsn, postgres_schema=mind_db.backend.postgres_schema
    )
    other_store = FrameStore(other, frame_store.embedder, chunking=cfg.chunking)
    pid_a = mind_db.scalar("SELECT pg_backend_pid()")
    pid_b = other.scalar("SELECT pg_backend_pid()")
    paused, resume = Event(), Event()
    execute = mind_db.backend.execute
    row_update = (
        "UPDATE sessions "
        if operation in ("forget", "correct", "forget_scope", "dismiss")
        else "UPDATE memory_frames "
    )

    def pause_after_row_lock(sql, params=()):
        result = execute(sql, params)
        if not paused.is_set() and sql.startswith(row_update):
            paused.set()
            assert resume.wait(10), "Controller never released lifecycle transaction"
        return result

    monkeypatch.setattr(mind_db.backend, "execute", pause_after_row_lock)

    def lifecycle():
        from gluomem.frames.expiry import sweep_expired_episodes
        from gluomem.frames.review import pin_memory, promote_episode_to_fact, review_memory
        from gluomem.write.gate import admit, make_scanner

        with mind_db.transaction():
            mind_db.execute("SET LOCAL lock_timeout = '8s'")
            if operation == "forget":
                return forget_memory(frame_store, frame.id)
            if operation == "correct":
                return correct_memory(frame_store, frame.id, "Harmless correction")
            if operation == "forget_scope":
                return forget_scope(frame_store)
            if operation in ("accept", "edit", "dismiss"):
                return review_memory(
                    frame_store, frame.id, action=operation, edited_content="Edited value"
                )
            if operation == "pin":
                return pin_memory(frame_store, frame.id)
            if operation == "promote":
                return promote_episode_to_fact(frame_store, frame.id)
            if operation == "sweep":
                return sweep_expired_episodes(frame_store, now_iso="2100-01-01")
            return admit(
                frame.content,
                gop=frame.gop_id,
                scanner=make_scanner(cfg),
                store=frame_store,
                kg=None,
                cfg=cfg,
                verified_by_user=True,
            )

    def retract():
        with other.transaction():
            other.execute("SET LOCAL lock_timeout = '8s'")
            return retract_source(other_store, event.source_ref)

    try:
        with psycopg.connect(pg_dsn, autocommit=True) as observer:
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(lifecycle)
                second = None
                try:
                    assert paused.wait(5), "Lifecycle never reached its row-lock barrier"
                    second = pool.submit(retract)
                    deadline = time.monotonic() + 5
                    while True:
                        blockers = observer.execute(
                            "SELECT pg_blocking_pids(%s)", (pid_b,)
                        ).fetchone()[0]
                        if pid_a in blockers:
                            break
                        assert time.monotonic() < deadline, "Retraction never blocked on lifecycle"
                        # Poll only; elapsed time does not determine the schedule.
                        resume.wait(0.01)
                finally:
                    resume.set()
                outcomes = []
                for future in (first, second):
                    if future is not None:
                        try:
                            outcomes.append(future.result(timeout=10))
                        except Exception as exc:
                            outcomes.append(exc)
                assert len(outcomes) == 2
                assert not any(isinstance(x, Exception) for x in outcomes), repr(outcomes)
        assert current_scope_generation(mind_db.backend, frame_store.scope) == before + 2
        assert frame_store.get_frame(frame.id).content == "[forgotten]"
        assert mind_db.scalar("SELECT COUNT(*) FROM memory_frames WHERE status = 'active'") == 0
        assert mind_db.scalar("SELECT COUNT(*) FROM source_retractions") == 1
        assert mind_db.scalar("SELECT payload_json FROM source_events") == "{}"
        assert (
            mind_db.scalar("SELECT summary FROM sessions WHERE gop_id = ?", (frame.gop_id,)) is None
        )
        with pytest.raises(SourceRetractedError):
            ingest_source_event(
                event.model_copy(update={"revision": 2, "force_reassert": True}),
                store=frame_store,
                cfg=cfg,
            )
    finally:
        resume.set()
        other.close()


@pytest.mark.parametrize("operation", ["forget", "correct", "forget_scope", "admit_verified"])
def test_retraction_then_lifecycle_does_not_restore_erased_memory(
    mind_db, frame_store, cfg, pg_dsn, monkeypatch, operation
):
    """The opposite schedule must revalidate after waiting, never revive a frame."""
    import psycopg

    from gluomem.write.gate import admit, make_scanner

    event = SourceEvent(event_id="reverse", source_ref="doc:reverse", content="Private original")
    frame = ingest_source_event(event, store=frame_store, cfg=cfg).frames[0]
    before = current_scope_generation(mind_db.backend, frame_store.scope)
    other = MindDB(
        backend_type="postgres", dsn=pg_dsn, postgres_schema=mind_db.backend.postgres_schema
    )
    other_store = FrameStore(other, frame_store.embedder, chunking=cfg.chunking)
    pid_a = mind_db.scalar("SELECT pg_backend_pid()")
    pid_b = other.scalar("SELECT pg_backend_pid()")
    paused, resume = Event(), Event()
    execute = mind_db.backend.execute

    def pause_retraction(sql, params=()):
        result = execute(sql, params)
        if not paused.is_set() and sql.startswith("UPDATE sessions "):
            paused.set()
            assert resume.wait(10)
        return result

    monkeypatch.setattr(mind_db.backend, "execute", pause_retraction)

    def lifecycle():
        with other.transaction():
            other.execute("SET LOCAL lock_timeout = '8s'")
            if operation == "forget":
                return forget_memory(other_store, frame.id)
            if operation == "forget_scope":
                return forget_scope(other_store)
            if operation == "correct":
                with pytest.raises(ValueError, match="Only active or candidate"):
                    correct_memory(other_store, frame.id, "Harmless correction")
            else:
                with pytest.raises(ValueError, match="changed before admission"):
                    admit(
                        frame.content,
                        gop=frame.gop_id,
                        scanner=make_scanner(cfg),
                        store=other_store,
                        kg=None,
                        cfg=cfg,
                        verified_by_user=True,
                    )

    try:
        with psycopg.connect(pg_dsn, autocommit=True) as observer:
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(retract_source, frame_store, event.source_ref)
                try:
                    assert paused.wait(5)
                    second = pool.submit(lifecycle)
                    deadline = time.monotonic() + 5
                    while (
                        pid_a
                        not in observer.execute("SELECT pg_blocking_pids(%s)", (pid_b,)).fetchone()[
                            0
                        ]
                    ):
                        assert time.monotonic() < deadline, "Lifecycle never waited for retraction"
                        resume.wait(0.01)
                finally:
                    resume.set()
                first.result(timeout=10)
                second.result(timeout=10)
        delta = 2 if operation in ("forget", "forget_scope") else 1
        assert current_scope_generation(mind_db.backend, frame_store.scope) == before + delta
        erased = frame_store.get_frame(frame.id)
        assert erased.status == "forgotten"
        assert erased.content == "[forgotten]"
        assert mind_db.scalar("SELECT COUNT(*) FROM memory_frames_fts") == 0
        assert mind_db.scalar("SELECT COUNT(*) FROM memory_frames_vec") == 0
        assert mind_db.scalar("SELECT COUNT(*) FROM source_retractions") == 1
    finally:
        resume.set()
        other.close()
