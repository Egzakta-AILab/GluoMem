"""Real two-connection standalone projection/retraction schedules."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from gluomem.frames.generation import current_scope_generation
from gluomem.frames.lifecycle import retract_source
from gluomem.frames.store import FrameStore
from gluomem.storage.db import MindDB
from gluomem.write.events import SourceEvent, ingest_source_event
from gluomem.write.gate import admit, make_scanner

pytestmark = pytest.mark.postgres


@pytest.mark.parametrize("flags", [{}, {"verified_by_user": True}, {"pinned": True}])
@pytest.mark.parametrize("phase", ["repair", "graph"])
def test_standalone_admission_holds_generation_through_projections(
    mind_db, frame_store, cfg, pg_dsn, monkeypatch, flags, phase, kg, assert_no_recall
):
    import psycopg

    event = SourceEvent(
        event_id="standalone", source_ref="doc:standalone", content="Private original"
    )
    frame = ingest_source_event(event, store=frame_store, cfg=cfg).frames[0]
    entity = kg.create_entity("person", event.content)
    kg.link_entity_frame(entity.id, frame.id)
    before = current_scope_generation(mind_db.backend, frame.scope)
    frame_store._delete_index_rows(frame.id)
    other = MindDB(
        backend_type="postgres", dsn=pg_dsn, postgres_schema=mind_db.backend.postgres_schema
    )
    other_store = FrameStore(other, frame_store.embedder, chunking=cfg.chunking)
    pid_a = mind_db.scalar("SELECT pg_backend_pid()")
    pid_b = other.scalar("SELECT pg_backend_pid()")
    paused, resume = Event(), Event()
    target = frame_store if phase == "repair" else kg
    method = "index_state" if phase == "repair" else "link_entity_frame"
    original = getattr(target, method)

    def barrier(*args, **kwargs):
        result = original(*args, **kwargs)
        if not paused.is_set():
            paused.set()
            assert resume.wait(10)
        return result

    monkeypatch.setattr(target, method, barrier)

    def standalone():
        return admit(
            frame.content,
            gop=frame.gop_id,
            scanner=make_scanner(cfg),
            store=frame_store,
            kg=kg,
            entity_ids=[entity.id],
            cfg=cfg,
            **flags,
        )

    blocked = False
    try:
        with psycopg.connect(pg_dsn, autocommit=True) as observer:
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(standalone)
                assert paused.wait(5)
                second = pool.submit(retract_source, other_store, event.source_ref)
                try:
                    deadline = time.monotonic() + 5
                    while not second.done():
                        if (
                            pid_a
                            in observer.execute("SELECT pg_blocking_pids(%s)", (pid_b,)).fetchone()[
                                0
                            ]
                        ):
                            blocked = True
                            break
                        assert time.monotonic() < deadline
                        resume.wait(0.01)
                finally:
                    resume.set()
                result = first.result(timeout=10)
                second.result(timeout=10)
        assert frame_store.get_frame(frame.id).content == "[forgotten]"
        assert mind_db.scalar("SELECT COUNT(*) FROM memory_frames_fts") == 0
        assert mind_db.scalar("SELECT COUNT(*) FROM memory_frame_chunks") == 0
        assert mind_db.scalar("SELECT COUNT(*) FROM memory_frames_vec") == 0
        assert mind_db.scalar("SELECT COUNT(*) FROM source_retractions") == 1
        assert mind_db.scalar("SELECT COUNT(*) FROM kg_entity_frames") == 0
        assert_no_recall(event.content, db=mind_db, store=frame_store, kg=kg)
        assert blocked, "Standalone admission released generation before its final projections"
        assert result.admitted
        assert current_scope_generation(mind_db.backend, frame.scope) == before + 1 + bool(flags)
    finally:
        resume.set()
        other.close()


@pytest.mark.parametrize("flags", [{}, {"verified_by_user": True}, {"pinned": True}])
def test_standalone_admission_waits_for_retraction_and_revalidates(
    mind_db, frame_store, cfg, pg_dsn, monkeypatch, flags, kg, assert_no_recall
):
    import psycopg

    event = SourceEvent(event_id="inverse", source_ref="doc:inverse", content="Private original")
    frame = ingest_source_event(event, store=frame_store, cfg=cfg).frames[0]
    before = current_scope_generation(mind_db.backend, frame.scope)
    other = MindDB(
        backend_type="postgres", dsn=pg_dsn, postgres_schema=mind_db.backend.postgres_schema
    )
    other_store = FrameStore(other, frame_store.embedder, chunking=cfg.chunking)
    pid_a = mind_db.scalar("SELECT pg_backend_pid()")
    pid_b = other.scalar("SELECT pg_backend_pid()")
    paused, resume = Event(), Event()
    execute = mind_db.backend.execute

    def retraction_barrier(sql, params=()):
        result = execute(sql, params)
        if not paused.is_set() and sql.startswith("UPDATE sessions "):
            paused.set()
            assert resume.wait(10)
        return result

    monkeypatch.setattr(mind_db.backend, "execute", retraction_barrier)

    def standalone():
        with pytest.raises(ValueError, match="changed before admission"):
            admit(
                frame.content,
                gop=frame.gop_id,
                scanner=make_scanner(cfg),
                store=other_store,
                kg=None,
                cfg=cfg,
                **flags,
            )

    try:
        with psycopg.connect(pg_dsn, autocommit=True) as observer:
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(retract_source, frame_store, event.source_ref)
                try:
                    assert paused.wait(5)
                    second = pool.submit(standalone)
                    deadline = time.monotonic() + 5
                    while (
                        pid_a
                        not in observer.execute("SELECT pg_blocking_pids(%s)", (pid_b,)).fetchone()[
                            0
                        ]
                    ):
                        assert not second.done(), "Admission did not wait for generation"
                        assert time.monotonic() < deadline
                        resume.wait(0.01)
                finally:
                    resume.set()
                first.result(timeout=10)
                second.result(timeout=10)
        assert current_scope_generation(mind_db.backend, frame.scope) == before + 1
        assert mind_db.scalar("SELECT COUNT(*) FROM source_retractions") == 1
        assert_no_recall(event.content, db=mind_db, store=frame_store, kg=kg)
    finally:
        resume.set()
        other.close()


@pytest.mark.parametrize("repair", ["ensure_indexed", "repair_indexes", "rebuild_index"])
def test_standalone_public_repair_rejects_concurrently_erased_snapshot(
    mind_db, frame_store, cfg, pg_dsn, monkeypatch, repair, kg, assert_no_recall
):
    event = SourceEvent(event_id="public", source_ref="doc:public", content="Private original")
    frame = ingest_source_event(event, store=frame_store, cfg=cfg).frames[0]
    frame_store._delete_index_rows(frame.id)
    other = MindDB(
        backend_type="postgres", dsn=pg_dsn, postgres_schema=mind_db.backend.postgres_schema
    )
    other_store = FrameStore(other, frame_store.embedder, chunking=cfg.chunking)
    original = frame_store.embedder.embed
    paused, resume = Event(), Event()

    def prepare_barrier(text):
        result = original(text)
        if not paused.is_set():
            paused.set()
            assert resume.wait(10)
        return result

    monkeypatch.setattr(frame_store.embedder, "embed", prepare_barrier)

    def standalone():
        if repair == "ensure_indexed":
            return frame_store.ensure_indexed(frame)
        if repair == "repair_indexes":
            return frame_store.repair_indexes()
        with pytest.raises(ValueError, match="changed during index rebuild"):
            return FrameStore.rebuild_index(mind_db, frame_store.embedder, chunking=cfg.chunking)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(standalone)
            try:
                assert paused.wait(5)
                # The repair's embedding phase owns no transaction. Retraction
                # must actually commit before its prepared snapshot is published.
                assert not mind_db.backend.in_transaction
                pool.submit(retract_source, other_store, event.source_ref).result(timeout=5)
            finally:
                resume.set()
            result = first.result(timeout=10)
        if repair == "ensure_indexed":
            assert result is False
        if repair == "repair_indexes":
            assert result == []
        assert mind_db.scalar("SELECT COUNT(*) FROM source_retractions") == 1
        assert_no_recall(event.content, db=mind_db, store=frame_store, kg=kg)
    finally:
        resume.set()
        other.close()
