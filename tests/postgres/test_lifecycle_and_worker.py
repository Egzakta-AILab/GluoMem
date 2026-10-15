"""Live PostgreSQL 18 erasure, lifecycle, and worker fencing contracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any

import pytest

from gluomem.config.app_config import AppConfig
from gluomem.frames.content_hash import suppression_key
from gluomem.frames.lifecycle import (
    correct_memory,
    forget_memory,
    forget_scope,
    get_scope_generation,
    retract_source,
)
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.service.jobs import JobQueue, LeaseLostError, MemoryJob
from gluomem.service.worker import MemoryWorker
from gluomem.storage.db import MindDB
from gluomem.storage.scope import MemoryScope, scope_sql
from gluomem.write.events import SourceEvent, ingest_source_event

pytestmark = pytest.mark.postgres


def _evt(text: str, event_id: str, *, scope: MemoryScope, source_ref: str = "doc") -> SourceEvent:
    return SourceEvent(event_id=event_id, source_ref=source_ref, content=text, scope=scope)


def test_postgres_hard_erasure_and_sibling_scope_isolation(
    mind_db: MindDB,
    frame_store: FrameStore,
    kg: KnowledgeGraph,
    cfg: AppConfig,
    assert_no_recall: Any,
) -> None:
    target = MemoryScope("egzakta", "alice", "personal")
    sibling = MemoryScope("egzakta", "bob", "personal")
    alice_store = FrameStore(mind_db, frame_store.embedder, chunking=cfg.chunking, scope=target)
    bob_store = FrameStore(mind_db, frame_store.embedder, chunking=cfg.chunking, scope=sibling)
    alice_kg = KnowledgeGraph(mind_db, scope=target)
    bob_kg = KnowledgeGraph(mind_db, scope=sibling)
    queue = JobQueue(mind_db.backend)

    # 1. Sibling scope setup (must remain untouched by Alice's lifecycle operations)
    bob_text = "Bob private vault key OMEGA."
    bob_evt = _evt(bob_text, "bob-1", scope=sibling, source_ref="bob-doc")
    bob_frame = ingest_source_event(bob_evt, store=bob_store, cfg=cfg).frames[0]
    bob_ent = bob_kg.create_entity("secret", "OMEGA", scope=sibling)
    bob_kg.link_entity_frame(bob_ent.id, bob_frame.id, scope=sibling)
    queue.enqueue(sibling, dedup_key="bob-1:1", payload=bob_evt.model_dump(mode="json"))

    # 2. forget_memory on Alice's frame + KG + job + source_event + tombstone suppression
    secret_a = "Alice vault passcode ZEPHYR."
    evt_a = _evt(secret_a, "alice-a", scope=target, source_ref="doc-a")
    frame_a = ingest_source_event(evt_a, store=alice_store, cfg=cfg).frames[0]
    queue.enqueue(target, dedup_key="alice-a:1", payload=evt_a.model_dump(mode="json"))
    ent_zephyr = alice_kg.create_entity("secret", "ZEPHYR", scope=target)
    ent_vault = alice_kg.create_entity("system", "AliceVault", scope=target)
    alice_kg.link_entity_frame(ent_zephyr.id, frame_a.id, scope=target)
    alice_kg.link_entity_frame(ent_vault.id, frame_a.id, scope=target)
    rel_a = alice_kg.create_relation(ent_vault.id, ent_zephyr.id, "guarded_by", scope=target)
    alice_kg.link_relation_frame(rel_a.id, frame_a.id, scope=target)

    forget_res = forget_memory(alice_store, frame_a.id, scope=target)
    assert set(forget_res.retired_entity_ids) == {ent_zephyr.id, ent_vault.id}
    assert forget_res.retired_relation_ids == [rel_a.id]
    assert_no_recall("ZEPHYR", db=mind_db, store=alice_store, kg=alice_kg)
    assert (
        mind_db.scalar(
            "SELECT COUNT(*) FROM memory_tombstone_keys WHERE suppression_key = ?",
            (suppression_key(secret_a),),
        )
        == 1
    )
    assert (
        ingest_source_event(
            _evt("alice vault passcode zephyr!", "alice-a-replay", scope=target),
            store=alice_store,
            cfg=cfg,
        ).frames
        == []
    )

    # 3. correct_memory scrubs superseded plaintext while keeping new value searchable
    old_claim = "Deployment freeze starts Wednesday."
    evt_b = _evt(old_claim, "alice-b", scope=target, source_ref="doc-b")
    frame_b = ingest_source_event(evt_b, store=alice_store, cfg=cfg).frames[0]
    corr = correct_memory(
        alice_store, frame_b.id, "Deployment freeze starts Thursday.", scope=target
    )
    assert corr.replacement_frame.supersedes == frame_b.id
    assert_no_recall("Wednesday", db=mind_db, store=alice_store, kg=alice_kg)
    assert mind_db.fts.search("Thursday", 10, scope=target) == [corr.replacement_frame.id]

    # 4. retract_source scrubs exclusively supported frames and retains shared ones
    sole_text = "Retracted runbook token QUASAR."
    evt_c = _evt(sole_text, "alice-c", scope=target, source_ref="retract-doc")
    sole_frame = ingest_source_event(evt_c, store=alice_store, cfg=cfg).frames[0]
    shared_text = "Shared architecture note on PostgreSQL."
    shared_frame = ingest_source_event(
        _evt(shared_text, "alice-shared-1", scope=target, source_ref="retract-doc"),
        store=alice_store,
        cfg=cfg,
    ).frames[0]
    ingest_source_event(
        _evt(shared_text, "alice-shared-2", scope=target, source_ref="keep-doc"),
        store=alice_store,
        cfg=cfg,
    )
    retract_res = retract_source(alice_store, "retract-doc", scope=target)
    assert retract_res.forgotten_frame_ids == [sole_frame.id]
    assert retract_res.retained_frame_ids == [shared_frame.id]
    assert_no_recall("QUASAR", db=mind_db, store=alice_store, kg=alice_kg)
    assert mind_db.fts.search("PostgreSQL", 10, scope=target) == [shared_frame.id]

    # 5. forget_scope erases all remaining frames, chunks, vectors, FTS, and KG rows for Alice
    unlinked_ent = alice_kg.create_entity("person", "AliceUnlinked", scope=target)
    scope_res = forget_scope(alice_store, scope=target)
    assert set(scope_res.forgotten_frame_ids) >= {corr.replacement_frame.id, shared_frame.id}
    assert unlinked_ent.id in scope_res.retired_entity_ids
    assert_no_recall("Thursday", db=mind_db, store=alice_store, kg=alice_kg)
    assert_no_recall("PostgreSQL", db=mind_db, store=alice_store, kg=alice_kg)
    assert (
        mind_db.scalar(
            f"SELECT COUNT(*) FROM knowledge_entities WHERE {scope_sql()}", target.as_tuple()
        )
        == 0
    )
    assert (
        mind_db.scalar(
            f"SELECT COUNT(*) FROM knowledge_relations WHERE {scope_sql()}", target.as_tuple()
        )
        == 0
    )
    assert mind_db.fts.search("Thursday", 10, scope=target) == []
    assert mind_db.frame_vec.search(alice_store.embedder.embed("Thursday"), 10, scope=target) == []

    # 6. Verify Bob's sibling scope is completely intact
    assert bob_store.get_frame(bob_frame.id).content == bob_text
    assert mind_db.fts.search("OMEGA", 10, scope=sibling) == [bob_frame.id]
    assert [
        rid
        for rid, _ in mind_db.frame_vec.search(
            bob_store.embedder.embed(bob_text), 10, scope=sibling
        )
    ] == [bob_frame.id]
    assert bob_kg.get_entity(bob_ent.id, scope=sibling).name == "OMEGA"
    assert bob_text in str(
        mind_db.scalar(f"SELECT payload FROM memory_jobs WHERE {scope_sql()}", sibling.as_tuple())
    )


def test_postgres_concurrent_job_queue_and_worker_generation_fencing(
    mind_db: MindDB,
    frame_store: FrameStore,
    kg: KnowledgeGraph,
    cfg: AppConfig,
    pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = mind_db.backend.postgres_schema  # type: ignore[attr-defined]
    scope = MemoryScope("egzakta", "worker-user", "personal")
    scoped_store = FrameStore(mind_db, frame_store.embedder, chunking=cfg.chunking, scope=scope)
    scoped_kg = KnowledgeGraph(mind_db, scope=scope)

    # 1. Concurrent enqueue (pg_advisory_xact_lock) and lease (FOR UPDATE SKIP LOCKED)
    handles = [
        MindDB(backend_type="postgres", dsn=pg_dsn, postgres_schema=schema, dim=384)
        for _ in range(2)
    ]
    barrier = Barrier(2)

    def concurrent_enqueue(idx: int) -> MemoryJob:
        q = JobQueue(handles[idx].backend, max_pending_per_scope=2)
        barrier.wait(timeout=5)
        return q.enqueue(
            scope,
            dedup_key="shared-dedup",
            payload={"event_id": "shared", "source_ref": "doc", "content": "Shared job payload."},
        )

    def concurrent_lease(idx: int) -> MemoryJob | None:
        q = JobQueue(handles[idx].backend, max_pending_per_scope=2)
        barrier.wait(timeout=5)
        return q.lease_next(f"pg-worker-{idx}")

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            enqueued = list(pool.map(concurrent_enqueue, range(2)))
        assert enqueued[0].id == enqueued[1].id

        with ThreadPoolExecutor(max_workers=2) as pool:
            leased = list(pool.map(concurrent_lease, range(2)))
        owners = [j for j in leased if j is not None]
        assert len(owners) == 1
        JobQueue(mind_db.backend).complete(
            owners[0].id, worker_id=owners[0].worker_id or "", attempt_count=owners[0].attempt_count
        )
    finally:
        for h in handles:
            h.close()

    # 2. Worker generation fencing (SELECT generation ... FOR UPDATE) on live Postgres
    seed = ingest_source_event(
        _evt("Initial fact before fencing.", "seed-1", scope=scope),
        store=scoped_store,
        cfg=cfg,
    ).frames[0]
    worker = MemoryWorker(mind_db, scoped_store, cfg=cfg, kg=scoped_kg, retry_backoff_seconds=0)
    queued_text = "Independent follow-up observation."
    job = worker.queue.enqueue(
        scope,
        dedup_key="fenced-1",
        payload={"event_id": "fenced-1", "source_ref": "doc", "content": queued_text},
    )
    assert job.scope_generation == 0

    first_attempt: list[MemoryJob] = []

    def bump_before_write(leased_job: MemoryJob) -> None:
        first_attempt.append(leased_job)
        forget_memory(scoped_store, seed.id, scope=scope)

    monkeypatch.setattr(worker, "_before_write", bump_before_write)
    requeued = worker.run_once()
    assert requeued is not None
    assert requeued.id == job.id
    assert requeued.status == "queued"
    assert requeued.scope_generation == get_scope_generation(mind_db.backend, scope) == 1
    assert requeued.max_attempts == job.max_attempts + 1

    monkeypatch.setattr(worker, "_before_write", lambda _: None)
    orig_assert = worker._assert_not_fenced

    def assert_locked_inside_tx(leased_job: MemoryJob) -> None:
        assert mind_db.in_transaction
        with pytest.raises(LeaseLostError):
            worker.queue.complete(
                leased_job.id,
                worker_id=worker.worker_id,
                attempt_count=first_attempt[0].attempt_count,
            )
        orig_assert(leased_job)

    monkeypatch.setattr(worker, "_assert_not_fenced", assert_locked_inside_tx)
    completed = worker.run_once()
    assert completed is not None and completed.status == "succeeded"
    assert completed.attempt_count == 2
    active = [f.content for f in scoped_store.all_frames(scope=scope) if f.status == "active"]
    assert active == [queued_text]
