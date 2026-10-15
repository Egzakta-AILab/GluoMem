"""Worker crash recovery, attempt ownership, and generation retry contracts."""

import signal
from typing import Any

import pytest

from gluomem.config.app_config import AppConfig
from gluomem.frames.lifecycle import forget_memory, get_scope_generation, retract_source
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.service.jobs import LeaseLostError, MemoryJob
from gluomem.service.worker import MemoryWorker
from gluomem.storage.db import MindDB
from gluomem.write import gate
from gluomem.write.events import SourceEvent, ingest_source_event


@pytest.fixture
def worker(mind_db: MindDB, frame_store: FrameStore, cfg: AppConfig, kg: KnowledgeGraph):
    return MemoryWorker(mind_db, frame_store, cfg=cfg, kg=kg, retry_backoff_seconds=0)


def test_crash_mid_lease_recovery_and_dead_letter(
    worker: MemoryWorker, frame_store: FrameStore
) -> None:
    queue = worker.queue
    job = queue.enqueue(
        dedup_key="recover",
        payload={
            "event_id": "recover",
            "source_ref": "doc",
            "content": "Redis Sentinel provides cache failover.",
        },
        max_attempts=2,
        now_iso="2026-03-01T10:00:00",
    )
    leased = queue.lease_next("crashed", now_iso="2026-03-01T10:00:05")
    assert leased is not None and leased.attempt_count == 1
    assert worker.run_once(now_iso="2026-03-01T10:00:30") is None
    completed = worker.run_once(now_iso="2026-03-01T10:01:10")
    assert completed is not None and completed.id == job.id and completed.status == "succeeded"
    assert completed.attempt_count == 2
    assert len(frame_store.all_frames()) == 1
    queue.enqueue(
        job_type="unknown", dedup_key="bad", max_attempts=2, now_iso="2026-03-01T10:02:00"
    )
    first = worker.run_once(now_iso="2026-03-01T10:02:01")
    second = worker.run_once(now_iso="2026-03-01T10:03:00")
    assert first is not None and first.status == "queued"
    assert second is not None and second.status == "dead_letter"


@pytest.mark.parametrize(("related", "retract"), [(True, False), (False, False), (True, True)])
def test_forget_after_lease_requeues_at_new_generation(
    worker: MemoryWorker,
    frame_store: FrameStore,
    monkeypatch: pytest.MonkeyPatch,
    related: bool,
    retract: bool,
) -> None:
    content = "Project Orion launches in Belgrade."
    original = ingest_source_event(
        SourceEvent(
            event_id="original", source_ref="doc", content=content, scope=frame_store.scope
        ),
        store=frame_store,
    ).frames[0]
    queued_content = content if related else "Project Helios is our backup satellite."
    job = worker.queue.enqueue(
        dedup_key="pending",
        payload={"event_id": "pending", "source_ref": "doc", "content": queued_content},
    )
    leases: list[MemoryJob] = []

    def intervene(leased: MemoryJob) -> None:
        leases.append(leased)
        (retract_source(frame_store, "doc") if retract else forget_memory(frame_store, original.id))

    monkeypatch.setattr(worker, "_before_write", intervene)
    result = worker.run_once()
    assert result is not None and result.id == job.id and result.status == "queued"
    assert result.scope_generation == get_scope_generation(frame_store.db.backend)
    assert result.scope_generation >= 1 and result.attempt_count == 1
    assert result.max_attempts - result.attempt_count == job.max_attempts
    assert frame_store.get_frame(original.id).status == "forgotten"
    monkeypatch.setattr(worker, "_before_write", lambda _: None)
    check = worker._assert_not_fenced

    def check_inside_transaction(job):
        assert frame_store.db.in_transaction
        with pytest.raises(LeaseLostError):
            worker.queue.complete(
                job.id, worker_id=worker.worker_id, attempt_count=leases[0].attempt_count
            )
        check(job)

    monkeypatch.setattr(worker, "_assert_not_fenced", check_inside_transaction)
    done = worker.run_once()
    assert done is not None and done.status == "succeeded"
    assert done.attempt_count == 2
    active = [frame for frame in frame_store.all_frames() if frame.status == "active"]
    assert [frame.content for frame in active] == ([] if related else [queued_content])


def test_forgotten_frame_cannot_be_cognified(
    worker: MemoryWorker,
    frame_store: FrameStore,
    kg: KnowledgeGraph,
    cfg: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = ingest_source_event(
        SourceEvent(event_id="original", source_ref="doc", content="Orion launches in Belgrade."),
        store=frame_store,
    ).frames[0]
    worker.queue.enqueue(
        job_type="cognify_frame",
        dedup_key="cognify",
        payload={"frame_id": frame.id, "entities": [{"name": "Orion"}]},
    )
    monkeypatch.setattr(worker, "_before_write", lambda _: forget_memory(frame_store, frame.id))
    stale = worker.run_once()
    assert stale is not None and stale.status == "queued"
    monkeypatch.setattr(worker, "_before_write", lambda _: None)
    done = worker.run_once()
    assert done is not None and done.status == "succeeded"
    assert kg.find_valid_entities("Orion", cfg.kg) == []


def test_mid_ingest_crash_retries_all_frames(
    worker: MemoryWorker, frame_store: FrameStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = worker.queue.enqueue(
        dedup_key="crash",
        payload={
            "event_id": "crash",
            "source_ref": "doc",
            "messages": [
                {"message_id": "one", "content": "Our primary database is PostgreSQL."},
                {"message_id": "two", "content": "Our cache is Redis."},
            ],
        },
    )
    admit = gate.admit
    crashed = False

    def crash_once(*args: Any, **kwargs: Any):
        nonlocal crashed
        result = admit(*args, **kwargs)
        if not crashed:
            crashed = True
            raise RuntimeError("simulated mid-ingest crash")
        return result

    monkeypatch.setattr(gate, "admit", crash_once)
    failed = worker.run_once()
    assert failed is not None and failed.status == "queued"
    assert frame_store.all_frames() == []
    done = worker.run_once()
    assert done is not None and done.id == job.id and done.status == "succeeded"
    assert len(frame_store.all_frames()) == 2
    assert all(frame_store.index_state(frame.id).ready for frame in frame_store.all_frames())


def test_stale_attempt_cannot_complete_and_worker_continues(
    worker: MemoryWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = worker.queue
    job = queue.enqueue(job_type="noop", dedup_key="lease", now_iso="2026-03-01T10:00:00")
    stale = queue.lease_next(worker.worker_id, lease_seconds=1, now_iso="2026-03-01T10:00:00")
    assert stale is not None
    queue.reclaim_expired_leases(now_iso="2026-03-01T10:00:02")
    current = queue.lease_next(worker.worker_id, now_iso="2026-03-01T10:00:02")
    assert current is not None
    with pytest.raises(LeaseLostError):
        queue.complete(job.id, worker_id=worker.worker_id, attempt_count=stale.attempt_count)
    with pytest.raises(LeaseLostError):
        queue.fail(
            job.id, "late failure", worker_id=worker.worker_id, attempt_count=stale.attempt_count
        )
    lease_next = queue.lease_next
    monkeypatch.setattr(queue, "lease_next", lambda *a, **kw: stale)
    assert worker.run_once() is None
    monkeypatch.setattr(queue, "lease_next", lease_next)
    queue.complete(current.id, worker_id=worker.worker_id, attempt_count=current.attempt_count)
    queue.enqueue(job_type="noop", dedup_key="next")
    worker.register_handler("noop", lambda _: None)
    done = worker.run_once()
    assert done is not None and done.status == "succeeded"


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_worker_loop_logs_failures_and_obeys_stop(
    worker: MemoryWorker, monkeypatch: pytest.MonkeyPatch, signum: int
) -> None:
    calls = 0

    def poll():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("database temporarily unavailable")
        signal.raise_signal(signum)
        return None

    monkeypatch.setattr(worker, "run_once", poll)
    assert worker.run_forever(poll_interval=0.01) == 0
    assert calls == 2


def test_worker_should_pause_and_sweep_expired_job(
    mind_db: MindDB, frame_store: FrameStore, cfg: AppConfig, kg: KnowledgeGraph
) -> None:
    cfg = cfg.with_overrides({"lifecycle.episode_ttl_days": 7})
    store = FrameStore(mind_db, frame_store.embedder, lifecycle=cfg.lifecycle)
    expired = ingest_source_event(
        SourceEvent(
            event_id="exp-job",
            source_ref="doc",
            content="Yesterday the worker queue saturated.",
            kind="episode",
            created_at="2026-01-01T00:00:00",
            scope=store.scope,
        ),
        store=store,
        cfg=cfg,
    ).frames[0]
    assert expired.expires_at == "2026-01-08T00:00:00"

    paused = True
    worker = MemoryWorker(
        mind_db,
        store,
        cfg=cfg,
        kg=kg,
        retry_backoff_seconds=0,
        should_pause=lambda: paused,
    )
    job = worker.queue.enqueue(
        job_type="sweep_expired",
        dedup_key="sweep:2026-01-10",
        payload={"now_iso": "2026-01-10T00:00:00"},
        now_iso="2026-01-10T00:00:00",
    )
    assert worker.run_once(now_iso="2026-01-10T00:00:00") is None
    queued = worker.queue.get_job(job.id)
    assert queued.status == "queued" and queued.attempt_count == 0
    assert store.get_frame(expired.id).status == "active"

    paused = False
    done = worker.run_once(now_iso="2026-01-10T00:00:00")
    assert done is not None and done.id == job.id and done.status == "succeeded"
    assert store.get_frame(expired.id).status == "expired"
    assert store.index_state(expired.id).frame_vec == 0


def test_worker_and_ingest_cognify_unifies_kg_resolution(
    mind_db: MindDB, frame_store: FrameStore, cfg: AppConfig, kg: KnowledgeGraph
) -> None:
    from gluomem.llm.fake import FakeExtractors
    from gluomem.llm.outputs import CognifyResult, KGEntity, KGRelation

    extractors = FakeExtractors(
        cognify_results=[
            CognifyResult(
                entities=[
                    KGEntity(name="Alice", entity_type="person"),
                    KGEntity(name="Belgrade", entity_type="city"),
                ],
                relations=[KGRelation(source="Alice", target="Belgrade", relation_type="lives_in")],
            )
        ],
        banner=False,
    )
    ingested = ingest_source_event(
        SourceEvent(
            event_id="evt-cog-1",
            source_ref="doc",
            content="Alice works at Acme in Belgrade.",
            event_time="2026-01-01T00:00:00",
            scope=frame_store.scope,
        ),
        store=frame_store,
        kg=kg,
        cfg=cfg,
        extractors=extractors,
    )
    assert len(ingested.frames) == 1
    assert kg.find_valid_entities("Alice", cfg.kg)

    # Worker cognify_frame resolves explicit entities/relations and retires conflicting edges
    f2 = ingest_source_event(
        SourceEvent(
            event_id="evt-cog-2",
            source_ref="doc",
            content="Alice moved to Novi Sad.",
            event_time="2026-02-01T00:00:00",
            scope=frame_store.scope,
        ),
        store=frame_store,
        cfg=cfg,
    ).frames[0]
    worker = MemoryWorker(
        mind_db, frame_store, cfg=cfg, kg=kg, extractors=extractors, retry_backoff_seconds=0
    )
    worker.queue.enqueue(
        job_type="cognify_frame",
        dedup_key="cog-f2",
        payload={
            "frame_id": f2.id,
            "entities": [{"name": "Alice", "entity_type": "person"}, {"name": "Novi Sad"}],
            "relations": [{"source": "Alice", "target": "Novi Sad", "relation_type": "lives_in"}],
        },
    )
    assert worker.run_once() is not None
    alice = kg.find_valid_entities("Alice", cfg.kg)[0]
    valid_rels = kg.find_valid_relations(alice.id, "lives_in")
    assert len(valid_rels) == 1
    novi_sad = kg.find_valid_entities("Novi Sad", cfg.kg)[0]
    assert valid_rels[0].target_id == novi_sad.id
