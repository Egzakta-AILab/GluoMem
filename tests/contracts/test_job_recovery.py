"""Tests for durable memory_jobs crash recovery, dead-lettering, and scope_generation fencing."""

from __future__ import annotations

from pathlib import Path

from gluomem.config.app_config import AppConfig
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.lifecycle import forget_memory, retract_source
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.search.hybrid import HybridSearch
from gluomem.service.jobs import JobQueue
from gluomem.service.worker import MemoryWorker
from gluomem.storage.db import MindDB
from gluomem.storage.scope import MemoryScope
from gluomem.write.events import SourceEvent, ingest_source_event


def test_crash_mid_lease_recovery_and_dead_letter(tmp_path: Path) -> None:
    cfg = AppConfig()
    db = MindDB(tmp_path / "jobs.mind", vector_dim=32)
    embedder = MockEmbedder(dim=32)
    scope = MemoryScope("egzakta", "mikhail", "personal")
    store = FrameStore(db, embedder=embedder, scope=scope)
    kg = KnowledgeGraph(db, scope=scope)
    search = HybridSearch(db, store, cfg, embedder=embedder, kg=kg, scope=scope)
    queue = JobQueue(db.backend, default_max_attempts=2)

    job = queue.enqueue(
        scope=scope,
        job_type="ingest_event",
        dedup_key="evt-job-1:1",
        payload={
            "event_id": "evt-job-1",
            "revision": 1,
            "source_type": "chat_turn",
            "source_ref": "conv-job-1",
            "content": "We decided to deploy Redis Sentinel for cache failover.",
        },
        now_iso="2026-03-01T10:00:00",
    )
    assert job.status == "queued"
    assert job.attempt_count == 0

    # Worker 1 leases the job at 10:00:05 (60s lease -> expires at 10:01:05) and crashes
    leased = queue.lease_next(
        "worker-crash", lease_seconds=60.0, now_iso="2026-03-01T10:00:05"
    )
    assert leased is not None
    assert leased.id == job.id
    assert leased.status == "leased"
    assert leased.attempt_count == 1

    # Before lease expiry (10:00:30), Worker 2 sees no available job
    worker2 = MemoryWorker(
        db, store, cfg=cfg, kg=kg, queue=queue, worker_id="worker-2"
    )
    assert worker2.run_once(now_iso="2026-03-01T10:00:30") is None

    # After lease expiry (10:01:10), Worker 2 reclaims and completes the job
    completed = worker2.run_once(now_iso="2026-03-01T10:01:10")
    assert completed is not None
    assert completed.id == job.id
    assert completed.status == "succeeded"
    assert completed.attempt_count == 2

    # Exactly 1 memory frame was written and indexed
    hits = search.search("Redis Sentinel cache failover", scope=scope).hits
    assert len(hits) == 1

    # A failing job transitions to dead_letter after max_attempts (2)
    queue.enqueue(
        scope=scope,
        job_type="unknown_type",
        dedup_key="bad-1",
        payload={},
        max_attempts=2,
        now_iso="2026-03-01T10:02:00",
    )
    attempt1 = worker2.run_once(now_iso="2026-03-01T10:02:01")
    assert attempt1 is not None and attempt1.status == "queued"
    assert attempt1.attempt_count == 1
    attempt2 = worker2.run_once(now_iso="2026-03-01T10:03:00")
    assert attempt2 is not None and attempt2.status == "dead_letter"
    assert attempt2.attempt_count == 2

    stats = queue.queue_stats(scope=scope)
    assert stats["counts"]["succeeded"] == 1
    assert stats["counts"]["dead_letter"] == 1

    db.close()


def test_in_flight_worker_fenced_when_forget_or_retract_intervenes(
    tmp_path: Path,
) -> None:
    cfg = AppConfig()
    db = MindDB(tmp_path / "fenced.mind", vector_dim=32)
    embedder = MockEmbedder(dim=32)
    scope = MemoryScope("egzakta", "mikhail", "personal")
    store = FrameStore(db, embedder=embedder, scope=scope)
    kg = KnowledgeGraph(db, scope=scope)
    search = HybridSearch(db, store, cfg, embedder=embedder, kg=kg, scope=scope)
    queue = JobQueue(db.backend)

    # 1. Ingest a frame synchronously, then enqueue a background cognify_frame job at gen=0
    evt = SourceEvent(
        event_id="evt-fence-1",
        revision=1,
        source_type="document",
        source_ref="doc-confidential",
        scope=scope,
        content="Project Orion launches in Belgrade.",
    )
    frame = ingest_source_event(evt, store=store).frames[0]

    cog_job = queue.enqueue(
        scope=scope,
        job_type="cognify_frame",
        dedup_key=f"cog:{frame.id}",
        payload={
            "frame_id": frame.id,
            "entities": [
                {"name": "Orion", "entity_type": "project"},
                {"name": "Belgrade", "entity_type": "city"},
            ],
            "relations": [
                {
                    "source": "Orion",
                    "target": "Belgrade",
                    "relation_type": "launches_in",
                }
            ],
        },
    )
    assert cog_job.scope_generation == 0

    worker = MemoryWorker(
        db, store, cfg=cfg, kg=kg, queue=queue, worker_id="worker-fence"
    )

    # Simulate forget_memory happening while the worker has leased the job and finished LLM compute,
    # right before it commits its KG writes.
    def _intervene_forget(_job: object) -> None:
        forget_memory(store, frame.id, scope=scope)

    worker.set_before_commit_hook(_intervene_forget)
    result_job = worker.run_once()
    assert result_job is not None
    assert result_job.id == cog_job.id
    assert result_job.status == "aborted_stale"
    assert "stale_scope_generation:0<1" in (result_job.last_error or "")

    # Verify zero KG entities/relations or resurrected frames exist after the fenced job
    assert kg.find_valid_entities("Orion", cfg.kg, scope=scope) == []
    assert kg.find_valid_entities("Belgrade", cfg.kg, scope=scope) == []
    assert search.search("Orion Belgrade", scope=scope).hits == []

    # 2. Also test an in-flight ingest_event job fenced by retract_source
    ingest_job = queue.enqueue(
        scope=scope,
        job_type="ingest_event",
        dedup_key="evt-fence-2:1",
        payload={
            "event_id": "evt-fence-2",
            "revision": 1,
            "source_type": "document",
            "source_ref": "doc-confidential",
            "content": "Project Helios is our backup satellite.",
        },
    )
    assert ingest_job.scope_generation == 1

    def _intervene_retract(_job: object) -> None:
        retract_source(store, "doc-confidential", scope=scope)

    worker.set_before_commit_hook(_intervene_retract)
    res2 = worker.run_once()
    assert res2 is not None
    assert res2.status == "aborted_stale"
    assert search.search("Helios satellite", scope=scope).hits == []

    db.close()
