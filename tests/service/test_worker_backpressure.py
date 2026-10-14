"""Queue backpressure, concurrent deduplication, and exclusive leasing contracts."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any

import pytest

from gluomem.service.jobs import BackpressureError, EventConflictError, JobQueue
from gluomem.storage.db import MindDB
from gluomem.storage.scope import DEFAULT_SCOPE, MemoryScope


def test_backpressure_and_payload_conflicts(mind_db: MindDB) -> None:
    queue = JobQueue(mind_db.backend, max_pending_per_scope=1)
    first = queue.enqueue(dedup_key="one", payload={"content": "one"})
    assert queue.enqueue(dedup_key="one", payload={"content": "one"}).id == first.id
    with pytest.raises(EventConflictError):
        queue.enqueue(dedup_key="one", payload={"content": "different"})
    with pytest.raises(BackpressureError):
        queue.enqueue(dedup_key="two")
    assert queue.enqueue(MemoryScope("other", "user"), dedup_key="one").status == "queued"


@pytest.mark.parametrize("operation", ["enqueue", "lease"])
def test_concurrent_queue_operation_is_atomic(
    mind_db: MindDB, operation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    handles = [MindDB(mind_db.path, dim=mind_db.dim) for _ in range(2)]
    barrier = Barrier(2)
    if operation == "lease":
        JobQueue(mind_db.backend).enqueue(dedup_key="only")
        reclaim = JobQueue.reclaim_expired_leases

        def interleave(queue: JobQueue, **kwargs: Any) -> int:
            count = reclaim(queue, **kwargs)
            barrier.wait(timeout=5)
            return count

        monkeypatch.setattr(JobQueue, "reclaim_expired_leases", interleave)

    def run(index: int):
        queue = JobQueue(handles[index].backend, max_pending_per_scope=1)
        barrier.wait(timeout=5)
        if operation == "enqueue":
            return queue.enqueue(dedup_key="same", payload={"content": "same"})
        return queue.lease_next(f"worker-{index}")

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(run, range(2)))
        if operation == "enqueue":
            assert results[0] is not None and results[1] is not None
            assert results[0].id == results[1].id
        else:
            assert sum(result is not None for result in results) == 1
        assert JobQueue(mind_db.backend).pending_count(DEFAULT_SCOPE) == 1
    finally:
        for handle in handles:
            handle.close()


def test_lease_uses_one_guarded_update(mind_db: MindDB, monkeypatch: pytest.MonkeyPatch) -> None:
    queue = JobQueue(mind_db.backend)
    queue.enqueue(dedup_key="only")
    statements: list[str] = []
    fetchone = mind_db.backend.fetchone

    def capture(sql: str, params: Any = ()):
        statements.append(sql)
        return fetchone(sql, params)

    monkeypatch.setattr(mind_db.backend, "fetchone", capture)
    assert queue.lease_next("worker") is not None
    assert len(statements) == 1
    assert "WHERE id = (SELECT id" in statements[0]
    assert "AND status = 'queued' RETURNING" in statements[0]


def test_bad_payload_dead_letters_without_empty_fallback(mind_db: MindDB) -> None:
    queue = JobQueue(mind_db.backend)
    job = queue.enqueue(dedup_key="bad")
    mind_db.execute("UPDATE memory_jobs SET payload = ? WHERE id = ?", ("not json", job.id))
    mind_db.commit()
    with pytest.raises(ValueError):
        queue.lease_next("worker")
    assert mind_db.scalar("SELECT status FROM memory_jobs WHERE id = ?", (job.id,)) == "dead_letter"
