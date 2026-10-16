"""A slow extraction must retain its lease without sharing a database connection."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Event

import pytest

from gluomem.config.app_config import AppConfig
from gluomem.frames.store import FrameStore
from gluomem.llm.fake import FakeExtractors
from gluomem.llm.outputs import ExtractedClaimSpec
from gluomem.service.jobs import JobQueue
from gluomem.service.worker import MemoryWorker
from gluomem.storage.db import MindDB
from gluomem.storage.postgres_backend import PostgresBackend


def test_slow_extraction_cannot_be_reclaimed(
    mind_db: MindDB, frame_store: FrameStore, cfg: AppConfig
) -> None:
    entered, release = Event(), Event()

    class SlowExtractors(FakeExtractors):
        def extract_event_claims(
            self, source_text: str, *, reference_date: datetime | None = None
        ) -> list[ExtractedClaimSpec]:
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test did not release extraction")
            return super().extract_event_claims(source_text, reference_date=reference_date)

    queue = JobQueue(mind_db.backend)
    job = queue.enqueue(
        dedup_key="slow-extraction",
        payload={
            "event_id": "slow-extraction",
            "source_ref": "slow-source",
            "content": "The release region is Belgrade.",
        },
    )

    def run():
        backend = mind_db.backend
        handle = (
            MindDB(
                dim=mind_db.dim,
                backend_type="postgres",
                dsn=backend.dsn,
                postgres_schema=backend.postgres_schema,
            )
            if isinstance(backend, PostgresBackend)
            else MindDB(mind_db.path, dim=mind_db.dim)
        )
        with handle as db:
            store = FrameStore(db, frame_store.embedder)
            worker = MemoryWorker(
                db,
                store,
                cfg=cfg,
                extractors=SlowExtractors(banner=False),
                worker_id="slow-worker",
                lease_seconds=0.4,
            )
            return worker.run_once()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run)
        try:
            assert entered.wait(5)
            Event().wait(0.9)
            assert queue.lease_next("contender") is None
        finally:
            release.set()
        completed = future.result(timeout=5)

    assert completed is not None
    assert completed.id == job.id
    assert completed.status == "succeeded"
    assert completed.attempt_count == 1
    assert len(frame_store.all_frames()) == 1


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_worker_rejects_invalid_lease_duration(
    mind_db: MindDB, frame_store: FrameStore, seconds: float
) -> None:
    with pytest.raises(ValueError, match="lease_seconds"):
        MemoryWorker(mind_db, frame_store, lease_seconds=seconds)
