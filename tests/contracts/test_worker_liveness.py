"""Operational fleet pulses survive idle polls and blocked external work."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Event

from gluomem.frames.store import FrameStore
from gluomem.llm.fake import FakeExtractors
from gluomem.service.jobs import JobQueue
from gluomem.service.readiness import worker_is_live
from gluomem.service.worker import MemoryWorker
from gluomem.storage.db import MindDB
from gluomem.storage.postgres_backend import PostgresBackend


def test_worker_pulses_during_blocked_extraction(mind_db, frame_store, cfg):
    entered, release = Event(), Event()

    class SlowExtractors(FakeExtractors):
        def extract_event_claims(self, source_text: str, *, reference_date: datetime | None = None):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test extraction not released")
            return super().extract_event_claims(source_text, reference_date=reference_date)

    JobQueue(mind_db.backend).enqueue(
        dedup_key="liveness",
        payload={"event_id": "liveness", "source_ref": "test:slow", "content": "Synthetic fact"},
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
        with handle:
            worker = MemoryWorker(
                handle,
                FrameStore(handle, frame_store.embedder),
                cfg=cfg,
                extractors=SlowExtractors(banner=False),
                lease_seconds=0.3,
            )
            return worker.run_once()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run)
        try:
            assert entered.wait(5)
            Event().wait(0.7)
            assert worker_is_live(mind_db.backend, stale_after_seconds=0.4)
        finally:
            release.set()
        completed = future.result(timeout=5)
        assert completed is not None
        assert completed.status == "succeeded"
