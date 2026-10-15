"""Process durable jobs with transactional generation fencing and lease-attempt ownership."""

from __future__ import annotations

import logging
import os
import signal
import socket
import threading
from collections.abc import Callable
from typing import Any

from gluomem.config.app_config import AppConfig
from gluomem.frames.eligibility import is_recall_eligible
from gluomem.frames.expiry import sweep_expired_episodes
from gluomem.frames.generation import current_scope_generation
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.llm.outputs import CognifyResult, Extractors, KGEntity, KGRelation
from gluomem.service.jobs import (
    JobQueue,
    LeaseLostError,
    MemoryJob,
    StaleScopeGenerationError,
    write_transaction,
)
from gluomem.service.schemas import IngestEventRequest
from gluomem.storage.db import MindDB
from gluomem.storage.scope import scope_sql
from gluomem.timeutil import now_utc_iso
from gluomem.write.cognify import link_and_retire_resolved, resolve_cognify_result
from gluomem.write.events import ingest_source_event

_LOG = logging.getLogger(__name__)

JobHandler = Callable[[MemoryJob], None]

_PAYLOAD_JOB_TYPES = frozenset({"ingest_event", "cognify_frame", "compact_gop"})


class MemoryWorker:
    """Processes leased jobs from ``JobQueue`` with ``scope_generation`` fencing."""

    def __init__(
        self,
        db: MindDB,
        store: FrameStore,
        *,
        cfg: AppConfig | None = None,
        kg: KnowledgeGraph | None = None,
        queue: JobQueue | None = None,
        extractors: Extractors | None = None,
        worker_id: str | None = None,
        lease_seconds: float = 60.0,
        retry_backoff_seconds: float = 5.0,
        should_pause: Callable[[], bool] | None = None,
    ) -> None:
        self._db = db
        self._store = store
        self._cfg = cfg if cfg is not None else AppConfig()
        self._kg = kg if kg is not None else KnowledgeGraph(db, scope=store.scope)
        self._queue = queue if queue is not None else JobQueue(db.backend)
        self._extractors = extractors
        self._worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"
        self._lease_seconds = float(lease_seconds)
        self._retry_backoff_seconds = float(retry_backoff_seconds)
        self._should_pause = should_pause
        self._handlers: dict[str, JobHandler] = {
            "ingest_event": self._handle_ingest_event,
            "cognify_frame": self._handle_cognify_frame,
            "compact_gop": self._handle_compact_gop,
            "sweep_expired": self._handle_sweep_expired,
        }
        self.stop_event = threading.Event()

    @property
    def queue(self) -> JobQueue:
        return self._queue

    @property
    def worker_id(self) -> str:
        return self._worker_id

    def register_handler(self, job_type: str, handler: JobHandler) -> None:
        """Register or override a handler for ``job_type``."""
        self._handlers[job_type] = handler

    def _before_write(self, job: MemoryJob) -> None:
        """Run immediately before opening the fenced write transaction."""

    def _assert_not_fenced(self, job: MemoryJob) -> None:
        backend = self._db.backend
        if backend.dialect == "postgres":
            backend.execute(
                "INSERT INTO scope_generations (tenant_id, principal_id, namespace, "
                "generation, updated_at) VALUES (?, ?, ?, 0, ?) "
                "ON CONFLICT (tenant_id, principal_id, namespace) DO NOTHING",
                (*job.scope.as_tuple(), now_utc_iso()),
            )
            backend.fetchone(
                f"SELECT generation FROM scope_generations WHERE {scope_sql()} FOR UPDATE",
                job.scope.as_tuple(),
            )
        current = current_scope_generation(backend, job.scope)
        if current != job.scope_generation:
            raise StaleScopeGenerationError(job.id, job.scope_generation, current)

    def _renew(self, job: MemoryJob) -> None:
        self._queue.renew_lease(
            job.id,
            self._worker_id,
            attempt_count=job.attempt_count,
            lease_seconds=self._lease_seconds,
        )

    def _handle_ingest_event(self, job: MemoryJob) -> None:
        event = IngestEventRequest.model_validate({**job.payload}).to_source_event(job.scope)
        ingest_source_event(
            event,
            store=self._store,
            kg=self._kg,
            cfg=self._cfg,
            extractors=self._extractors,
        )

    def _handle_cognify_frame(self, job: MemoryJob) -> None:
        payload = job.payload
        frame_id = int(payload["frame_id"])
        entities: list[dict[str, Any]] = list(payload.get("entities") or [])
        relations: list[dict[str, Any]] = list(payload.get("relations") or [])

        frame = self._store.get_frame(frame_id, scope=job.scope)
        if not is_recall_eligible(frame):
            return

        self._renew(job)
        if entities or relations:
            raw = CognifyResult(
                entities=[
                    KGEntity(
                        name=str(ent.get("name", "")).strip(),
                        entity_type=str(ent.get("entity_type", "concept")).strip() or "concept",
                        properties=ent.get("properties")
                        if isinstance(ent.get("properties"), dict)
                        else {},
                    )
                    for ent in entities
                    if str(ent.get("name", "")).strip()
                ],
                relations=[
                    KGRelation(
                        source=str(rel.get("source", "")).strip(),
                        target=str(rel.get("target", "")).strip(),
                        relation_type=str(rel.get("relation_type", "related_to")).strip()
                        or "related_to",
                        confidence=float(rel.get("confidence", 1.0)),
                    )
                    for rel in relations
                    if str(rel.get("source", "")).strip() and str(rel.get("target", "")).strip()
                ],
            )
        elif self._extractors is not None and self._cfg.write.cognify.enabled:
            raw = self._extractors.cognify(frame.content)
        else:
            return

        resolved = resolve_cognify_result(
            raw,
            kg=self._kg,
            cfg=self._cfg,
            valid_from=frame.event_time,
            scope=job.scope,
        )
        link_and_retire_resolved(
            resolved,
            frame.id,
            kg=self._kg,
            cfg=self._cfg,
            scope=job.scope,
        )

    def _handle_compact_gop(self, job: MemoryJob) -> None:
        gop_id = str(job.payload["gop_id"])
        self._store.compact(gop_id, self._cfg, scope=job.scope)

    def _handle_sweep_expired(self, job: MemoryJob) -> None:
        now_iso = job.payload.get("now_iso")
        sweep_expired_episodes(
            self._store,
            scope=job.scope,
            now_iso=str(now_iso) if now_iso is not None else None,
        )

    def run_once(self, *, now_iso: str | None = None) -> MemoryJob | None:
        """Execute one leased job; stale writes roll back and retry at the new generation."""
        if self._should_pause is not None and self._should_pause():
            return None
        try:
            job = self._queue.lease_next(
                self._worker_id, lease_seconds=self._lease_seconds, now_iso=now_iso
            )
        except ValueError:
            _LOG.exception("Invalid job payload dead-lettered")
            return None
        if job is None:
            return None
        try:
            try:
                self._before_write(job)
                with write_transaction(self._db.backend):
                    self._assert_not_fenced(job)
                    self._queue.renew_lease(
                        job.id,
                        self._worker_id,
                        attempt_count=job.attempt_count,
                        lease_seconds=self._lease_seconds,
                        now_iso=now_iso,
                    )
                    handler = self._handlers.get(job.job_type)
                    if handler is None:
                        raise ValueError(f"unknown_job_type:{job.job_type}")
                    # Forget and retract redact queued payloads to {}; nothing is left to write.
                    if job.payload or job.job_type not in _PAYLOAD_JOB_TYPES:
                        handler(job)
                    return self._queue.complete(
                        job.id,
                        worker_id=self._worker_id,
                        attempt_count=job.attempt_count,
                        now_iso=now_iso,
                    )
            except StaleScopeGenerationError as exc:
                return self._queue.requeue_stale(job, exc.current_generation, now_iso=now_iso)
            except LeaseLostError:
                raise
            except Exception as exc:
                return self._queue.fail(
                    job.id,
                    f"{type(exc).__name__}: {exc}",
                    worker_id=self._worker_id,
                    attempt_count=job.attempt_count,
                    retry_backoff_seconds=self._retry_backoff_seconds,
                    now_iso=now_iso,
                )
        except LeaseLostError:
            _LOG.warning("Lost lease for job %s attempt %s", job.id, job.attempt_count)
            return None

    def run_until_empty(
        self, *, max_jobs: int = 100, now_iso: str | None = None
    ) -> list[MemoryJob]:
        """Run jobs sequentially until no eligible queued jobs remain or ``max_jobs`` is reached."""
        processed: list[MemoryJob] = []
        for _ in range(max_jobs):
            res = self.run_once(now_iso=now_iso)
            if res is None:
                break
            processed.append(res)
        return processed

    def run_forever(self, *, poll_interval: float = 1.0, max_iterations: int | None = None) -> int:
        """Poll until stopped, logging failures and backing off between idle/error polls."""
        previous: dict[int, Any] = {}
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous[signum] = signal.getsignal(signum)
                signal.signal(signum, lambda *_: self.stop_event.set())
        count, iterations = 0, 0
        try:
            while not self.stop_event.is_set():
                if max_iterations is not None and iterations >= max_iterations:
                    break
                iterations += 1
                try:
                    result = self.run_once()
                    if result is not None:
                        count += 1
                        continue
                except Exception:
                    _LOG.exception("Worker poll failed")
                self.stop_event.wait(max(0.01, poll_interval))
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        return count
