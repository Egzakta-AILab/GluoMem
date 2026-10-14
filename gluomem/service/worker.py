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
from gluomem.frames.generation import current_scope_generation
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
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
        worker_id: str | None = None,
        lease_seconds: float = 60.0,
        retry_backoff_seconds: float = 5.0,
    ) -> None:
        self._db = db
        self._store = store
        self._cfg = cfg if cfg is not None else AppConfig()
        self._kg = kg if kg is not None else KnowledgeGraph(db, scope=store.scope)
        self._queue = queue if queue is not None else JobQueue(db.backend)
        self._worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"
        self._lease_seconds = float(lease_seconds)
        self._retry_backoff_seconds = float(retry_backoff_seconds)
        self._handlers: dict[str, JobHandler] = {
            "ingest_event": self._handle_ingest_event,
            "cognify_frame": self._handle_cognify_frame,
            "compact_gop": self._handle_compact_gop,
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
        ingest_source_event(event, store=self._store, kg=self._kg, cfg=self._cfg)

    def _handle_cognify_frame(self, job: MemoryJob) -> None:
        payload = job.payload
        frame_id = int(payload["frame_id"])
        entities: list[dict[str, Any]] = list(payload.get("entities") or [])
        relations: list[dict[str, Any]] = list(payload.get("relations") or [])

        frame = self._store.get_frame(frame_id, scope=job.scope)
        if not is_recall_eligible(frame):
            return

        entity_map: dict[str, int] = {}
        for ent in entities:
            self._renew(job)
            name = str(ent.get("name", "")).strip()
            etype = str(ent.get("entity_type", "concept")).strip() or "concept"
            if not name:
                continue
            existing = self._kg.find_valid_entities(
                name, self._cfg.kg, entity_type=etype, scope=job.scope
            )
            if existing:
                e_obj = existing[0]
            else:
                e_obj = self._kg.create_entity(
                    etype,
                    name,
                    ent.get("properties") if isinstance(ent.get("properties"), dict) else None,
                    scope=job.scope,
                )
            self._kg.link_entity_frame(e_obj.id, frame.id, scope=job.scope)
            entity_map[name.lower()] = e_obj.id

        for rel in relations:
            self._renew(job)
            src_name = str(rel.get("source", "")).strip().lower()
            tgt_name = str(rel.get("target", "")).strip().lower()
            rel_type = str(rel.get("relation_type", "related_to")).strip()
            if src_name in entity_map and tgt_name in entity_map and rel_type:
                relation = self._kg.create_relation(
                    entity_map[src_name],
                    entity_map[tgt_name],
                    rel_type,
                    confidence=float(rel.get("confidence", 1.0)),
                    scope=job.scope,
                )
                self._kg.link_relation_frame(relation.id, frame.id, scope=job.scope)

    def _handle_compact_gop(self, job: MemoryJob) -> None:
        gop_id = str(job.payload["gop_id"])
        self._store.compact(gop_id, self._cfg, scope=job.scope)

    def run_once(self, *, now_iso: str | None = None) -> MemoryJob | None:
        """Execute one leased job; stale writes roll back and retry at the new generation."""
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
