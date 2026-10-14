"""Durable background worker (``gluomem-worker``) with leases, retries, and fencing."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from gluomem.config.app_config import AppConfig
from gluomem.frames.eligibility import is_recall_eligible
from gluomem.frames.lifecycle import get_scope_generation
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.service.jobs import (
    JobQueue,
    MemoryJob,
    StaleScopeGenerationError,
)
from gluomem.storage.db import MindDB
from gluomem.write.events import (
    SourceEvent,
    SourceEventMessage,
    ingest_source_event,
)

JobHandler = Callable[[MemoryJob], None]


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
        worker_id: str = "gluomem-worker-1",
        lease_seconds: float = 60.0,
        retry_backoff_seconds: float = 5.0,
    ) -> None:
        self._db = db
        self._store = store
        self._cfg = cfg if cfg is not None else AppConfig()
        self._kg = kg if kg is not None else KnowledgeGraph(db, scope=store.scope)
        self._queue = queue if queue is not None else JobQueue(db.backend)
        self._worker_id = worker_id
        self._lease_seconds = float(lease_seconds)
        self._retry_backoff_seconds = float(retry_backoff_seconds)
        self._handlers: dict[str, JobHandler] = {
            "ingest_event": self._handle_ingest_event,
            "cognify_frame": self._handle_cognify_frame,
            "compact_gop": self._handle_compact_gop,
        }
        self._before_commit_hook: Callable[[MemoryJob], None] | None = None

    @property
    def queue(self) -> JobQueue:
        return self._queue

    @property
    def worker_id(self) -> str:
        return self._worker_id

    def register_handler(self, job_type: str, handler: JobHandler) -> None:
        """Register or override a handler for ``job_type``."""
        self._handlers[job_type] = handler

    def set_before_commit_hook(
        self, hook: Callable[[MemoryJob], None] | None
    ) -> None:
        """Install an optional hook invoked after compute and immediately before DB write/commit."""
        self._before_commit_hook = hook

    def _assert_not_fenced(self, job: MemoryJob) -> None:
        current_gen = get_scope_generation(self._db.backend, job.scope)
        if current_gen != job.scope_generation:
            raise StaleScopeGenerationError(
                job.id, job.scope_generation, current_gen
            )

    def _handle_ingest_event(self, job: MemoryJob) -> None:
        payload = job.payload
        raw_msgs = payload.get("messages") or []
        messages = [
            SourceEventMessage(
                message_id=str(m.get("message_id", f"m-{idx}")),
                role=str(m.get("role", "user")),
                content=str(m.get("content", "")),
                timestamp=m.get("timestamp") or m.get("event_time"),
            )
            for idx, m in enumerate(raw_msgs)
            if isinstance(m, dict)
        ]
        evt = SourceEvent(
            event_id=str(payload.get("event_id") or job.dedup_key),
            revision=int(payload.get("revision", 1)),
            source_type=payload.get("source_type", "chat_turn"),
            source_ref=str(payload.get("source_ref", job.dedup_key)),
            scope=job.scope,
            content=payload.get("content"),
            messages=messages,
            kind=payload.get("kind"),
            confidence=float(payload.get("confidence", 1.0)),
            require_review=bool(payload.get("require_review", False)),
            pinned=bool(payload.get("pinned", False)),
            event_time=payload.get("event_time"),
            created_at=payload.get("created_at"),
            force_reassert=bool(payload.get("force_reassert", False)),
        )
        if self._before_commit_hook is not None:
            self._before_commit_hook(job)
        self._assert_not_fenced(job)
        ingest_source_event(
            evt,
            store=self._store,
            kg=self._kg,
            cfg=self._cfg,
        )

    def _handle_cognify_frame(self, job: MemoryJob) -> None:
        payload = job.payload
        frame_id = int(payload["frame_id"])
        entities: list[dict[str, Any]] = list(payload.get("entities") or [])
        relations: list[dict[str, Any]] = list(payload.get("relations") or [])

        if self._before_commit_hook is not None:
            self._before_commit_hook(job)
        self._assert_not_fenced(job)

        frame = self._store.get_frame(frame_id, scope=job.scope)
        if not is_recall_eligible(frame) and frame.status != "candidate":
            return

        entity_map: dict[str, int] = {}
        for ent in entities:
            self._assert_not_fenced(job)
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
            self._assert_not_fenced(job)
            src_name = str(rel.get("source", "")).strip().lower()
            tgt_name = str(rel.get("target", "")).strip().lower()
            rel_type = str(rel.get("relation_type", "related_to")).strip()
            if src_name in entity_map and tgt_name in entity_map and rel_type:
                self._kg.create_relation(
                    entity_map[src_name],
                    entity_map[tgt_name],
                    rel_type,
                    confidence=float(rel.get("confidence", 1.0)),
                    scope=job.scope,
                )

    def _handle_compact_gop(self, job: MemoryJob) -> None:
        gop_id = str(job.payload["gop_id"])
        if self._before_commit_hook is not None:
            self._before_commit_hook(job)
        self._assert_not_fenced(job)
        self._store.compact(gop_id, self._cfg, scope=job.scope)

    def run_once(self, *, now_iso: str | None = None) -> MemoryJob | None:
        """Lease and execute at most one job, returning the resulting ``MemoryJob`` state."""
        job = self._queue.lease_next(
            self._worker_id,
            lease_seconds=self._lease_seconds,
            now_iso=now_iso,
        )
        if job is None:
            return None

        handler = self._handlers.get(job.job_type)
        if handler is None:
            return self._queue.fail(
                job.id,
                f"unknown_job_type:{job.job_type}",
                worker_id=self._worker_id,
                retry_backoff_seconds=self._retry_backoff_seconds,
                now_iso=now_iso,
            )

        try:
            self._assert_not_fenced(job)
            handler(job)
            return self._queue.complete(
                job.id, worker_id=self._worker_id, now_iso=now_iso
            )
        except StaleScopeGenerationError as exc:
            return self._queue.mark_aborted_stale(
                job.id,
                current_generation=exc.current_generation,
                now_iso=now_iso,
            )
        except Exception as exc:
            return self._queue.fail(
                job.id,
                f"{type(exc).__name__}: {exc}",
                worker_id=self._worker_id,
                retry_backoff_seconds=self._retry_backoff_seconds,
                now_iso=now_iso,
            )

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

    def run_forever(
        self,
        *,
        poll_interval: float = 1.0,
        max_iterations: int | None = None,
    ) -> int:
        """Poll and process jobs in a loop until ``max_iterations`` (if set) is reached."""
        processed_count = 0
        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            iterations += 1
            res = self.run_once()
            if res is not None:
                processed_count += 1
            else:
                time.sleep(max(0.01, float(poll_interval)))
        return processed_count
