"""Durable job queue with atomic leases and attempt fencing."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal

from gluomem.frames.lifecycle import get_scope_generation
from gluomem.frames.source_fences import assert_source_not_retracted
from gluomem.storage.backend import DatabaseBackend
from gluomem.storage.scope import DEFAULT_SCOPE, MemoryScope, scope_sql
from gluomem.timeutil import now_utc, parse_utc
from gluomem.write.events import EventConflictError

JobStatus = Literal["queued", "leased", "succeeded", "dead_letter"]
JobType = Literal["ingest_event", "cognify_frame", "compact_gop", "sweep_expired"]
_JOB_COLS = (
    "id, tenant_id, principal_id, namespace, job_type, dedup_key, payload, "
    "status, attempt_count, max_attempts, scope_generation, worker_id, "
    "lease_expires_at, available_at, last_error, created_at, updated_at"
)


class BackpressureError(RuntimeError):
    """A scope has reached its pending job limit."""

    def __init__(self, scope: MemoryScope, pending: int, limit: int) -> None:
        super().__init__(f"{pending} pending jobs >= limit {limit}")
        self.scope, self.pending, self.limit = scope, pending, limit


class LeaseLostError(RuntimeError):
    """The worker no longer owns this lease attempt."""


class StaleScopeGenerationError(RuntimeError):
    """The scope generation changed before the job's write."""

    def __init__(self, job_id: int, job_generation: int, current_generation: int) -> None:
        super().__init__(f"job {job_id} generation {job_generation} != {current_generation}")
        self.current_generation = current_generation


@dataclass(frozen=True)
class MemoryJob:
    """A durable ``memory_jobs`` row."""

    id: int
    scope: MemoryScope
    job_type: str
    dedup_key: str
    payload: dict[str, Any]
    status: JobStatus
    attempt_count: int
    max_attempts: int
    scope_generation: int
    worker_id: str | None
    lease_expires_at: str | None
    available_at: str
    last_error: str | None
    created_at: str
    updated_at: str


def _row_to_job(row: Mapping[str, Any]) -> MemoryJob:
    fields = dict(row)
    scope = MemoryScope(*(fields.pop(key) for key in ("tenant_id", "principal_id", "namespace")))
    payload = fields["payload"]
    fields["payload"] = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(fields["payload"], dict):
        raise ValueError("job payload must be a JSON object")
    return MemoryJob(scope=scope, **fields)


def _resolve_now(now_iso: str | None) -> str:
    return parse_utc(now_iso if now_iso is not None else now_utc()).strftime("%Y-%m-%dT%H:%M:%S.%f")


def _add_seconds(base_iso: str, seconds: float) -> str:
    return (parse_utc(base_iso) + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%S.%f")


@contextmanager
def write_transaction(backend: DatabaseBackend) -> Iterator[None]:
    """Acquire SQLite's write lock before any read-modify-write operation."""
    outer = not backend.in_transaction
    with backend.transaction():
        if outer and backend.dialect == "sqlite":
            backend.execute("BEGIN IMMEDIATE")
        yield


class JobQueue:
    """Durable queue backed by SQLite or PostgreSQL."""

    def __init__(
        self,
        backend: DatabaseBackend,
        *,
        max_pending_per_scope: int = 50,
        default_max_attempts: int = 3,
    ) -> None:
        self._backend = backend
        self._max_pending_per_scope = max_pending_per_scope
        self._default_max_attempts = default_max_attempts

    def get_job(self, job_id: int) -> MemoryJob:
        row = self._backend.fetchone(f"SELECT {_JOB_COLS} FROM memory_jobs WHERE id = ?", (job_id,))
        if row is None:
            raise KeyError(job_id)
        return _row_to_job(row)

    def pending_count(self, scope: MemoryScope) -> int:
        return int(
            self._backend.scalar(
                f"SELECT COUNT(*) FROM memory_jobs WHERE {scope_sql()} "
                "AND status IN ('queued', 'leased')",
                scope.as_tuple(),
            )
        )

    def enqueue(
        self,
        scope: MemoryScope = DEFAULT_SCOPE,
        job_type: str = "ingest_event",
        dedup_key: str = "",
        payload: dict[str, Any] | None = None,
        *,
        max_attempts: int | None = None,
        max_pending_per_scope: int | None = None,
        now_iso: str | None = None,
    ) -> MemoryJob:
        """Enqueue once per key, rejecting conflicting payloads and excess pending work."""
        if not dedup_key.strip():
            raise ValueError("dedup_key must be non-empty")
        attempts = max_attempts if max_attempts is not None else self._default_max_attempts
        limit = (
            max_pending_per_scope
            if max_pending_per_scope is not None
            else self._max_pending_per_scope
        )
        if attempts < 1 or limit < 1:
            raise ValueError("attempt and pending limits must be positive")
        now = _resolve_now(now_iso)
        body = payload if payload is not None else {}
        with write_transaction(self._backend):
            if job_type == "ingest_event" and isinstance(body.get("source_ref"), str):
                assert_source_not_retracted(self._backend, scope, body["source_ref"], lock=True)
            if self._backend.dialect == "postgres":
                self._backend.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(?))", (json.dumps(scope.as_tuple()),)
                )
            existing = self._backend.fetchone(
                f"SELECT {_JOB_COLS} FROM memory_jobs WHERE {scope_sql()} "
                "AND job_type = ? AND dedup_key = ?",
                (*scope.as_tuple(), job_type, dedup_key),
            )
            if existing is not None:
                job = _row_to_job(existing)
                if job.payload != body:
                    raise EventConflictError("job key already has a different payload")
                return job
            pending = self.pending_count(scope)
            if pending >= limit:
                raise BackpressureError(scope, pending, limit)
            row = self._backend.fetchone(
                "INSERT INTO memory_jobs (tenant_id, principal_id, namespace, job_type, "
                "dedup_key, payload, max_attempts, scope_generation, available_at, created_at, "
                "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (tenant_id, principal_id, namespace, job_type, dedup_key) "
                "DO NOTHING RETURNING id",
                (
                    *scope.as_tuple(),
                    job_type,
                    dedup_key,
                    json.dumps(body, ensure_ascii=False),
                    attempts,
                    get_scope_generation(self._backend, scope),
                    now,
                    now,
                    now,
                ),
            )
            if row is None:
                row = self._backend.fetchone(
                    "SELECT id FROM memory_jobs "
                    f"WHERE {scope_sql()} AND job_type = ? AND dedup_key = ?",
                    (*scope.as_tuple(), job_type, dedup_key),
                )
            assert row is not None
            job = self.get_job(int(row["id"]))
            if job.payload != body:
                raise EventConflictError("job key already has a different payload")
            return job

    def reclaim_expired_leases(self, *, now_iso: str | None = None) -> int:
        """Requeue expired leases, dead-lettering exhausted attempts."""
        now = _resolve_now(now_iso)
        with write_transaction(self._backend):
            rows = self._backend.fetchall(
                "UPDATE memory_jobs SET status = CASE WHEN attempt_count >= max_attempts "
                "THEN 'dead_letter' ELSE 'queued' END, worker_id = NULL, lease_expires_at = NULL, "
                "available_at = ?, updated_at = ?, last_error = 'lease_expired' "
                "WHERE status = 'leased' AND lease_expires_at <= ? RETURNING id",
                (now, now, now),
            )
        return len(rows)

    def lease_next(
        self, worker_id: str, *, lease_seconds: float = 60.0, now_iso: str | None = None
    ) -> MemoryJob | None:
        """Atomically lease the next available job and increment its attempt token."""
        now = _resolve_now(now_iso)
        self.reclaim_expired_leases(now_iso=now)
        lock = " FOR UPDATE SKIP LOCKED" if self._backend.dialect == "postgres" else ""
        invalid: ValueError | None = None
        job: MemoryJob | None = None
        with write_transaction(self._backend):
            row = self._backend.fetchone(
                "UPDATE memory_jobs SET status = 'leased', worker_id = ?, "
                "attempt_count = attempt_count + 1, lease_expires_at = ?, updated_at = ? "
                "WHERE id = (SELECT id FROM memory_jobs WHERE status = 'queued' "
                f"AND available_at <= ? ORDER BY id LIMIT 1{lock}) AND status = 'queued' "
                f"RETURNING {_JOB_COLS}",
                (worker_id, _add_seconds(now, lease_seconds), now, now),
            )
            if row is not None:
                try:
                    job = _row_to_job(row)
                except ValueError as exc:
                    self._backend.execute(
                        "UPDATE memory_jobs SET status = 'dead_letter', worker_id = NULL, "
                        "lease_expires_at = NULL, last_error = ? WHERE id = ? "
                        "AND status = 'leased' AND worker_id = ? AND attempt_count = ?",
                        (str(exc), row["id"], worker_id, row["attempt_count"]),
                    )
                    invalid = exc
        if invalid is not None:
            raise invalid
        return job

    def _update_lease(
        self,
        job_id: int,
        worker_id: str,
        attempt_count: int,
        assignment: str,
        params: tuple[Any, ...],
    ) -> MemoryJob:
        with write_transaction(self._backend):
            row = self._backend.fetchone(
                f"UPDATE memory_jobs SET {assignment} WHERE id = ? AND status = 'leased' "
                f"AND worker_id = ? AND attempt_count = ? RETURNING {_JOB_COLS}",
                (*params, job_id, worker_id, attempt_count),
            )
            if row is None:
                raise LeaseLostError(f"job {job_id} lease attempt {attempt_count} lost")
            return _row_to_job(row)

    def complete(
        self, job_id: int, *, worker_id: str, attempt_count: int, now_iso: str | None = None
    ) -> MemoryJob:
        """Complete only the worker's current lease attempt."""
        return self._update_lease(
            job_id,
            worker_id,
            attempt_count,
            "status = 'succeeded', worker_id = NULL, lease_expires_at = NULL, updated_at = ?",
            (_resolve_now(now_iso),),
        )

    def fail(
        self,
        job_id: int,
        error: str,
        *,
        worker_id: str,
        attempt_count: int,
        retry_backoff_seconds: float = 5.0,
        now_iso: str | None = None,
    ) -> MemoryJob:
        """Requeue a failed lease with backoff or dead-letter exhausted attempts."""
        now = _resolve_now(now_iso)
        available = _add_seconds(now, retry_backoff_seconds * 2 ** max(0, attempt_count - 1))
        return self._update_lease(
            job_id,
            worker_id,
            attempt_count,
            "status = CASE WHEN attempt_count >= max_attempts THEN 'dead_letter' "
            "ELSE 'queued' END, "
            "worker_id = NULL, lease_expires_at = NULL, available_at = ?, last_error = ?, "
            "updated_at = ?",
            (available, error, now),
        )

    def requeue_stale(
        self, job: MemoryJob, generation: int, *, now_iso: str | None = None
    ) -> MemoryJob:
        """Retry a fenced job at the current generation without charging a failure."""
        now = _resolve_now(now_iso)
        assert job.worker_id is not None
        # Restoring the allowance must not reuse a previous lease's attempt token.
        return self._update_lease(
            job.id,
            job.worker_id,
            job.attempt_count,
            "status = 'queued', scope_generation = ?, max_attempts = max_attempts + 1, "
            "worker_id = NULL, lease_expires_at = NULL, available_at = ?, "
            "last_error = NULL, updated_at = ?",
            (generation, now, now),
        )

    def renew_lease(
        self,
        job_id: int,
        worker_id: str,
        *,
        attempt_count: int,
        lease_seconds: float = 60.0,
        now_iso: str | None = None,
    ) -> MemoryJob:
        """Extend only the worker's current lease attempt."""
        now = _resolve_now(now_iso)
        return self._update_lease(
            job_id,
            worker_id,
            attempt_count,
            "lease_expires_at = ?, updated_at = ?",
            (_add_seconds(now, lease_seconds), now),
        )

    def queue_stats(self, scope: MemoryScope | None = None) -> dict[str, Any]:
        """Return status counts and the age of the oldest pending job."""
        where = f"WHERE {scope_sql()}" if scope is not None else ""
        params = scope.as_tuple() if scope is not None else ()
        counts = dict.fromkeys(("queued", "leased", "succeeded", "dead_letter"), 0)
        for row in self._backend.fetchall(
            f"SELECT status, COUNT(*) AS cnt FROM memory_jobs {where} GROUP BY status", params
        ):
            counts[row["status"]] = int(row["cnt"])
        oldest = self._backend.scalar(
            f"SELECT MIN(created_at) FROM memory_jobs {where} "
            f"{'AND' if where else 'WHERE'} status IN ('queued', 'leased')",
            params,
        )
        age = max(0.0, (now_utc() - parse_utc(oldest)).total_seconds()) if oldest else 0.0
        return {
            "counts": counts,
            "pending": counts["queued"] + counts["leased"],
            "dead_letter": counts["dead_letter"],
            "oldest_pending_age_seconds": round(age, 3),
        }
