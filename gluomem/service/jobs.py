"""Durable SQLite/PostgreSQL ``memory_jobs`` queue with leases, backpressure, and fencing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal, cast

from gluomem.frames.lifecycle import get_scope_generation
from gluomem.storage.backend import DatabaseBackend
from gluomem.storage.scope import DEFAULT_SCOPE, MemoryScope
from gluomem.timeutil import now_utc, parse_utc, to_storage, utc_key

JobStatus = Literal[
    "queued",
    "leased",
    "succeeded",
    "failed",
    "dead_letter",
    "aborted_stale",
]
JobType = Literal[
    "ingest_event",
    "cognify_frame",
    "compact_gop",
]

_JOB_COLS = (
    "id, tenant_id, principal_id, namespace, job_type, dedup_key, payload, "
    "status, attempt_count, max_attempts, scope_generation, worker_id, "
    "lease_expires_at, available_at, llm_calls_used, max_llm_calls, "
    "tokens_used, max_tokens, last_error, created_at, updated_at"
)


class BackpressureError(RuntimeError):
    """Raised when a scope exceeds ``max_pending_per_scope`` active jobs."""

    def __init__(self, scope: MemoryScope, pending: int, limit: int) -> None:
        super().__init__(
            f"backpressure limit exceeded for scope {scope.as_tuple()}: "
            f"{pending} pending jobs >= limit {limit}"
        )
        self.scope = scope
        self.pending = pending
        self.limit = limit


class JobBudgetExceededError(RuntimeError):
    """Raised when a job exceeds its per-job LLM call or token cap."""

    def __init__(self, job_id: int, reason: str) -> None:
        super().__init__(f"job {job_id} budget exceeded: {reason}")
        self.job_id = job_id
        self.reason = reason


class StaleScopeGenerationError(RuntimeError):
    """Raised when a job's ``scope_generation`` lags the scope's current generation."""

    def __init__(
        self, job_id: int, job_generation: int, current_generation: int
    ) -> None:
        super().__init__(
            f"job {job_id} fenced: job scope_generation={job_generation} "
            f"< current_generation={current_generation}"
        )
        self.job_id = job_id
        self.job_generation = job_generation
        self.current_generation = current_generation


@dataclass(frozen=True)
class MemoryJob:
    """Durable background job row in ``memory_jobs``."""

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
    llm_calls_used: int
    max_llm_calls: int
    tokens_used: int
    max_tokens: int
    last_error: str | None
    created_at: str
    updated_at: str


def _row_to_job(row: Any) -> MemoryJob:
    raw_payload = row["payload"]
    if isinstance(raw_payload, str):
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            payload = {}
    elif isinstance(raw_payload, dict):
        payload = dict(raw_payload)
    else:
        payload = {}

    return MemoryJob(
        id=int(row["id"]),
        scope=MemoryScope(
            tenant_id=str(row["tenant_id"]),
            principal_id=str(row["principal_id"]),
            namespace=str(row["namespace"]),
        ),
        job_type=str(row["job_type"]),
        dedup_key=str(row["dedup_key"]),
        payload=payload,
        status=cast(JobStatus, str(row["status"])),
        attempt_count=int(row["attempt_count"]),
        max_attempts=int(row["max_attempts"]),
        scope_generation=int(row["scope_generation"]),
        worker_id=str(row["worker_id"]) if row["worker_id"] is not None else None,
        lease_expires_at=(
            str(row["lease_expires_at"])
            if row["lease_expires_at"] is not None
            else None
        ),
        available_at=str(row["available_at"]),
        llm_calls_used=int(row["llm_calls_used"]),
        max_llm_calls=int(row["max_llm_calls"]),
        tokens_used=int(row["tokens_used"]),
        max_tokens=int(row["max_tokens"]),
        last_error=str(row["last_error"]) if row["last_error"] is not None else None,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _resolve_now(now_iso: str | None) -> str:
    val = to_storage(now_iso) if now_iso is not None else to_storage(now_utc())
    assert val is not None
    return val


def _add_seconds(base_iso: str, seconds: float) -> str:
    dt: datetime = parse_utc(base_iso)
    val = to_storage(dt + timedelta(seconds=float(seconds)))
    assert val is not None
    return val


class JobQueue:
    """Durable job queue backed by ``memory_jobs`` (SQLite and PostgreSQL)."""

    def __init__(
        self,
        backend: DatabaseBackend,
        *,
        max_pending_per_scope: int = 50,
        default_max_attempts: int = 3,
    ) -> None:
        self._backend = backend
        self._max_pending_per_scope = int(max_pending_per_scope)
        self._default_max_attempts = int(default_max_attempts)

    def get_job(self, job_id: int) -> MemoryJob:
        row = self._backend.fetchone(
            f"SELECT {_JOB_COLS} FROM memory_jobs WHERE id = ?",
            (int(job_id),),
        )
        if row is None:
            raise KeyError(f"no memory_job with id={job_id}")
        return _row_to_job(row)

    def pending_count(self, scope: MemoryScope) -> int:
        row = self._backend.fetchone(
            "SELECT COUNT(*) AS cnt FROM memory_jobs "
            "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? "
            "AND status IN ('queued', 'leased')",
            scope.as_tuple(),
        )
        return int(row["cnt"]) if row is not None else 0

    def record_usage(
        self,
        job_id: int,
        *,
        llm_calls: int = 0,
        tokens: int = 0,
    ) -> MemoryJob:
        """Increment ``llm_calls_used`` and ``tokens_used`` and enforce per-job caps."""
        with self._backend.transaction():
            self._backend.execute(
                "UPDATE memory_jobs SET "
                "llm_calls_used = llm_calls_used + ?, "
                "tokens_used = tokens_used + ? "
                "WHERE id = ?",
                (int(llm_calls), int(tokens), int(job_id)),
            )
        job = self.get_job(job_id)
        if job.llm_calls_used > job.max_llm_calls:
            raise JobBudgetExceededError(
                job_id,
                f"llm_calls_used={job.llm_calls_used} > max_llm_calls={job.max_llm_calls}",
            )
        if job.tokens_used > job.max_tokens:
            raise JobBudgetExceededError(
                job_id,
                f"tokens_used={job.tokens_used} > max_tokens={job.max_tokens}",
            )
        return job

    def check_fencing(self, job: MemoryJob) -> bool:
        """Return ``True`` iff ``job.scope_generation`` matches the scope's current generation."""
        current_gen = get_scope_generation(self._backend, job.scope)
        return current_gen == job.scope_generation

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
        """Idempotently enqueue a job in ``scope`` while enforcing per-scope backpressure."""
        if not dedup_key or not dedup_key.strip():
            raise ValueError("dedup_key must be non-empty")
        eff_now = _resolve_now(now_iso)
        eff_max_attempts = (
            int(max_attempts)
            if max_attempts is not None
            else self._default_max_attempts
        )
        eff_limit = (
            int(max_pending_per_scope)
            if max_pending_per_scope is not None
            else self._max_pending_per_scope
        )
        payload_json = json.dumps(
            payload or {}, ensure_ascii=False, separators=(",", ":")
        )

        with self._backend.transaction():
            existing = self._backend.fetchone(
                f"SELECT {_JOB_COLS} FROM memory_jobs "
                "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? "
                "AND job_type = ? AND dedup_key = ?",
                (
                    scope.tenant_id,
                    scope.principal_id,
                    scope.namespace,
                    job_type,
                    dedup_key,
                ),
            )
            if existing is not None:
                existing_job = _row_to_job(existing)
                if existing_job.status in ("queued", "leased", "succeeded"):
                    return existing_job

            pending = self.pending_count(scope)
            if pending >= eff_limit:
                raise BackpressureError(scope, pending, eff_limit)

            current_gen = get_scope_generation(self._backend, scope)
            if existing is not None:
                self._backend.execute(
                    "UPDATE memory_jobs SET payload = ?, status = 'queued', "
                    "attempt_count = 0, max_attempts = ?, scope_generation = ?, "
                    "worker_id = NULL, lease_expires_at = NULL, available_at = ?, "
                    "last_error = NULL, updated_at = ? WHERE id = ?",
                    (
                        payload_json,
                        eff_max_attempts,
                        current_gen,
                        eff_now,
                        eff_now,
                        int(existing["id"]),
                    ),
                )
                job_id = int(existing["id"])
            else:
                row = self._backend.fetchone(
                    "INSERT INTO memory_jobs "
                    "(tenant_id, principal_id, namespace, job_type, dedup_key, "
                    "payload, status, attempt_count, max_attempts, scope_generation, "
                    "worker_id, lease_expires_at, available_at, last_error, "
                    "created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'queued', 0, ?, ?, NULL, NULL, ?, NULL, ?, ?) "
                    "RETURNING id",
                    (
                        scope.tenant_id,
                        scope.principal_id,
                        scope.namespace,
                        job_type,
                        dedup_key,
                        payload_json,
                        eff_max_attempts,
                        current_gen,
                        eff_now,
                        eff_now,
                        eff_now,
                    ),
                )
                if row is None:
                    raise RuntimeError("memory_jobs INSERT did not return an id")
                job_id = int(row["id"])

        return self.get_job(job_id)

    def reclaim_expired_leases(self, *, now_iso: str | None = None) -> int:
        """Reclaim leased jobs whose ``lease_expires_at <= now_iso`` (or dead-letter)."""
        eff_now = _resolve_now(now_iso)
        now_key = utc_key(eff_now)
        expired_rows = self._backend.fetchall(
            f"SELECT {_JOB_COLS} FROM memory_jobs "
            "WHERE status = 'leased' AND lease_expires_at IS NOT NULL "
            "AND REPLACE(lease_expires_at, ' ', 'T') <= ? "
            "ORDER BY id ASC",
            (now_key,),
        )
        if not expired_rows:
            return 0

        with self._backend.transaction():
            for row in expired_rows:
                job = _row_to_job(row)
                if job.attempt_count >= job.max_attempts:
                    self._backend.execute(
                        "UPDATE memory_jobs SET status = 'dead_letter', "
                        "worker_id = NULL, lease_expires_at = NULL, "
                        "last_error = COALESCE(last_error, 'lease_expired_max_attempts'), "
                        "updated_at = ? WHERE id = ?",
                        (eff_now, job.id),
                    )
                else:
                    self._backend.execute(
                        "UPDATE memory_jobs SET status = 'queued', "
                        "worker_id = NULL, lease_expires_at = NULL, "
                        "available_at = ?, updated_at = ? WHERE id = ?",
                        (eff_now, eff_now, job.id),
                    )
        return len(expired_rows)

    def lease_next(
        self,
        worker_id: str,
        *,
        lease_seconds: float = 60.0,
        now_iso: str | None = None,
    ) -> MemoryJob | None:
        """Reclaim expired leases and atomically lease the next available queued job."""
        eff_now = _resolve_now(now_iso)
        now_key = utc_key(eff_now)
        lease_until = _add_seconds(eff_now, lease_seconds)

        self.reclaim_expired_leases(now_iso=eff_now)

        lock_clause = (
            " FOR UPDATE SKIP LOCKED" if self._backend.dialect == "postgres" else ""
        )
        with self._backend.transaction():
            row = self._backend.fetchone(
                "SELECT id FROM memory_jobs "
                "WHERE status = 'queued' "
                "AND REPLACE(available_at, ' ', 'T') <= ? "
                f"ORDER BY id ASC LIMIT 1{lock_clause}",
                (now_key,),
            )
            if row is None:
                return None
            job_id = int(row["id"])
            self._backend.execute(
                "UPDATE memory_jobs SET status = 'leased', worker_id = ?, "
                "attempt_count = attempt_count + 1, lease_expires_at = ?, "
                "updated_at = ? WHERE id = ?",
                (worker_id, lease_until, eff_now, job_id),
            )
        return self.get_job(job_id)

    def mark_aborted_stale(
        self,
        job_id: int,
        *,
        current_generation: int,
        now_iso: str | None = None,
    ) -> MemoryJob:
        """Mark a job ``aborted_stale`` when ``scope_generation`` moved ahead."""
        eff_now = _resolve_now(now_iso)
        job = self.get_job(job_id)
        err = f"stale_scope_generation:{job.scope_generation}<{current_generation}"
        with self._backend.transaction():
            self._backend.execute(
                "UPDATE memory_jobs SET status = 'aborted_stale', "
                "worker_id = NULL, lease_expires_at = NULL, "
                "last_error = ?, updated_at = ? WHERE id = ?",
                (err, eff_now, job_id),
            )
        return self.get_job(job_id)

    def complete(
        self,
        job_id: int,
        *,
        worker_id: str | None = None,
        now_iso: str | None = None,
    ) -> MemoryJob:
        """Complete a leased job after verifying ``scope_generation`` fencing."""
        eff_now = _resolve_now(now_iso)
        job = self.get_job(job_id)
        if worker_id is not None and job.worker_id not in (None, worker_id):
            raise RuntimeError(
                f"job {job_id} is leased by {job.worker_id!r}, not {worker_id!r}"
            )
        current_gen = get_scope_generation(self._backend, job.scope)
        if current_gen != job.scope_generation:
            self.mark_aborted_stale(
                job_id, current_generation=current_gen, now_iso=eff_now
            )
            raise StaleScopeGenerationError(
                job_id, job.scope_generation, current_gen
            )

        with self._backend.transaction():
            self._backend.execute(
                "UPDATE memory_jobs SET status = 'succeeded', "
                "lease_expires_at = NULL, updated_at = ? WHERE id = ?",
                (eff_now, job_id),
            )
        return self.get_job(job_id)

    def fail(
        self,
        job_id: int,
        error: str,
        *,
        worker_id: str | None = None,
        retry_backoff_seconds: float = 5.0,
        now_iso: str | None = None,
    ) -> MemoryJob:
        """Record a job failure; requeue with exponential backoff or dead-letter when exhausted."""
        eff_now = _resolve_now(now_iso)
        job = self.get_job(job_id)
        if worker_id is not None and job.worker_id not in (None, worker_id):
            raise RuntimeError(
                f"job {job_id} is leased by {job.worker_id!r}, not {worker_id!r}"
            )

        if job.attempt_count >= job.max_attempts:
            with self._backend.transaction():
                self._backend.execute(
                    "UPDATE memory_jobs SET status = 'dead_letter', "
                    "worker_id = NULL, lease_expires_at = NULL, "
                    "last_error = ?, updated_at = ? WHERE id = ?",
                    (error, eff_now, job_id),
                )
        else:
            delay = float(retry_backoff_seconds) * (
                2 ** max(0, job.attempt_count - 1)
            )
            next_avail = _add_seconds(eff_now, delay)
            with self._backend.transaction():
                self._backend.execute(
                    "UPDATE memory_jobs SET status = 'queued', "
                    "worker_id = NULL, lease_expires_at = NULL, "
                    "available_at = ?, last_error = ?, updated_at = ? WHERE id = ?",
                    (next_avail, error, eff_now, job_id),
                )
        return self.get_job(job_id)

    def queue_stats(self, scope: MemoryScope | None = None) -> dict[str, Any]:
        """Return queue counts by status and ``oldest_pending_age_seconds``."""
        counts: dict[str, int] = {
            "queued": 0,
            "leased": 0,
            "succeeded": 0,
            "failed": 0,
            "dead_letter": 0,
            "aborted_stale": 0,
        }
        if scope is not None:
            rows = self._backend.fetchall(
                "SELECT status, COUNT(*) AS cnt FROM memory_jobs "
                "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? "
                "GROUP BY status",
                scope.as_tuple(),
            )
            oldest_row = self._backend.fetchone(
                "SELECT MIN(created_at) AS oldest FROM memory_jobs "
                "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? "
                "AND status IN ('queued', 'leased')",
                scope.as_tuple(),
            )
        else:
            rows = self._backend.fetchall(
                "SELECT status, COUNT(*) AS cnt FROM memory_jobs GROUP BY status"
            )
            oldest_row = self._backend.fetchone(
                "SELECT MIN(created_at) AS oldest FROM memory_jobs "
                "WHERE status IN ('queued', 'leased')"
            )

        for r in rows:
            counts[str(r["status"])] = int(r["cnt"])

        oldest_age = 0.0
        if oldest_row is not None and oldest_row["oldest"] is not None:
            oldest_dt = parse_utc(str(oldest_row["oldest"]))
            oldest_age = max(0.0, (now_utc() - oldest_dt).total_seconds())

        return {
            "counts": counts,
            "pending": counts.get("queued", 0) + counts.get("leased", 0),
            "dead_letter": counts.get("dead_letter", 0),
            "oldest_pending_age_seconds": round(oldest_age, 3),
        }
