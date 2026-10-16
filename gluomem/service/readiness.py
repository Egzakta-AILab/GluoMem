"""Fleet liveness, not proof of model reachability or successful extraction."""

from gluomem.storage.backend import DatabaseBackend
from gluomem.timeutil import now_utc, parse_utc

WORKER_HEARTBEAT_KEY = "operations.worker_poll_at"


def record_worker_poll(backend: DatabaseBackend, *, now_iso: str | None = None) -> None:
    """Record a successful queue poll; no job content or worker identity is retained."""
    stamp = parse_utc(now_iso or now_utc()).strftime("%Y-%m-%dT%H:%M:%S.%f")
    with backend.transaction():
        backend.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE "
            "SET value = excluded.value WHERE meta.value < excluded.value",
            (WORKER_HEARTBEAT_KEY, stamp),
        )


def worker_is_live(backend: DatabaseBackend, *, stale_after_seconds: float = 30.0) -> bool:
    """A missing, corrupt, future-dated or expired fleet heartbeat is not live."""
    stamp = backend.scalar("SELECT value FROM meta WHERE key = ?", (WORKER_HEARTBEAT_KEY,))
    if not stamp:
        return False
    try:
        age = (now_utc() - parse_utc(stamp)).total_seconds()
    except (TypeError, ValueError):
        return False
    return 0 <= age < stale_after_seconds
