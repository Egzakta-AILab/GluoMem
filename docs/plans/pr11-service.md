# Service, queue, and client repairs

Observable: concurrent requests cannot share transaction state; leases are atomic
and attempt-fenced; stale jobs retry at the current generation without resurrecting
forgotten evidence; both SDK transports implement the same authenticated API.

Owned changes: `gluomem/service/**`, `gluomem/client/**`, serve/worker sections of
`gluomem/cli.py`, service/client extras in `pyproject.toml`, `tests/service/**`,
`tests/contracts/test_job_recovery.py`, and only the `memory_jobs` tables in both
storage SQL schemas.

Delete budget accounting, terminal stale/unused failed states, duplicate SDK
operations, handwritten response conversion, and dead request/response fields
first. Delete budget assertions and replace the duplicated auth/SDK scenarios with
parametrized contracts. The queue contract proves duplicate enqueue, exclusive
lease ownership, recovery, and attempt fencing; the HTTP concurrency contract
proves a failed request cannot roll back another request's persisted write.

Use request-owned database connections, bounded Pydantic-only wire models, a
shared request-spec operations layer, and signal-aware worker shutdown. Keep
authentication as a shared-secret gateway trust model, not principal verification.
Remove unused service dependencies and expose existing httpx as a client extra.

Integration dependencies: atomic lifecycle ingest/generation reads and the
dedicated EventConflictError land on the lifecycle branch; Postgres transaction
and metadata schema fixes land elsewhere. Do not edit those modules. The parent
must regenerate `uv.lock` after combining extra changes.
The sync payload-conflict contract is strict-xfailed only while the dedicated
exception is absent; it automatically runs normally after integration. Mid-ingest
crash/retry passes without an xfail because the worker owns the outer transaction.

The service now requires file-backed SQLite (or PostgreSQL); independent
connections cannot reopen a private `:memory:` database. Wire responses live in
`gluomem/client/models.py`; canonical message validation stays server-side.
Both list and export serialize each frame once and group by ids. Existing
databases need the integrator's migration for the narrowed job CHECK, dropped
budget columns, and legacy space-separated job timestamps.

Verification: `uv run ruff check . && uv run pyrefly check && uv run pytest -q`;
client import profiling; `org dev census --diff 30b7d4a` and `org dev perf service`.
Commit locally on `fix/pr11-service`; do not push or open a PR from this subtask.

The org measurement commands are KVARK-specific: `--repo` produces an all-zero
census here, and `service` is not a supported perf surface. Report actual
GluoMem source/test line and AST test-function deltas separately, not the
misleading default census from another checkout.
