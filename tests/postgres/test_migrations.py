"""Live schema creation, reversal and metadata drift contracts."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from gluomem.migrations import (
    get_current,
    get_heads,
    make_alembic_config,
    run_downgrade,
    run_upgrade,
)
from gluomem.storage.db import MindDB

pytestmark = pytest.mark.postgres


def test_two_schemas_upgrade_downgrade_and_metadata(
    pg_dsn: str, pg_schema_factory: Callable[[int], str]
) -> None:
    from alembic import command

    first, second = pg_schema_factory(384), pg_schema_factory(4)
    assert get_current(pg_dsn, schema=first) == get_heads()
    assert get_current(pg_dsn, schema=second) == get_heads()
    for schema, dim in [(first, 384), (second, 4)]:
        command.check(make_alembic_config(pg_dsn, schema=schema, vector_dim=dim))
        run_downgrade(pg_dsn, schema=schema, vector_dim=dim)
        assert get_current(pg_dsn, schema=schema) == []
        run_upgrade(pg_dsn, schema=schema, vector_dim=dim)
        db = MindDB(backend_type="postgres", dsn=pg_dsn, postgres_schema=schema, dim=dim)
        try:
            assert db.scalar("SELECT COUNT(*) FROM memory_frames") == 0
        finally:
            db.close()


def test_downgrade_preserves_retraction_without_ingested_memory(
    mind_db: MindDB, frame_store, pg_dsn: str
) -> None:
    from sqlalchemy.exc import DBAPIError

    from gluomem.frames.lifecycle import retract_source
    from gluomem.storage.postgres_backend import PostgresBackend

    backend = mind_db.backend
    assert isinstance(backend, PostgresBackend)
    retract_source(frame_store, "source:never-ingested")
    assert mind_db.scalar("SELECT COUNT(*) FROM source_retractions") == 1
    assert mind_db.scalar("SELECT COUNT(*) FROM memory_tombstones") == 0
    with pytest.raises(DBAPIError, match="durable source retractions"):
        run_downgrade(pg_dsn, "0001", schema=backend.postgres_schema)
    assert get_current(pg_dsn, schema=backend.postgres_schema) == get_heads()
    assert mind_db.scalar("SELECT source_ref FROM source_retractions") == "source:never-ingested"


def test_downgrade_serializes_with_uncommitted_retraction(mind_db: MindDB, pg_dsn: str) -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor

    import psycopg
    from psycopg import sql
    from sqlalchemy.exc import DBAPIError

    from gluomem.storage.postgres_backend import PostgresBackend

    backend = mind_db.backend
    assert isinstance(backend, PostgresBackend)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with psycopg.connect(pg_dsn) as writer:
            writer.execute(
                sql.SQL("SET search_path TO {}, public").format(
                    sql.Identifier(backend.postgres_schema)
                )
            )
            writer.execute(
                "INSERT INTO source_retractions VALUES "
                "('local', 'default', 'personal', 'concurrent-source', '2026-01-01T00:00:00Z')"
            )
            future = pool.submit(run_downgrade, pg_dsn, "0001", schema=backend.postgres_schema)
            try:
                deadline = time.monotonic() + 4
                while not mind_db.scalar(
                    "SELECT EXISTS (SELECT 1 FROM pg_stat_activity "
                    "WHERE ? = ANY(pg_blocking_pids(pid)) AND query = ?)",
                    (
                        writer.info.backend_pid,
                        "LOCK TABLE source_retractions IN ACCESS EXCLUSIVE MODE",
                    ),
                ):
                    assert time.monotonic() < deadline, "downgrade did not lock before checking"
                    time.sleep(0.01)
            finally:
                writer.commit()
        with pytest.raises(DBAPIError, match="durable source retractions"):
            future.result(timeout=5)
    assert get_current(pg_dsn, schema=backend.postgres_schema) == get_heads()
    assert mind_db.scalar("SELECT source_ref FROM source_retractions") == "concurrent-source"
