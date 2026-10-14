"""Live schema creation, reversal and metadata drift contracts."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from gluomem.migrations import get_current, make_alembic_config, run_downgrade, run_upgrade
from gluomem.storage.db import MindDB

pytestmark = pytest.mark.postgres


def test_two_schemas_upgrade_downgrade_and_metadata(
    pg_dsn: str, pg_schema_factory: Callable[[int], str]
) -> None:
    from alembic import command

    first, second = pg_schema_factory(384), pg_schema_factory(4)
    assert get_current(pg_dsn, schema=first) == ["0001"]
    assert get_current(pg_dsn, schema=second) == ["0001"]
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
