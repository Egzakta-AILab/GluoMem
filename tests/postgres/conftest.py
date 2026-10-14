"""Isolated live PostgreSQL fixtures, enabled by ``GLUOMEM_TEST_PG_DSN``."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from uuid import uuid4

import pytest

from gluomem.migrations import run_upgrade
from gluomem.storage.db import MindDB


@pytest.fixture
def pg_dsn() -> str:
    dsn = os.environ.get("GLUOMEM_TEST_PG_DSN")
    if not dsn:
        pytest.skip("GLUOMEM_TEST_PG_DSN not set")
    return dsn


@pytest.fixture
def pg_schema_factory(pg_dsn: str) -> Iterator[Callable[[int], str]]:
    import psycopg
    from psycopg import sql

    schemas: list[str] = []

    def create(dim: int = 384) -> str:
        schema = f"test_{uuid4().hex}"
        schemas.append(schema)
        run_upgrade(pg_dsn, schema=schema, vector_dim=dim)
        return schema

    try:
        yield create
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            for schema in schemas:
                conn.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema))
                )


@pytest.fixture
def mind_db(pg_dsn: str, pg_schema_factory: Callable[[int], str]) -> Iterator[MindDB]:
    db = MindDB(backend_type="postgres", dsn=pg_dsn, postgres_schema=pg_schema_factory(384))
    try:
        yield db
    finally:
        db.close()
