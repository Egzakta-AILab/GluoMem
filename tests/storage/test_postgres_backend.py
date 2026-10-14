"""PostgreSQL backend unit tests and optional live database parity tests."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

from gluomem.storage.backend import VectorDimMismatchError
from gluomem.storage.postgres_backend import (
    PgTsVectorStore,
    PgVectorStore,
    PostgresBackend,
    _adapt_qmark_to_pyformat,
    _PgRow,
)


def test_sql_adaptation_and_pg_row() -> None:
    assert _adapt_qmark_to_pyformat("SELECT * FROM meta WHERE key = ?") == (
        "SELECT * FROM meta WHERE key = %s"
    )
    assert _adapt_qmark_to_pyformat("SELECT '?' FROM t WHERE a = ? AND b LIKE 'x%y'") == (
        "SELECT '?' FROM t WHERE a = %s AND b LIKE 'x%y'"
    )

    row = _PgRow({"id": 7, "name": "alice"})
    assert row[0] == 7
    assert row["id"] == 7
    assert row["name"] == "alice"
    assert list(row) == [7, "alice"]
    assert len(row) == 2
    assert row.keys() == ["id", "name"]
    assert row.values() == [7, "alice"]
    with pytest.raises(KeyError):
        _ = row["missing"]


def test_pg_vector_store_and_tsvector_store_mocked() -> None:
    calls: list[tuple[str, tuple[Any, ...]]] = []

    class FakeBackend:
        def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
            calls.append((sql, tuple(params)))

        def executemany(self, sql: str, seq: list[tuple[Any, ...]]) -> None:
            calls.append((sql, tuple(seq)))

        def fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[_PgRow]:
            calls.append((sql, tuple(params)))
            return [
                _PgRow({"rowid": 1, "id": 1, "distance": 0.25, "content": "a"}),
                _PgRow({"rowid": 2, "id": 2, "distance": 0.5, "content": "b"}),
            ]

        def fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> _PgRow | None:
            calls.append((sql, tuple(params)))
            return _PgRow({"1": 1})

        def scalar(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
            calls.append((sql, tuple(params)))
            return 2

        def commit(self) -> None:
            pass

    backend = FakeBackend()
    vs = PgVectorStore(backend, table="memory_frames_vec", dim=4)  # type: ignore[arg-type]
    v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vs.add(1, v)
    assert "ON CONFLICT (rowid) DO UPDATE" in calls[-1][0]

    with pytest.raises(VectorDimMismatchError):
        vs.add(2, np.zeros(8, dtype=np.float32))

    assert vs.search(v, k=2) == [(1, 0.25), (2, 0.5)]
    assert vs.has(1) is True
    assert vs.count() == 2
    assert vs.ids() == {1, 2}
    vs.delete(1)
    assert "DELETE FROM memory_frames_vec" in calls[-1][0]
    vs.clear()

    fts = PgTsVectorStore(backend)  # type: ignore[arg-type]
    fts.upsert(1, "hello world")
    assert "INSERT INTO memory_frames_fts" in calls[-1][0]
    assert fts.search("hello world", limit=5) == [1, 2]
    assert fts.search("   ", limit=5) == []
    assert fts.search_like(["hello"], limit=5) == [1, 2]


def test_postgres_backend_validates_inputs() -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        PostgresBackend("postgresql://localhost/db", dim=0)
    with pytest.raises(ValueError, match="non-empty dsn"):
        PostgresBackend("", dim=4)
    with pytest.raises(ValueError, match="Invalid postgres_schema"):
        PostgresBackend("postgresql://localhost/db", dim=4, postgres_schema="bad-name!")


@pytest.mark.skipif(
    not os.environ.get("GLUOMEM_TEST_PG_DSN"),
    reason="GLUOMEM_TEST_PG_DSN not set",
)
def test_live_postgres_backend_roundtrip() -> None:
    dsn = os.environ["GLUOMEM_TEST_PG_DSN"]
    backend = PostgresBackend(dsn=dsn, dim=4, postgres_schema="gluomem_test")
    try:
        vs = PgVectorStore(backend, "memory_frames_vec", dim=4)
        v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        vs.add(1, v)
        assert vs.search(v, k=1)[0][0] == 1
    finally:
        backend.close()
