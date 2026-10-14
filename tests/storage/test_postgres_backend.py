"""PostgreSQL backend unit tests and optional live database parity tests."""

from __future__ import annotations

import pytest

from gluomem.storage.postgres_backend import (
    PostgresBackend,
    _adapt_qmark_to_pyformat,
    _PgRow,
)


def test_sql_adaptation_and_pg_row() -> None:
    assert _adapt_qmark_to_pyformat("SELECT * FROM meta WHERE key = ?") == (
        "SELECT * FROM meta WHERE key = %s"
    )
    assert _adapt_qmark_to_pyformat("SELECT '?' FROM t WHERE a = ? AND b LIKE 'x%y'") == (
        "SELECT '?' FROM t WHERE a = %s AND b LIKE 'x%%y'"
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


def test_postgres_backend_validates_inputs() -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        PostgresBackend("postgresql://localhost/db", dim=0)
    with pytest.raises(ValueError, match="non-empty dsn"):
        PostgresBackend("", dim=4)
    with pytest.raises(ValueError, match="unsafe PostgreSQL schema"):
        PostgresBackend("postgresql://localhost/db", dim=4, postgres_schema="bad-name!")


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT ? -- ? 10%\n", "SELECT %s -- ? 10%%\n"),
        ("SELECT ? /* ? /* ? */ 5% */", "SELECT %s /* ? /* ? */ 5%% */"),
        ("SELECT \"?\", 'it''s ? 5%', ?", "SELECT \"?\", 'it''s ? 5%%', %s"),
        ("SELECT $$? 5%$$, $tag$?$tag$, ?", "SELECT $$? 5%%$$, $tag$?$tag$, %s"),
        (r"SELECT E'it\'s ? 5%', ?", r"SELECT E'it\'s ? 5%%', %s"),
    ],
)
def test_qmark_literals_and_comments(sql: str, expected: str) -> None:
    assert _adapt_qmark_to_pyformat(sql) == expected
