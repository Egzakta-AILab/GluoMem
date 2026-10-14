"""Tests for GluoMem Alembic migrations, CLI migrate subcommands, and scoped Postgres SQL."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from gluomem.cli import main
from gluomem.migrations import (
    get_heads,
    get_revisions,
    normalize_sqlalchemy_url,
    render_migration_sql,
)
from gluomem.storage.postgres_backend import PgTsVectorStore, PgVectorStore, _PgRow
from gluomem.storage.scope import MemoryScope


def test_alembic_revisions_and_heads() -> None:
    heads = get_heads()
    assert heads == ["0001_initial_scoped_schema"]
    revs = get_revisions()
    assert len(revs) >= 1
    assert revs[0]["revision"] == "0001_initial_scoped_schema"
    assert revs[0]["down_revision"] is None


def test_normalize_sqlalchemy_url() -> None:
    assert (
        normalize_sqlalchemy_url("postgres://u:p@localhost:5432/db")
        == "postgresql+psycopg://u:p@localhost:5432/db"
    )
    assert (
        normalize_sqlalchemy_url("postgresql://u:p@localhost:5432/db")
        == "postgresql+psycopg://u:p@localhost:5432/db"
    )
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_sqlalchemy_url("   ")


def test_alembic_offline_sql_upgrade_and_downgrade() -> None:
    up_sql = render_migration_sql(
        direction="upgrade",
        schema="gluomem_custom",
        vector_dim=768,
    )
    assert "CREATE SCHEMA IF NOT EXISTS gluomem_custom" in up_sql
    assert "vector(768)" in up_sql
    assert "idx_frames_vec_hnsw" in up_sql
    assert "idx_chunks_vec_hnsw" in up_sql
    assert "tenant_id" in up_sql and "TEXT NOT NULL DEFAULT 'local'" in up_sql
    assert "gluomem_custom.alembic_version" in up_sql

    down_sql = render_migration_sql(
        direction="downgrade",
        schema="gluomem_custom",
        vector_dim=768,
    )
    assert "DROP TABLE IF EXISTS gluomem_custom.memory_frames CASCADE" in down_sql
    assert "DROP TABLE IF EXISTS gluomem_custom.sessions CASCADE" in down_sql


def test_cli_migrate_heads_and_sql(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["migrate", "heads"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out == "0001_initial_scoped_schema"

    rc_up = main(
        [
            "migrate",
            "upgrade",
            "--sql",
            "--schema",
            "tenant_schema",
            "--vector-dim",
            "512",
        ]
    )
    assert rc_up == 0
    up_out = capsys.readouterr().out
    assert "CREATE SCHEMA IF NOT EXISTS tenant_schema" in up_out
    assert "vector(512)" in up_out

    rc_down = main(
        [
            "migrate",
            "downgrade",
            "0001_initial_scoped_schema:base",
            "--sql",
            "--schema",
            "tenant_schema",
        ]
    )
    assert rc_down == 0
    down_out = capsys.readouterr().out
    assert "DROP TABLE IF EXISTS tenant_schema.memory_frames CASCADE" in down_out


def test_pg_vector_and_tsvector_scoped_sql_generation() -> None:
    calls: list[tuple[str, tuple[Any, ...]]] = []

    class FakeBackend:
        def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
            calls.append((sql, tuple(params)))

        def fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[_PgRow]:
            calls.append((sql, tuple(params)))
            return [_PgRow({"rowid": 11, "id": 11, "distance": 0.12})]

    backend = FakeBackend()
    scope = MemoryScope("t_eu", "p_99", "workspace")
    qvec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    fvec = PgVectorStore(backend, table="memory_frames_vec", dim=4)  # type: ignore[arg-type]
    assert fvec.search(qvec, k=3, scope=scope, now_iso="2026-05-01T00:00:00") == [
        (11, 0.12)
    ]
    sql_fvec, params_fvec = calls[-1]
    assert "JOIN memory_frames m ON m.id = v.rowid" in sql_fvec
    assert "m.tenant_id = ?" in sql_fvec
    assert ("t_eu", "p_99", "workspace") == params_fvec[1:4]

    cvec = PgVectorStore(backend, table="memory_frame_chunks_vec", dim=4)  # type: ignore[arg-type]
    assert cvec.search(qvec, k=3, scope=scope) == [(11, 0.12)]
    sql_cvec, _ = calls[-1]
    assert "JOIN memory_frame_chunks c ON c.id = v.rowid" in sql_cvec
    assert "JOIN memory_frames m ON m.id = c.frame_id" in sql_cvec

    fts = PgTsVectorStore(backend)  # type: ignore[arg-type]
    assert fts.search(
        "hyperion", limit=5, scope=scope, now_iso="2026-05-01T00:00:00"
    ) == [11]
    sql_fts, params_fts = calls[-1]
    assert "m.tenant_id = ?" in sql_fts
    assert "websearch_to_tsquery('simple', ?)" in sql_fts
    assert "websearch_to_tsquery('english', ?)" in sql_fts
    assert ("t_eu", "p_99", "workspace") == params_fts[2:5]

    assert fts.search_like("hyperion", limit=5, scope=scope) == [11]
    sql_like, params_like = calls[-1]
    assert "tenant_id = ?" in sql_like
    assert "folded_content LIKE ?" in sql_like
    assert ("t_eu", "p_99", "workspace") == params_like[2:5]
