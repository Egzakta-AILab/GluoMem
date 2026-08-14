"""MindDB pragma tests.

A freshly opened ``*.mind`` DB must run in WAL mode, enforce foreign keys, and
have sqlite-vec loaded (``vec_version()`` non-empty). These tests drive the
public ``MindDB`` surface against a ``tmp_path`` ``*.mind`` file.
"""

from __future__ import annotations

from pathlib import Path

from gluomem.storage.db import MindDB


def _open(tmp_path: Path, dim: int = 384) -> MindDB:
    return MindDB(tmp_path / "test.mind", dim=dim)


def test_journal_mode_is_wal(tmp_path: Path) -> None:
    """PRAGMA journal_mode → 'wal'."""
    db = _open(tmp_path)
    mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"
    db.close()


def test_foreign_keys_on(tmp_path: Path) -> None:
    """PRAGMA foreign_keys → 1."""
    db = _open(tmp_path)
    fk = db.conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert fk == 1
    db.close()


def test_vec_version_non_empty(tmp_path: Path) -> None:
    """SELECT vec_version() returns a non-empty string."""
    db = _open(tmp_path)
    version = db.conn.execute("SELECT vec_version()").fetchone()[0]
    assert isinstance(version, str)
    assert version.strip() != ""
    db.close()


def test_vec0_tables_created_at_configured_dim(tmp_path: Path) -> None:
    """vec0 tables are created at the configured dim, not a hardcoded literal."""
    dim = 64
    db = _open(tmp_path, dim=dim)
    for tbl in ("memory_frames_vec", "memory_frame_chunks_vec"):
        ddl = db.conn.execute(
            "SELECT sql FROM sqlite_master WHERE name=?", (tbl,)
        ).fetchone()[0]
        assert f"float[{dim}]" in ddl, f"{tbl} DDL missing dim {dim}: {ddl}"
    db.close()


def test_reopen_is_idempotent(tmp_path: Path) -> None:
    """Reopening an existing .mind re-runs schema (IF NOT EXISTS) without error."""
    path = tmp_path / "test.mind"
    db1 = MindDB(path, dim=384)
    db1.close()
    db2 = MindDB(path, dim=384)
    # Still WAL, FK on, vec loaded.
    assert db2.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert db2.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert db2.conn.execute("SELECT vec_version()").fetchone()[0]
    db2.close()


def test_meta_get_set_roundtrip(tmp_path: Path) -> None:
    """MindDB exposes a meta get/set over the ``meta`` table."""
    db = _open(tmp_path)
    assert db.meta_get("missing") is None
    db.meta_set("schema_version", "1")
    assert db.meta_get("schema_version") == "1"
    # Setting again overwrites.
    db.meta_set("schema_version", "2")
    assert db.meta_get("schema_version") == "2"
    db.close()
