"""MindDB pragmas, schema constraints, and metadata roundtrip tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from gluomem.storage.db import MindDB


def test_pragmas_vec_extension_and_idempotent_reopen(tmp_path: Path) -> None:
    path = tmp_path / "test.mind"
    db = MindDB(path, dim=64)
    assert db.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert db.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert db.conn.execute("SELECT vec_version()").fetchone()[0].strip() != ""
    for tbl in ("memory_frames_vec", "memory_frame_chunks_vec"):
        ddl = db.conn.execute(
            "SELECT sql FROM sqlite_master WHERE name=?", (tbl,)
        ).fetchone()[0]
        assert "float[64]" in ddl
    db.close()

    db2 = MindDB(path, dim=64)
    assert db2.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert db2.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    db2.close()


def test_meta_get_set_roundtrip(mind_db: MindDB) -> None:
    assert mind_db.meta_get("missing") is None
    mind_db.meta_set("schema_version", "1")
    assert mind_db.meta_get("schema_version") == "1"
    mind_db.meta_set("schema_version", "2")
    assert mind_db.meta_get("schema_version") == "2"


def test_schema_constraints_and_fts_tokenizer(mind_db: MindDB) -> None:
    c = mind_db.conn
    c.execute("INSERT INTO sessions(gop_id) VALUES (?)", ("gop-1",))
    c.execute(
        "INSERT INTO memory_frames(frame_type, gop_id, content, source) "
        "VALUES ('I', 'gop-1', 'hello', ?)",
        ("team_sync",),
    )
    assert (
        c.execute("SELECT source FROM memory_frames WHERE gop_id='gop-1'").fetchone()[0]
        == "team_sync"
    )
    with pytest.raises(sqlite3.IntegrityError):
        c.execute(
            "INSERT INTO memory_frames(frame_type, gop_id, content, source) "
            "VALUES ('I', 'gop-1', 'bad', ?)",
            ("external",),
        )

    c.execute("INSERT INTO knowledge_entities(entity_type, name) VALUES ('tool','git')")
    c.execute("INSERT INTO knowledge_entities(entity_type, name) VALUES ('tool','svn')")
    ids = [r[0] for r in c.execute("SELECT id FROM knowledge_entities ORDER BY id")]
    c.execute(
        "INSERT INTO knowledge_relations(source_id, target_id, relation_type) "
        "VALUES (?, ?, 'related')",
        (ids[0], ids[1]),
    )
    c.execute(
        "INSERT INTO knowledge_relations(source_id, target_id, relation_type) "
        "VALUES (?, ?, 'depends_on')",
        (ids[0], ids[1]),
    )
    assert c.execute("SELECT COUNT(*) FROM knowledge_relations").fetchone()[0] == 2
    with pytest.raises(sqlite3.IntegrityError):
        c.execute(
            "INSERT INTO knowledge_relations(source_id, target_id, relation_type) "
            "VALUES (?, ?, 'related')",
            (ids[0], ids[1]),
        )

    ddl = c.execute(
        "SELECT sql FROM sqlite_master WHERE name='memory_frames_fts'"
    ).fetchone()[0]
    assert "porter unicode61" in ddl
