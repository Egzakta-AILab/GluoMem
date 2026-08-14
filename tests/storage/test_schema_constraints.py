"""Schema constraint tests.

Verifies the schema enforced by ``schema.sql`` + ``MindDB``:
- ``memory_frames.source`` admits only the 8 defined values (CHECK).
- ``knowledge_relations`` enforces ``UNIQUE(source_id, target_id, relation_type)``.
- ``memory_frames_fts`` is declared with ``tokenize='porter unicode61'``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from gluomem.storage.db import MindDB


def _open(tmp_path: Path) -> MindDB:
    return MindDB(tmp_path / "test.mind", dim=384)


def _insert_frame(db: MindDB, *, source: str) -> None:
    """Raw insert into memory_frames with the given source value.

    Creates the prerequisite session row first so the gop_id FK is satisfied.
    """
    db.conn.execute(
        "INSERT INTO sessions(gop_id) VALUES (?)", ("gop-1",)
    )
    db.conn.execute(
        "INSERT INTO memory_frames(frame_type, gop_id, content, source) "
        "VALUES ('I', 'gop-1', 'hello', ?)",
        (source,),
    )
    db.conn.commit()


def test_source_rejects_external(tmp_path: Path) -> None:
    """source='external' raises IntegrityError."""
    db = _open(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_frame(db, source="external")
    db.close()


def test_source_accepts_team_sync(tmp_path: Path) -> None:
    """source='team_sync' (one of the 8 defined values) succeeds."""
    db = _open(tmp_path)
    _insert_frame(db, source="team_sync")
    row = db.conn.execute(
        "SELECT source FROM memory_frames WHERE gop_id='gop-1'"
    ).fetchone()
    assert row[0] == "team_sync"
    db.close()


def test_relation_triple_unique(tmp_path: Path) -> None:
    """Duplicate (source_id,target_id,relation_type) raises."""
    db = _open(tmp_path)
    c = db.conn
    c.execute("INSERT INTO knowledge_entities(entity_type, name) VALUES ('tool','git')")
    c.execute("INSERT INTO knowledge_entities(entity_type, name) VALUES ('tool','svn')")
    ids = c.execute("SELECT id FROM knowledge_entities ORDER BY id").fetchall()
    src, tgt = ids[0][0], ids[1][0]
    c.execute(
        "INSERT INTO knowledge_relations(source_id, target_id, relation_type) "
        "VALUES (?, ?, 'related')",
        (src, tgt),
    )
    with pytest.raises(sqlite3.IntegrityError):
        c.execute(
            "INSERT INTO knowledge_relations(source_id, target_id, relation_type) "
            "VALUES (?, ?, 'related')",
            (src, tgt),
        )
    db.close()


def test_relation_triple_different_type_allowed(tmp_path: Path) -> None:
    """A different relation_type on the same pair is allowed (UNIQUE is per triple)."""
    db = _open(tmp_path)
    c = db.conn
    c.execute("INSERT INTO knowledge_entities(entity_type, name) VALUES ('tool','git')")
    c.execute("INSERT INTO knowledge_entities(entity_type, name) VALUES ('tool','svn')")
    ids = c.execute("SELECT id FROM knowledge_entities ORDER BY id").fetchall()
    src, tgt = ids[0][0], ids[1][0]
    c.execute(
        "INSERT INTO knowledge_relations(source_id, target_id, relation_type) "
        "VALUES (?, ?, 'related')",
        (src, tgt),
    )
    c.execute(
        "INSERT INTO knowledge_relations(source_id, target_id, relation_type) "
        "VALUES (?, ?, 'depends_on')",
        (src, tgt),
    )
    n = c.execute("SELECT COUNT(*) FROM knowledge_relations").fetchone()[0]
    assert n == 2
    db.close()


def test_fts_tokenizer_is_porter_unicode61(tmp_path: Path) -> None:
    """The fts DDL string contains 'porter unicode61'."""
    db = _open(tmp_path)
    ddl = db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='memory_frames_fts'"
    ).fetchone()[0]
    assert "porter unicode61" in ddl
    db.close()
