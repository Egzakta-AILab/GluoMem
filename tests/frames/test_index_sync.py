"""Index-sync tests.

Writing any frame must synchronously populate its FTS row, whole-frame vector
row, and >=1 chunk row via the ``FrameStore._index`` write-through hook. The
event_time / created_at split is also verified here through the index-enabled
store (the frame-semantics test in ``test_framestore.py`` covers the bare
column split; this re-asserts it with indexing on).
"""

from __future__ import annotations

from pathlib import Path

from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.storage.db import MindDB


def _store(tmp_path: Path, *, dim: int = 384) -> tuple[MindDB, FrameStore]:
    db = MindDB(tmp_path / "test.mind", dim=dim)
    emb = MockEmbedder(dim=dim)
    return db, FrameStore(db, embedder=emb)


# --- index sync -------------------------------------------------------------


def test_index_sync_fts_vec_chunk(tmp_path: Path) -> None:
    """Writing 'alpha beta gamma' syncs FTS + vec + chunk rows.

    After the write: the FTS table MATCHes 'beta' and returns the frame id; the
    vec table has exactly 1 row for the frame id; the chunk table has >=1 row
    for the frame id.
    """
    db, store = _store(tmp_path)
    frame = store.create_i_frame("gop-1", "alpha beta gamma")

    # FTS: MATCH 'beta' returns the frame id.
    fts_ids = [
        r[0]
        for r in db.conn.execute(
            "SELECT rowid FROM memory_frames_fts WHERE memory_frames_fts MATCH 'beta'"
        ).fetchall()
    ]
    assert frame.id in fts_ids, f"FTS MATCH 'beta' did not return frame id {frame.id}"

    # Whole-frame vector: exactly 1 row for the frame id.
    vec_count = db.conn.execute(
        "SELECT COUNT(*) FROM memory_frames_vec WHERE rowid=?", (frame.id,)
    ).fetchone()[0]
    assert vec_count == 1, f"expected 1 vec row for frame {frame.id}, got {vec_count}"

    # Chunks: at least 1 row for the frame id.
    chunk_count = db.conn.execute(
        "SELECT COUNT(*) FROM memory_frame_chunks WHERE frame_id=?", (frame.id,)
    ).fetchone()[0]
    assert chunk_count >= 1, (
        f"expected >=1 chunk row for frame {frame.id}, got {chunk_count}"
    )

    # The chunk vec table also has a row per chunk (chunk rowid == chunk.id).
    chunk_vec_count = db.conn.execute(
        "SELECT COUNT(*) FROM memory_frame_chunks_vec WHERE rowid IN "
        "(SELECT id FROM memory_frame_chunks WHERE frame_id=?)",
        (frame.id,),
    ).fetchone()[0]
    assert chunk_vec_count == chunk_count
    db.close()


def test_index_sync_p_frame(tmp_path: Path) -> None:
    """A P-frame write also syncs FTS + vec + chunk rows (every write funnels through _index)."""
    db, store = _store(tmp_path)
    i = store.create_i_frame("gop-1", "base fact")
    p = store.create_p_frame("gop-1", "refinement delta", base_id=i.id)

    fts_ids = [
        r[0]
        for r in db.conn.execute(
            "SELECT rowid FROM memory_frames_fts WHERE memory_frames_fts MATCH 'refinement'"
        ).fetchall()
    ]
    assert p.id in fts_ids
    vec_count = db.conn.execute(
        "SELECT COUNT(*) FROM memory_frames_vec WHERE rowid=?", (p.id,)
    ).fetchone()[0]
    assert vec_count == 1
    chunk_count = db.conn.execute(
        "SELECT COUNT(*) FROM memory_frame_chunks WHERE frame_id=?", (p.id,)
    ).fetchone()[0]
    assert chunk_count >= 1
    db.close()


def test_index_sync_b_frame_indexes_description(tmp_path: Path) -> None:
    """A B-frame indexes its human-readable description, not the raw JSON payload."""
    db, store = _store(tmp_path)
    i = store.create_i_frame("gop-1", "the sky is green")
    b = store.create_b_frame(
        "gop-1",
        description="the sky is actually blue",
        base_id=i.id,
        references=[i.id],
    )
    # FTS MATCH on the description text returns the B-frame id.
    fts_ids = [
        r[0]
        for r in db.conn.execute(
            "SELECT rowid FROM memory_frames_fts WHERE memory_frames_fts MATCH 'blue'"
        ).fetchall()
    ]
    assert b.id in fts_ids
    # The raw JSON should NOT be the indexed text (no MATCH on the 'references' key).
    json_ids = [
        r[0]
        for r in db.conn.execute(
            "SELECT rowid FROM memory_frames_fts WHERE memory_frames_fts MATCH 'references'"
        ).fetchall()
    ]
    assert b.id not in json_ids
    db.close()


def test_index_sync_dedup_does_not_double_index(tmp_path: Path) -> None:
    """Dedup returns the existing frame and does not create a second vec/FTS/chunk row."""
    db, store = _store(tmp_path)
    f1 = store.create_i_frame("gop-1", "remember the milk")
    f2 = store.create_i_frame("gop-1", "remember the milk")
    assert f1.id == f2.id
    vec_count = db.conn.execute(
        "SELECT COUNT(*) FROM memory_frames_vec WHERE rowid=?", (f1.id,)
    ).fetchone()[0]
    assert vec_count == 1
    chunk_count = db.conn.execute(
        "SELECT COUNT(*) FROM memory_frame_chunks WHERE frame_id=?", (f1.id,)
    ).fetchone()[0]
    assert chunk_count == 1
    db.close()


# --- event-time split -------------------------------------------------------


def test_event_time_split_with_indexing(tmp_path: Path) -> None:
    """A past event_time reads back while created_at differs.

    With index sync enabled, a frame written with an explicit past event_time
    still stores event_time distinctly from the ingest-time created_at.
    """
    db, store = _store(tmp_path)
    frame = store.create_i_frame(
        "gop-1", "meeting happened", event_time="2020-01-01T00:00:00"
    )
    back = store.get_frame(frame.id)
    assert back.event_time == "2020-01-01T00:00:00"
    assert back.created_at != back.event_time
    # created_at is the ingest 'now' (not the past event_time).
    assert back.created_at.startswith("20")
    db.close()
