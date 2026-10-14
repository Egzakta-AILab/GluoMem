"""Index failure-atomicity tests.

The ``FrameStore._index`` write-through hook embeds everything BEFORE touching
any index row and lands FTS + vec + chunk writes in one transaction. An
embedder failure must therefore leave existing index rows (compaction
re-index) fully intact, and a failed first index must leave no partial rows —
never a half-indexed frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore, FrameType
from gluomem.storage.db import MindDB


class FlakyEmbedder:
    """MockEmbedder wrapper whose embed / embed_batch fail on command."""

    def __init__(self, dim: int) -> None:
        self._inner = MockEmbedder(dim=dim)
        self.fail_embed = False
        self.fail_batch = False

    @property
    def dim(self) -> int:
        return self._inner.dim

    def embed(self, text: str) -> np.ndarray:
        if self.fail_embed:
            raise RuntimeError("embedder down (embed)")
        return self._inner.embed(text)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if self.fail_batch:
            raise RuntimeError("embedder down (embed_batch)")
        return self._inner.embed_batch(texts)


def _store(tmp_path: Path, *, dim: int = 384) -> tuple[MindDB, FrameStore, FlakyEmbedder]:
    db = MindDB(tmp_path / "test.mind", dim=dim)
    emb = FlakyEmbedder(dim)
    return db, FrameStore(db, embedder=emb), emb


def _index_counts(db: MindDB, frame_id: int) -> tuple[int, int, int, int]:
    """(fts, frame_vec, chunks, chunk_vecs) row counts for ``frame_id``."""
    fts = db.conn.execute(
        "SELECT COUNT(*) FROM memory_frames_fts WHERE rowid=?", (frame_id,)
    ).fetchone()[0]
    vec = db.conn.execute(
        "SELECT COUNT(*) FROM memory_frames_vec WHERE rowid=?", (frame_id,)
    ).fetchone()[0]
    chunks = db.conn.execute(
        "SELECT COUNT(*) FROM memory_frame_chunks WHERE frame_id=?", (frame_id,)
    ).fetchone()[0]
    chunk_vecs = db.conn.execute(
        "SELECT COUNT(*) FROM memory_frame_chunks_vec WHERE rowid IN "
        "(SELECT id FROM memory_frame_chunks WHERE frame_id=?)",
        (frame_id,),
    ).fetchone()[0]
    return fts, vec, chunks, chunk_vecs


@pytest.mark.parametrize("failure", ["fail_embed", "fail_batch"])
def test_reindex_failure_leaves_old_index_intact(
    tmp_path: Path, failure: str
) -> None:
    """A failed re-index (compaction path) keeps the previous index rows.

    Old chunk rows must not be deleted before the new embeddings exist —
    an embedder failure mid-re-index would otherwise orphan the frame with
    no chunks at all.
    """
    db, store, emb = _store(tmp_path)
    frame = store.create_i_frame("gop-1", "alpha beta gamma")
    before = _index_counts(db, frame.id)
    assert before == (1, 1, 1, 1)

    setattr(emb, failure, True)
    with pytest.raises(RuntimeError, match="embedder down"):
        store._index(frame)

    assert _index_counts(db, frame.id) == before, (
        "failed re-index must leave the previous FTS/vec/chunk rows untouched"
    )
    # The old FTS row still MATCHes — search stays consistent after the failure.
    fts_ids = [
        r[0]
        for r in db.conn.execute(
            "SELECT rowid FROM memory_frames_fts WHERE memory_frames_fts MATCH 'beta'"
        ).fetchall()
    ]
    assert frame.id in fts_ids
    db.close()


def _frame_rows(db: MindDB) -> int:
    return db.conn.execute("SELECT COUNT(*) FROM memory_frames").fetchone()[0]


def _fts_ids(db: MindDB, term: str) -> list[int]:
    return [
        int(r[0])
        for r in db.conn.execute(
            "SELECT rowid FROM memory_frames_fts WHERE memory_frames_fts MATCH ?",
            (term,),
        ).fetchall()
    ]


@pytest.mark.parametrize("failure", ["fail_embed", "fail_batch"])
@pytest.mark.parametrize("kind", ["I", "P", "B"])
def test_first_write_embed_failure_leaves_nothing_and_retry_succeeds(
    tmp_path: Path, failure: str, kind: str
) -> None:
    """An embedder failure on a new frame leaves no frame row and no index
    rows; the retried write then lands once, fully indexed."""
    db, store, emb = _store(tmp_path)
    base = store.create_i_frame("gop-1", "base alpha")

    def write():
        if kind == "I":
            return store.create_i_frame("gop-1", "delta epsilon zeta")
        if kind == "P":
            return store.create_p_frame("gop-1", "delta epsilon zeta", base.id)
        return store.create_b_frame("gop-1", "delta epsilon zeta", base.id, [base.id])

    setattr(emb, failure, True)
    with pytest.raises(RuntimeError, match="embedder down"):
        write()
    assert _frame_rows(db) == 1
    assert _fts_ids(db, "epsilon") == []

    setattr(emb, failure, False)
    frame = write()
    assert _frame_rows(db) == 2
    assert store.index_state(frame.id).ready
    assert _fts_ids(db, "epsilon") == [frame.id]
    db.close()


def test_sql_failure_mid_index_rolls_back_frame_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SQL failure after the FTS and vec rows are written rolls back the
    frame row too; reopening the file shows nothing, and a retry succeeds."""
    db, store, _ = _store(tmp_path)

    def boom(*_a, **_k):
        raise RuntimeError("chunk write failed")

    monkeypatch.setattr(store, "_index_chunks", boom)
    with pytest.raises(RuntimeError, match="chunk write failed"):
        store.create_i_frame("gop-1", "delta epsilon zeta")
    db.close()

    db = MindDB(tmp_path / "test.mind", dim=384)
    store = FrameStore(db, embedder=FlakyEmbedder(384))
    assert _frame_rows(db) == 0
    assert db.conn.execute("SELECT COUNT(*) FROM memory_frames_vec").fetchone()[0] == 0
    assert _fts_ids(db, "epsilon") == []

    frame = store.create_i_frame("gop-1", "delta epsilon zeta")
    assert store.index_state(frame.id).ready
    db.close()


def _legacy_unindexed(store: FrameStore, content: str, *, frame_type: FrameType = "I",
                      base_id: int | None = None) -> int:
    """Insert a frame row WITHOUT its index, as a pre-fix build could leave it."""
    from gluomem.frames.content_hash import content_hash
    from gluomem.frames.sessions import get_or_create_session

    get_or_create_session(store._conn, "gop-1")
    fid = store._insert(
        gop="gop-1", frame_type=frame_type, base_frame_id=base_id, content=content,
        importance="normal", source="user_stated", trust="normal", event_time=None,
        content_hash=content_hash(content),
    )
    store._conn.commit()
    return fid


def test_dedup_hit_repairs_legacy_unindexed_frame(tmp_path: Path) -> None:
    """Retrying a write whose earlier attempt left an unindexed row repairs
    that row instead of hiding it behind dedup; no duplicate is created."""
    db, store, _ = _store(tmp_path)
    fid = _legacy_unindexed(store, "delta epsilon zeta")
    state = store.index_state(fid)
    assert not state.ready and state.frame_vec == 0 and not state.fts_current

    again = store.create_i_frame("gop-1", "delta epsilon zeta")
    assert again.id == fid
    assert _frame_rows(db) == 1
    assert store.index_state(fid).ready
    assert _fts_ids(db, "epsilon") == [fid]
    db.close()


def test_gate_dedup_hit_repairs_legacy_unindexed_pframe(tmp_path: Path) -> None:
    """The gate's exact-hash dedup (all frame types) repairs too."""
    from gluomem.config.app_config import load_config
    from gluomem.write.gate import InjectionScanner, admit

    cfg = load_config()
    db, store, _ = _store(tmp_path)
    base = store.create_i_frame("gop-1", "base alpha")
    pid = _legacy_unindexed(store, "delta epsilon zeta", frame_type="P", base_id=base.id)

    res = admit(
        "delta epsilon zeta", gop="gop-1", scanner=InjectionScanner(cfg.safety),
        store=store, kg=None, cfg=cfg, frame_type="P", base_id=base.id,
    )
    assert res.deduped and res.frame is not None and res.frame.id == pid
    assert store.index_state(pid).ready
    db.close()


def test_repair_indexes_recovers_existing_file(tmp_path: Path) -> None:
    """Explicit recovery for an existing file: missing and stale projections
    are rebuilt after reopen; ready frames are left alone."""
    db, store, _ = _store(tmp_path)
    ok = store.create_i_frame("gop-1", "alpha ready")
    missing = _legacy_unindexed(store, "beta missing")
    stale = store.create_i_frame("gop-1", "gamma stale")
    db.conn.execute(
        "UPDATE memory_frames SET content='gamma rewritten' WHERE id=?", (stale.id,)
    )
    db.conn.commit()
    db.close()

    db = MindDB(tmp_path / "test.mind", dim=384)
    store = FrameStore(db, embedder=FlakyEmbedder(384))
    assert not store.index_state(stale.id).ready
    assert store.repair_indexes() == [missing, stale.id]
    for fid in (ok.id, missing, stale.id):
        assert store.index_state(fid).ready
    assert _fts_ids(db, "rewritten") == [stale.id]
    assert store.repair_indexes() == []
    db.close()


def test_reindex_success_replaces_index(tmp_path: Path) -> None:
    """A successful re-index still replaces rows (no duplicates, counts stay 1:1)."""
    db, store, _ = _store(tmp_path)
    frame = store.create_i_frame("gop-1", "eta theta iota")
    store._index(frame)  # compaction-style re-index
    fts, vec, chunks, chunk_vecs = _index_counts(db, frame.id)
    assert (fts, vec) == (1, 1)
    assert chunks >= 1 and chunk_vecs == chunks
    db.close()
