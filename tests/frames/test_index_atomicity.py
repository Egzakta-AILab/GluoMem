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
from gluomem.frames.store import FrameStore
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


@pytest.mark.parametrize("failure", ["fail_embed", "fail_batch"])
def test_first_index_failure_leaves_no_partial_rows(
    tmp_path: Path, failure: str
) -> None:
    """An embedder failure during the first index writes NO index rows.

    The frame row itself is committed before ``_index`` runs (documented), but
    the index must be all-or-nothing: no FTS row without a vec row, etc.
    """
    db, store, emb = _store(tmp_path)
    setattr(emb, failure, True)
    with pytest.raises(RuntimeError, match="embedder down"):
        store.create_i_frame("gop-1", "delta epsilon zeta")

    row = db.conn.execute(
        "SELECT id FROM memory_frames WHERE gop_id='gop-1'"
    ).fetchone()
    assert row is not None, "the frame row itself is committed before _index"
    assert _index_counts(db, int(row[0])) == (0, 0, 0, 0), (
        "a failed first index must leave no partial FTS/vec/chunk rows"
    )
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
