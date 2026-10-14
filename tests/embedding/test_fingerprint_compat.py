"""One embedder per ``*.mind`` file.

A file records the fingerprint of the embedder that wrote its vectors. A
same-dim encoder with a different model or preprocessing is rejected before
any vector is read or written, after a cold reopen too. Switching encoders is
an explicit, all-or-nothing ``FrameStore.rebuild_index``. All offline: only
``MockEmbedder`` variants and local test doubles.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from gluomem.config.app_config import load_config
from gluomem.embedding.compat import (
    FINGERPRINT_META_KEY,
    EmbeddingMismatchError,
    adopt_embedder,
    stored_fingerprint,
)
from gluomem.embedding.fingerprint import EmbedderFingerprint, fingerprint_of
from gluomem.embedding.providers import MiniLMEmbedder, MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.search.hybrid import HybridSearch
from gluomem.storage.db import MindDB

DIM = 384


class _OtherEncoder:
    """Same dim as the mock, different model: its vectors are not comparable."""

    def __init__(self, dim: int = DIM) -> None:
        self._inner = MockEmbedder(dim=dim)

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def fingerprint(self) -> EmbedderFingerprint:
        return EmbedderFingerprint(
            implementation="test.other",
            model="other-encoder",
            revision="r1",
            dim=self.dim,
            preprocessing="none",
        )

    def embed(self, text: str) -> np.ndarray:
        return self._inner.embed("other:" + text)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]


class _FailsAfter(MockEmbedder):
    """Normalized mock that raises on the n-th single embed (rebuild failure)."""

    def __init__(self, n: int) -> None:
        super().__init__(dim=DIM, normalize=True)
        self._left = n

    def embed(self, text: str) -> np.ndarray:
        self._left -= 1
        if self._left < 0:
            raise RuntimeError("encoder down")
        return super().embed(text)


def _path(tmp_path: Path) -> Path:
    return tmp_path / "compat.mind"


def _seed(tmp_path: Path, embedder: object | None = None) -> list[int]:
    """Write two frames with ``embedder`` (default mock) and close the file."""
    db = MindDB(_path(tmp_path), dim=DIM)
    store = FrameStore(db, embedder=embedder or MockEmbedder(dim=DIM))  # type: ignore[arg-type]
    ids = [
        store.create_i_frame("g", "alpha deploy window friday").id,
        store.create_i_frame("g", "beta rollback plan").id,
    ]
    db.close()
    return ids


def _vec_bytes(db: MindDB) -> dict[int, bytes]:
    return {
        int(r[0]): bytes(r[1])
        for r in db.conn.execute("SELECT rowid, embedding FROM memory_frames_vec")
    }


# --- rejection ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("embedder", "fields"),
    [
        (MockEmbedder(dim=DIM, normalize=True), ["preprocessing"]),
        (_OtherEncoder(), ["implementation", "model", "revision", "preprocessing"]),
    ],
    ids=["changed-preprocessing", "same-dim-other-model"],
)
def test_mismatched_embedder_rejected_after_cold_reopen(
    tmp_path: Path, embedder: object, fields: list[str]
) -> None:
    _seed(tmp_path)
    db = MindDB(_path(tmp_path), dim=DIM)
    before = _vec_bytes(db)
    with pytest.raises(EmbeddingMismatchError) as exc:
        FrameStore(db, embedder=embedder)  # type: ignore[arg-type]
    assert exc.value.fields == fields
    assert "rebuild_index" in str(exc.value)
    # Rejected before any write: vectors and the recorded fingerprint are untouched.
    assert _vec_bytes(db) == before
    assert stored_fingerprint(db) == fingerprint_of(MockEmbedder(dim=DIM))
    db.close()


def test_query_embedder_mismatch_rejected_before_search(tmp_path: Path) -> None:
    _seed(tmp_path)
    db = MindDB(_path(tmp_path), dim=DIM)
    store = FrameStore(db, embedder=MockEmbedder(dim=DIM))
    with pytest.raises(EmbeddingMismatchError):
        HybridSearch(db, store, load_config(), embedder=_OtherEncoder())  # type: ignore[arg-type]
    # Default query embedder is the store's own.
    search = HybridSearch(db, store, load_config().with_overrides({"search.rerank.enabled": False}))
    assert search.search("alpha").hits
    db.close()


def test_matching_embedder_binds_after_cold_reopen(tmp_path: Path) -> None:
    ids = _seed(tmp_path)
    db = MindDB(_path(tmp_path), dim=DIM)
    store = FrameStore(db, embedder=MockEmbedder(dim=DIM))
    assert store.fingerprint == fingerprint_of(MockEmbedder(dim=DIM))
    assert store.get_frame(ids[0]).content == "alpha deploy window friday"
    db.close()


def test_first_bind_on_empty_file_records_fingerprint(tmp_path: Path) -> None:
    db = MindDB(_path(tmp_path), dim=DIM)
    assert stored_fingerprint(db) is None
    FrameStore(db, embedder=MockEmbedder(dim=DIM))
    assert stored_fingerprint(db) == fingerprint_of(MockEmbedder(dim=DIM))
    # Once recorded, even an empty file refuses another encoder.
    with pytest.raises(EmbeddingMismatchError):
        FrameStore(db, embedder=_OtherEncoder())  # type: ignore[arg-type]
    db.close()


# --- files written before fingerprints --------------------------------------


def _strip_fingerprint(tmp_path: Path) -> None:
    conn = sqlite3.connect(_path(tmp_path))
    conn.execute("DELETE FROM meta WHERE key=?", (FINGERPRINT_META_KEY,))
    conn.commit()
    conn.close()


def test_legacy_file_with_vectors_needs_explicit_adopt(tmp_path: Path) -> None:
    _seed(tmp_path)
    _strip_fingerprint(tmp_path)
    db = MindDB(_path(tmp_path), dim=DIM)
    with pytest.raises(EmbeddingMismatchError) as exc:
        FrameStore(db, embedder=MockEmbedder(dim=DIM))
    assert exc.value.stored is None
    assert "adopt_embedder" in str(exc.value)

    adopt_embedder(db, MockEmbedder(dim=DIM))
    store = FrameStore(db, embedder=MockEmbedder(dim=DIM))
    cfg = load_config().with_overrides({"search.rerank.enabled": False})
    assert HybridSearch(db, store, cfg).search("alpha").hits
    # adopt never overrides a different recorded fingerprint.
    with pytest.raises(EmbeddingMismatchError):
        adopt_embedder(db, _OtherEncoder())
    db.close()


# --- explicit rebuild --------------------------------------------------------


def test_rebuild_switches_encoder_and_survives_reopen(tmp_path: Path) -> None:
    ids = _seed(tmp_path)
    new = MockEmbedder(dim=DIM, normalize=True)
    db = MindDB(_path(tmp_path), dim=DIM)
    store = FrameStore.rebuild_index(db, new)
    assert store.fingerprint == fingerprint_of(new)
    vecs = _vec_bytes(db)
    for fid in ids:
        expected = new.embed(store.get_frame(fid).content).tobytes()
        assert vecs[fid] == expected
    assert all(store.index_state(fid).ready for fid in ids)
    db.close()

    db = MindDB(_path(tmp_path), dim=DIM)
    with pytest.raises(EmbeddingMismatchError):
        FrameStore(db, embedder=MockEmbedder(dim=DIM))
    store = FrameStore(db, embedder=MockEmbedder(dim=DIM, normalize=True))
    cfg = load_config().with_overrides({"search.rerank.enabled": False})
    assert {h.frame.id for h in HybridSearch(db, store, cfg).search("alpha").hits} >= {ids[0]}
    db.close()


def test_rebuild_adopts_legacy_file(tmp_path: Path) -> None:
    _seed(tmp_path)
    _strip_fingerprint(tmp_path)
    db = MindDB(_path(tmp_path), dim=DIM)
    store = FrameStore.rebuild_index(db, MockEmbedder(dim=DIM))
    assert stored_fingerprint(db) == store.fingerprint
    db.close()


def test_failed_rebuild_leaves_file_unchanged(tmp_path: Path) -> None:
    _seed(tmp_path)
    db = MindDB(_path(tmp_path), dim=DIM)
    before = _vec_bytes(db)
    chunks_before = db.conn.execute("SELECT COUNT(*) FROM memory_frame_chunks").fetchone()
    with pytest.raises(RuntimeError, match="encoder down"):
        FrameStore.rebuild_index(db, _FailsAfter(1))
    assert _vec_bytes(db) == before
    assert db.conn.execute("SELECT COUNT(*) FROM memory_frame_chunks").fetchone() == chunks_before
    assert stored_fingerprint(db) == fingerprint_of(MockEmbedder(dim=DIM))
    FrameStore(db, embedder=MockEmbedder(dim=DIM))  # still bound to the old encoder
    db.close()


# --- fingerprint contents ----------------------------------------------------


def test_fingerprint_covers_model_revision_dim_preprocessing_schema() -> None:
    a = MiniLMEmbedder("sentence-transformers/all-MiniLM-L6-v2", dim=DIM, revision="abc")
    b = MiniLMEmbedder("sentence-transformers/all-MiniLM-L6-v2", dim=DIM, revision="def")
    fa = fingerprint_of(a)
    assert fa.implementation == "sentence-transformers"
    assert fa.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert fa.dim == DIM and fa.index_schema.startswith("sqlite-vec/")
    assert fa.differences(fingerprint_of(b)) == ["revision"]
    assert a._model is None, "fingerprinting must not load the model"
    assert EmbedderFingerprint.from_json(fa.to_json()) == fa
    assert fa.digest != fingerprint_of(b).digest


def test_undeclared_embedder_is_identified_by_class() -> None:
    class Plain:
        dim = DIM

        def embed(self, text: str) -> np.ndarray:
            return np.zeros(DIM, dtype=np.float32)

        def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
            return [self.embed(t) for t in texts]

    fp = fingerprint_of(Plain())
    assert fp.implementation.endswith("Plain")
    assert fp != fingerprint_of(MockEmbedder(dim=DIM))
