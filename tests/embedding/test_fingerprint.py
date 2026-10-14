"""Embedder fingerprint compatibility, adoption, and atomic index rebuild tests."""

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
)
from gluomem.embedding.fingerprint import EmbedderFingerprint, fingerprint_of
from gluomem.embedding.providers import MiniLMEmbedder, MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.search.hybrid import HybridSearch
from gluomem.storage.db import MindDB

DIM = 384


class _OtherEncoder:
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
    def __init__(self, n: int) -> None:
        super().__init__(dim=DIM, normalize=True)
        self._left = n

    def embed(self, text: str) -> np.ndarray:
        self._left -= 1
        if self._left < 0:
            raise RuntimeError("encoder down")
        return super().embed(text)


def _seed(path: Path, embedder: object | None = None) -> list[int]:
    db = MindDB(path, dim=DIM)
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


@pytest.mark.parametrize(
    ("embedder", "fields"),
    [
        (MockEmbedder(dim=DIM, normalize=True), ["preprocessing"]),
        (_OtherEncoder(), ["implementation", "model", "revision", "preprocessing"]),
    ],
)
def test_mismatched_embedder_rejected_after_reopen(
    tmp_path: Path, embedder: object, fields: list[str]
) -> None:
    path = tmp_path / "compat.mind"
    _seed(path)
    db = MindDB(path, dim=DIM)
    before = _vec_bytes(db)
    with pytest.raises(EmbeddingMismatchError) as exc:
        FrameStore(db, embedder=embedder)  # type: ignore[arg-type]
    assert exc.value.fields == fields
    assert _vec_bytes(db) == before
    db.close()


def test_query_embedder_mismatch_and_legacy_adopt(tmp_path: Path) -> None:
    path = tmp_path / "compat.mind"
    _seed(path)
    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM meta WHERE key=?", (FINGERPRINT_META_KEY,))
    conn.commit()
    conn.close()

    db = MindDB(path, dim=DIM)
    with pytest.raises(EmbeddingMismatchError, match="adopt_embedder"):
        FrameStore(db, embedder=MockEmbedder(dim=DIM))
    adopt_embedder(db, MockEmbedder(dim=DIM))
    store = FrameStore(db, embedder=MockEmbedder(dim=DIM))
    with pytest.raises(EmbeddingMismatchError):
        HybridSearch(db, store, load_config(), embedder=_OtherEncoder())  # type: ignore[arg-type]
    cfg = load_config().with_overrides({"search.rerank.enabled": False})
    assert HybridSearch(db, store, cfg).search("alpha").hits
    db.close()


def test_rebuild_index_and_rollback_on_failure(tmp_path: Path) -> None:
    path = tmp_path / "compat.mind"
    ids = _seed(path)
    db = MindDB(path, dim=DIM)
    before = _vec_bytes(db)
    with pytest.raises(RuntimeError, match="encoder down"):
        FrameStore.rebuild_index(db, _FailsAfter(1))
    assert _vec_bytes(db) == before

    new = MockEmbedder(dim=DIM, normalize=True)
    store = FrameStore.rebuild_index(db, new)
    assert store.fingerprint == fingerprint_of(new)
    assert all(store.index_state(fid).ready for fid in ids)
    db.close()


def test_fingerprint_value_and_undeclared_embedder() -> None:
    a = MiniLMEmbedder("sentence-transformers/all-MiniLM-L6-v2", dim=DIM, revision="abc")
    b = MiniLMEmbedder("sentence-transformers/all-MiniLM-L6-v2", dim=DIM, revision="def")
    fa = fingerprint_of(a)
    assert fa.differences(fingerprint_of(b)) == ["revision"]
    assert EmbedderFingerprint.from_json(fa.to_json()) == fa

    class Plain:
        dim = DIM

        def embed(self, text: str) -> np.ndarray:
            return np.zeros(DIM, dtype=np.float32)

        def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
            return [self.embed(t) for t in texts]

    assert fingerprint_of(Plain()).implementation.endswith("Plain")
