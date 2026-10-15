"""Embedding and reranker provider tests."""

from __future__ import annotations

import logging
import socket

import numpy as np
import pytest

from gluomem.config.app_config import AppConfig
from gluomem.embedding.base import Embedder
from gluomem.embedding.providers import MiniLMEmbedder, MockEmbedder, make_embedder
from gluomem.embedding.reranker import CrossEncoderReranker, Reranker, StubReranker

_MOCK = {"embedding.model": "mock", "components.allow_fake": True}


def test_mock_embedder_deterministic_and_offline() -> None:
    emb: Embedder = MockEmbedder(dim=384)
    assert isinstance(emb, Embedder)
    assert emb.dim == 384

    a = emb.embed("remember the milk")
    b = emb.embed("remember the milk")
    c = emb.embed("remember the cookies")
    assert a.shape == (384,) and a.dtype == np.float32
    assert a.tobytes() == b.tobytes()
    assert a.tobytes() != c.tobytes()

    batch = emb.embed_batch(["alpha", "beta"])
    assert len(batch) == 2
    assert batch[0].tobytes() == emb.embed("alpha").tobytes()

    orig_socket = socket.socket

    class _NoNet(socket.socket):
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise OSError("network disabled in test")

    socket.socket = _NoNet  # type: ignore[assignment,method-assign]
    try:
        assert emb.embed("offline embedding works").shape == (384,)
    finally:
        socket.socket = orig_socket  # type: ignore[assignment,method-assign]


def test_make_embedder_mock_banner_only_from_factory(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    with caplog.at_level(logging.WARNING, logger="gluomem.embedding"):
        direct = MockEmbedder(dim=384)
    assert "MOCK" not in capsys.readouterr().err
    assert not caplog.records
    assert direct.embed("x").shape == (384,)

    cfg = AppConfig().with_overrides(_MOCK)
    with caplog.at_level(logging.WARNING, logger="gluomem.embedding"):
        emb = make_embedder(cfg)
    assert isinstance(emb, MockEmbedder)
    assert "MOCK" in capsys.readouterr().err
    assert any("MOCK" in r.getMessage() for r in caplog.records)


def test_lazy_construction_of_minilm_and_reranker() -> None:
    emb = MiniLMEmbedder("sentence-transformers/all-MiniLM-L6-v2", dim=384)
    assert emb.dim == 384 and emb._model is None and isinstance(emb, Embedder)

    r: Reranker = StubReranker(logit=1.25)
    assert isinstance(r, Reranker)
    assert r.score("query", ["d1", "d2"]) == [1.25, 1.25]

    ce = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert ce._model is None and isinstance(ce, Reranker)


def test_multilingual_e5_small_registry_and_prefix_embedding() -> None:
    cfg = AppConfig().with_overrides({"embedding.model": "multilingual_e5_small"})
    entry = cfg.models["multilingual_e5_small"]
    assert entry.provider == "inprocess"
    assert entry.provider_model == "intfloat/multilingual-e5-small"
    assert entry.dim == 384
    assert entry.query_prefix == "query: "
    assert entry.passage_prefix == "passage: "

    emb = MockEmbedder(
        dim=384,
        query_prefix=entry.query_prefix,
        passage_prefix=entry.passage_prefix,
    )
    plain = MockEmbedder(dim=384)
    assert emb.fingerprint != plain.fingerprint
    assert "query_prefix='query: '" in emb.fingerprint.preprocessing
    assert "passage_prefix='passage: '" in emb.fingerprint.preprocessing

    # embed() uses passage_prefix, embed_query() uses query_prefix
    assert np.array_equal(emb.embed("hello"), plain.embed("passage: hello"))
    assert np.array_equal(emb.embed_query("hello"), plain.embed("query: hello"))
    assert not np.array_equal(emb.embed("hello"), emb.embed_query("hello"))
    batch = emb.embed_batch(["alpha", "beta"])
    assert np.array_equal(batch[0], plain.embed("passage: alpha"))
    assert np.array_equal(batch[1], plain.embed("passage: beta"))


def test_hybrid_search_uses_embed_query_for_prefixed_embedder(
    tmp_path: pytest.TempPathFactory,
) -> None:
    from pathlib import Path

    from gluomem.frames.store import FrameStore
    from gluomem.search.hybrid import HybridSearch
    from gluomem.storage.db import MindDB

    base_dir = Path(str(tmp_path))
    db = MindDB(base_dir / "e5.mind", dim=384)
    calls: list[tuple[str, str]] = []

    class _TrackingPrefixedEmbedder(MockEmbedder):
        def embed(self, text: str) -> np.ndarray:
            calls.append(("passage", text))
            return super().embed(text)

        def embed_query(self, text: str) -> np.ndarray:
            calls.append(("query", text))
            return super().embed_query(text)

    emb = _TrackingPrefixedEmbedder(
        dim=384,
        query_prefix="query: ",
        passage_prefix="passage: ",
    )
    cfg = AppConfig().with_overrides({"search.rerank.enabled": False})
    store = FrameStore(db, embedder=emb, chunking=cfg.chunking)
    store.create_i_frame("g1", "Server backup runs at midnight UTC.")
    hs = HybridSearch(db, store, cfg, embedder=emb)
    res = hs.search("When does the backup run?", limit=3)
    assert res.hits
    assert ("query", "When does the backup run?") in calls
    db.close()


@pytest.mark.slow
def test_real_minilm_and_crossencoder() -> None:
    emb = MiniLMEmbedder("sentence-transformers/all-MiniLM-L6-v2", dim=384)
    vec = emb.embed("hello world")
    assert vec.shape == (384,) and vec.dtype == np.float32

    ce = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    logits = ce.score("what is memory", ["a memory layer", "unrelated text"])
    assert len(logits) == 2 and all(isinstance(x, float) for x in logits)
