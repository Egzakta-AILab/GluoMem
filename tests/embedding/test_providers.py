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


@pytest.mark.slow
def test_real_minilm_and_crossencoder() -> None:
    emb = MiniLMEmbedder("sentence-transformers/all-MiniLM-L6-v2", dim=384)
    vec = emb.embed("hello world")
    assert vec.shape == (384,) and vec.dtype == np.float32

    ce = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    logits = ce.score("what is memory", ["a memory layer", "unrelated text"])
    assert len(logits) == 2 and all(isinstance(x, float) for x in logits)
