"""Mock-banner tests.

Selecting the mock embedder as a FALLBACK emits a LOUD all-caps ``MOCK`` banner
to stderr AND ``logging.warning``, and the mock still returns native-dim (384)
float32 vectors. The factory is ``make_embedder(cfg)``: when the configured
model is unavailable or ``embedding.model == 'mock'`` it returns the mock and
prints the banner.

No test here downloads a model — the mock path is fully offline.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from gluomem.config.app_config import AppConfig
from gluomem.embedding.base import Embedder
from gluomem.embedding.providers import MockEmbedder, make_embedder


def test_make_embedder_mock_explicit_emits_banner(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """embedding.model=='mock' selects mock + emits an all-caps MOCK banner."""
    cfg = AppConfig().with_overrides({"embedding.model": "mock"})
    assert cfg.embedding.model == "mock"

    with caplog.at_level(logging.WARNING, logger="gluomem.embedding"):
        emb = make_embedder(cfg)

    # It is the mock embedder, returning native-dim vectors.
    assert isinstance(emb, MockEmbedder)
    assert emb.dim == cfg.storage.vector_dim == 384
    vec = emb.embed("hello world")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert vec.dtype == np.float32

    # The banner is loud and all-caps MOCK in BOTH stderr and the log.
    err = capsys.readouterr().err
    assert "MOCK" in err, f"stderr did not contain the all-caps MOCK banner: {err!r}"
    log_text = "\n".join(r.getMessage() for r in caplog.records)
    assert "MOCK" in log_text, (
        f"logging.warning did not contain the all-caps MOCK banner: {log_text!r}"
    )


def test_make_embedder_mock_returns_native_dim_vectors() -> None:
    """The fallback mock returns native-dim (384) vectors, deterministic."""
    cfg = AppConfig().with_overrides({"embedding.model": "mock"})
    emb = make_embedder(cfg)
    a = emb.embed("deterministic vector")
    b = emb.embed("deterministic vector")
    assert a.shape == (384,)
    assert a.tobytes() == b.tobytes(), "mock fallback must stay deterministic"


def test_make_embedder_mock_satisfies_protocol() -> None:
    """The fallback mock is a structural Embedder."""
    cfg = AppConfig().with_overrides({"embedding.model": "mock"})
    emb: Embedder = make_embedder(cfg)
    assert isinstance(emb, Embedder)


def test_mock_banner_only_on_fallback_not_direct_construction(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Direct MockEmbedder construction (not via the factory) does NOT emit the banner.

    The banner is tied to FALLBACK selection — only ``make_embedder`` choosing
    the mock because the configured model is unavailable/== 'mock' emits it.
    Constructing a MockEmbedder directly (e.g. for tests) stays quiet.
    """
    with caplog.at_level(logging.WARNING, logger="gluomem.embedding"):
        emb = MockEmbedder(dim=384)
    err = capsys.readouterr().err
    assert "MOCK" not in err, "direct construction must not emit the fallback banner"
    log_text = "\n".join(r.getMessage() for r in caplog.records)
    assert "MOCK" not in log_text
    # Still works as an embedder.
    assert emb.embed("x").shape == (384,)
