"""Mock-banner tests.

When ``make_embedder`` builds the mock (``embedding.model == 'mock'`` with
``components.allow_fake``) it emits a LOUD all-caps ``MOCK`` banner to stderr
AND ``logging.warning``, and the mock returns native-dim (384) float32
vectors. The factory never selects the mock as a fallback
(``tests/embedding/test_provider_readiness.py``).

No test here downloads a model — the mock path is fully offline.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from gluomem.config.app_config import AppConfig
from gluomem.embedding.base import Embedder
from gluomem.embedding.providers import MockEmbedder, make_embedder

_MOCK = {"embedding.model": "mock", "components.allow_fake": True}


def test_make_embedder_mock_explicit_emits_banner(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """embedding.model=='mock' selects mock + emits an all-caps MOCK banner."""
    cfg = AppConfig().with_overrides(_MOCK)
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
    """The factory-built mock returns native-dim (384) vectors, deterministic."""
    cfg = AppConfig().with_overrides(_MOCK)
    emb = make_embedder(cfg)
    a = emb.embed("deterministic vector")
    b = emb.embed("deterministic vector")
    assert a.shape == (384,)
    assert a.tobytes() == b.tobytes(), "factory mock must stay deterministic"


def test_make_embedder_mock_satisfies_protocol() -> None:
    """The factory-built mock is a structural Embedder."""
    cfg = AppConfig().with_overrides(_MOCK)
    emb: Embedder = make_embedder(cfg)
    assert isinstance(emb, Embedder)


def test_mock_banner_only_from_factory_not_direct_construction(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Direct MockEmbedder construction (not via the factory) does NOT emit the banner.

    The banner is tied to factory selection. Constructing a MockEmbedder
    directly (e.g. for tests) stays quiet.
    """
    with caplog.at_level(logging.WARNING, logger="gluomem.embedding"):
        emb = MockEmbedder(dim=384)
    err = capsys.readouterr().err
    assert "MOCK" not in err, "direct construction must not emit the factory banner"
    log_text = "\n".join(r.getMessage() for r in caplog.records)
    assert "MOCK" not in log_text
    # Still works as an embedder.
    assert emb.embed("x").shape == (384,)
