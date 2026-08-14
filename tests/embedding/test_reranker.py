"""Reranker tests — protocol seam + stub.

The :class:`Reranker` protocol and :class:`CrossEncoderReranker` impl exist
behind the protocol. The cross-encoder lazy-loads on first ``score`` so
construction does NOT download. The gate uses a deterministic
:class:`StubReranker`; any real cross-encoder test is ``@pytest.mark.slow`` and
excluded from the gate.
"""

from __future__ import annotations

import pytest

from gluomem.embedding.reranker import (
    CrossEncoderReranker,
    Reranker,
    StubReranker,
)


def test_stub_reranker_satisfies_protocol() -> None:
    """StubReranker is a structural Reranker."""
    r: Reranker = StubReranker(logit=2.5)
    assert isinstance(r, Reranker)


def test_stub_reranker_returns_fixed_logit_per_doc() -> None:
    """StubReranker.score returns one fixed logit per doc, aligned with docs."""
    r = StubReranker(logit=1.25)
    out = r.score("query", ["d1", "d2", "d3"])
    assert out == [1.25, 1.25, 1.25]
    assert len(out) == 3


def test_crossencoder_construction_does_not_download() -> None:
    """Constructing CrossEncoderReranker is lazy — no model load on __init__."""
    r = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    # The model is not loaded until the first score() call.
    assert r._model is None
    assert isinstance(r, Reranker)


@pytest.mark.slow
def test_crossencoder_real_score() -> None:
    """slow — loads the real cross-encoder (downloads on first call).

    Excluded from the gate (``-m 'not slow'``). Only run opt-in.
    """
    r = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    logits = r.score("what is memory", ["a memory layer", "unrelated text"])
    assert len(logits) == 2
    assert all(isinstance(x, float) for x in logits)
