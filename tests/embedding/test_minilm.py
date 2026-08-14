"""MiniLM embedder tests.

The real :class:`MiniLMEmbedder` lazy-loads ``sentence-transformers`` on first
``embed`` call so construction does NOT download. The non-slow test only
asserts the lazy-load seam (no model load on __init__). The real-model test is
``@pytest.mark.slow`` and excluded from the gate.
"""

from __future__ import annotations

import pytest

from gluomem.embedding.base import Embedder
from gluomem.embedding.providers import MiniLMEmbedder


def test_minilm_construction_does_not_download() -> None:
    """Constructing MiniLMEmbedder is lazy — no model load on __init__."""
    emb = MiniLMEmbedder("sentence-transformers/all-MiniLM-L6-v2", dim=384)
    assert emb.dim == 384
    # The model is not loaded until the first embed() call.
    assert emb._model is None
    assert isinstance(emb, Embedder)


@pytest.mark.slow
def test_minilm_real_embed_native_dim() -> None:
    """slow — loads the real MiniLM model (downloads on first call).

    Excluded from the gate (``-m 'not slow'``). Only run opt-in.
    """
    import numpy as np

    emb = MiniLMEmbedder("sentence-transformers/all-MiniLM-L6-v2", dim=384)
    vec = emb.embed("hello world")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert vec.dtype == np.float32
