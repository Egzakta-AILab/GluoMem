"""Mock embedder tests.

The hash-based mock embedder must be deterministic, offline, and emit
native-dim (384) float32 vectors: ``dim == 384``, embedding the same text
twice returns byte-identical arrays of shape ``(384,)`` with no network.
"""

from __future__ import annotations

import numpy as np

from gluomem.embedding.base import Embedder
from gluomem.embedding.providers import MockEmbedder


def test_mock_dim_is_384() -> None:
    """Default dim == 384 (native, no pad)."""
    emb = MockEmbedder(dim=384)
    assert emb.dim == 384


def test_mock_embed_shape_and_dtype() -> None:
    """embed() returns a float32 ndarray of shape (dim,)."""
    emb = MockEmbedder(dim=384)
    vec = emb.embed("hello world")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert vec.dtype == np.float32


def test_mock_embed_deterministic_byte_identical() -> None:
    """Same text twice → byte-identical arrays."""
    emb = MockEmbedder(dim=384)
    a = emb.embed("remember the milk")
    b = emb.embed("remember the milk")
    assert a.tobytes() == b.tobytes()
    # different text should (overwhelmingly likely) differ
    c = emb.embed("remember the cookies")
    assert a.tobytes() != c.tobytes()


def test_mock_embed_batch_consistent_with_single() -> None:
    """embed_batch returns one vector per text, each identical to embed()."""
    texts = ["alpha", "beta", "gamma"]
    emb = MockEmbedder(dim=384)
    batch = emb.embed_batch(texts)
    assert len(batch) == len(texts)
    for text, vec in zip(texts, batch, strict=True):
        assert vec.shape == (384,)
        assert vec.dtype == np.float32
        assert vec.tobytes() == emb.embed(text).tobytes()


def test_mock_embedder_is_offline() -> None:
    """The mock embedder performs no network I/O (pure hash-based).

    We assert it can be constructed and produce vectors without any socket
    activity by running it in a context where network access is unavailable.
    A deterministic hash-based embedder never opens a socket.
    """
    import socket

    orig_socket = socket.socket
    raised: list[bool] = []

    class _NoNet(socket.socket):
        def __init__(self, *args: object, **kwargs: object) -> None:
            raised.append(True)
            raise OSError("network disabled in test")

    socket.socket = _NoNet  # type: ignore[assignment,method-assign]
    try:
        emb = MockEmbedder(dim=384)
        vec = emb.embed("offline embedding works")
        assert vec.shape == (384,)
        # No socket was opened.
        assert not raised
    finally:
        socket.socket = orig_socket  # type: ignore[assignment,method-assign]


def test_mock_embedder_satisfies_protocol() -> None:
    """MockEmbedder is a structural instance of the Embedder protocol."""
    emb: Embedder = MockEmbedder(dim=384)
    assert isinstance(emb, Embedder)
