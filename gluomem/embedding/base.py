"""Structural-typing protocol for text embedders."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from gluomem.storage.backend import assert_dim

__all__ = ["Embedder", "assert_dim", "embed_query_text"]


@runtime_checkable
class Embedder(Protocol):
    """Turn text into a fixed-dimension ``float32`` vector."""

    @property
    def dim(self) -> int: ...

    def embed(self, text: str) -> np.ndarray: ...

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]: ...


def embed_query_text(embedder: Embedder, text: str) -> np.ndarray:
    """Embed a search query using ``embedder.embed_query`` when available."""
    fn = getattr(embedder, "embed_query", None)
    if callable(fn):
        return fn(text)  # type: ignore[no-any-return]
    return embedder.embed(text)
