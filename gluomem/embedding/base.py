"""gluomem.embedding.base — the Embedder protocol.

A minimal, structural-typing protocol that every embedder (mock, MiniLM, ...)
satisfies. The deterministic ``MockEmbedder``, the lazy-loading MiniLM
provider, and the loud-banner-on-fallback factory live in
:mod:`gluomem.embedding.providers`.

The protocol is intentionally small: ``dim`` plus ``embed`` / ``embed_batch``.
Vectors are native-dim ``float32`` ndarrays of shape ``(dim,)`` — no zero-pad.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """An embedder turns text into a fixed-dim float32 vector.

    Implementations MUST be deterministic for identical inputs within a single
    process (the mock is fully deterministic across processes; MiniLM is
    deterministic given a pinned model). Vectors are at the embedder's native
    dim — never zero-padded or truncated.
    """

    @property
    def dim(self) -> int:
        """The native embedding dimension (e.g. 384 for MiniLM)."""
        ...

    def embed(self, text: str) -> np.ndarray:
        """Embed a single ``text`` → float32 ndarray of shape ``(self.dim,)``."""
        ...

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of texts → one float32 ``(dim,)`` vector per text."""
        ...
