"""gluomem.embedding.fingerprint — dim-mismatch guard.

A small helper that asserts an incoming vector's length equals the store's
native dim and raises a clear ``VectorDimMismatchError`` on mismatch. It NEVER
truncates or zero-pads. ``SqliteVecStore.add`` / ``add_batch`` call it before
any write so a wrong-dim vector cannot be persisted.
"""

from __future__ import annotations

import numpy as np


class VectorDimMismatchError(ValueError):
    """Raised when a vector's length != the store's native dim.

    Carries the expected and actual lengths for a clear, actionable message.
    No truncation or zero-padding is performed — the caller must supply a
    native-dim vector.
    """

    def __init__(self, expected: int, actual: int) -> None:
        self.expected: int = expected
        self.actual: int = actual
        super().__init__(
            f"Vector dim mismatch: expected {expected} but got {actual}. "
            f"gluomem stores vectors at the embedder's native dim — no truncation "
            f"or zero-padding is performed. Re-embed at dim={expected} or "
            f"create a store at dim={actual}."
        )


def assert_dim(vec: np.ndarray, dim: int) -> None:
    """Raise ``VectorDimMismatchError`` unless ``vec`` has length ``dim``.

    Accepts any 1-D ndarray-like; coerces the length via ``np.asarray``. A
    wrong-dim vector raises and the caller must not write it.
    """
    arr = np.asarray(vec)
    if arr.shape != (dim,):
        actual = int(arr.shape[0]) if arr.ndim >= 1 else 0
        raise VectorDimMismatchError(expected=dim, actual=actual)
