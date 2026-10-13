"""gluomem.embedding.fingerprint — vector compatibility guards.

Two guards:

- :func:`assert_dim` rejects a vector whose length is not the store's native
  dim (``VectorDimMismatchError``). It NEVER truncates or zero-pads.
  ``SqliteVecStore.add`` / ``add_batch`` call it before any write.
- :class:`EmbedderFingerprint` identifies what produced a file's vectors:
  implementation, model, revision, dim, preprocessing and index schema. Two
  encoders with the same dim but a different model or preprocessing have
  different fingerprints, so their vectors are never mixed
  (:mod:`gluomem.embedding.compat` enforces this per ``*.mind`` file).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

#: Version of the vector layout GluoMem writes: one float32 vector per frame
#: plus one per chunk, in sqlite-vec ``vec0`` tables. Bump it when that layout
#: changes so files written under the old layout are rejected, not misread.
INDEX_SCHEMA = "sqlite-vec/float32/frame+chunk/v1"


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


@dataclass(frozen=True)
class EmbedderFingerprint:
    """What produced a set of vectors. Contains no secrets or user content.

    ``revision`` is ``None`` when the model is not pinned; such a fingerprint
    cannot detect an upstream model update published under the same name.
    """

    implementation: str
    model: str
    revision: str | None
    dim: int
    preprocessing: str
    index_schema: str = INDEX_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> EmbedderFingerprint:
        data = json.loads(raw)
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    @property
    def digest(self) -> str:
        """Short stable identifier for logs and reports."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:16]

    def differences(self, other: EmbedderFingerprint) -> list[str]:
        """Names of the fields whose values differ from ``other``."""
        return [
            f.name
            for f in fields(self)
            if getattr(self, f.name) != getattr(other, f.name)
        ]


def fingerprint_of(embedder: Any) -> EmbedderFingerprint:
    """The embedder's declared fingerprint, or a class-based one.

    Embedders shipped with GluoMem declare a ``fingerprint`` property. A
    third-party embedder without one is identified by its class and dim, which
    still separates it from every built-in encoder of the same dim.
    """
    declared = getattr(embedder, "fingerprint", None)
    if isinstance(declared, EmbedderFingerprint):
        return declared
    cls = type(embedder)
    return EmbedderFingerprint(
        implementation=f"{cls.__module__}.{cls.__qualname__}",
        model="undeclared",
        revision=None,
        dim=int(embedder.dim),
        preprocessing="undeclared",
    )
