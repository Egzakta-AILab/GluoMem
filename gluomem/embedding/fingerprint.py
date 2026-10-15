"""Vector compatibility guards: dimension assertion and ``EmbedderFingerprint``."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import Any

from gluomem.storage.backend import VectorDimMismatchError, assert_dim

INDEX_SCHEMA = "sqlite-vec/float32/frame+chunk/v1"


@dataclass(frozen=True)
class EmbedderFingerprint:
    """Identity of the encoder that produced a set of vectors."""

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
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:16]

    def differences(self, other: EmbedderFingerprint) -> list[str]:
        return [f.name for f in fields(self) if getattr(self, f.name) != getattr(other, f.name)]


def fingerprint_of(embedder: Any) -> EmbedderFingerprint:
    """Return the embedder's declared fingerprint, or a class-based fallback."""
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


__all__ = [
    "EmbedderFingerprint",
    "INDEX_SCHEMA",
    "VectorDimMismatchError",
    "assert_dim",
    "fingerprint_of",
]
