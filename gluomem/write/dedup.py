"""Semantic (paraphrase) near-duplicate detection per GOP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gluomem.embedding.base import Embedder


def _unit(vec: np.ndarray) -> np.ndarray:
    """L2-normalize to a unit vector (float32); a zero vector is returned as-is."""
    v = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0.0 else v


@dataclass(frozen=True)
class DedupProbe:
    """Outcome of one probe, carrying the candidate's unit embedding."""

    is_dup: bool
    matched_id: int | None
    cos: float
    vector: np.ndarray


class SemanticDeduper:
    """Per-GOP embedding-cosine near-duplicate gate for one conversation's ingest."""

    def __init__(self, embedder: Embedder, *, threshold: float) -> None:
        self._embedder = embedder
        self._threshold = float(threshold)
        self._by_gop: dict[str, tuple[list[int], list[np.ndarray]]] = {}

    def probe(self, gop: str, content: str) -> DedupProbe:
        """Return whether ``content`` duplicates an existing frame in ``gop``."""
        v = _unit(self._embedder.embed(content))
        entry = self._by_gop.get(gop)
        if entry is None or not entry[0]:
            return DedupProbe(is_dup=False, matched_id=None, cos=0.0, vector=v)
        ids, vecs = entry
        sims = np.vstack(vecs) @ v
        i = int(np.argmax(sims))
        cos = float(sims[i])
        if cos >= self._threshold:
            return DedupProbe(is_dup=True, matched_id=ids[i], cos=cos, vector=v)
        return DedupProbe(is_dup=False, matched_id=None, cos=cos, vector=v)

    def record(self, gop: str, frame_id: int, vector: np.ndarray) -> None:
        """Add an admitted frame's unit ``vector`` to ``gop`` for future probes."""
        ids, vecs = self._by_gop.setdefault(gop, ([], []))
        ids.append(int(frame_id))
        vecs.append(np.asarray(vector, dtype=np.float32))

    def forget(self, gop: str, frame_id: int) -> None:
        """Drop a cached entry whose frame no longer exists."""
        entry = self._by_gop.get(gop)
        if entry is None:
            return
        ids, vecs = entry
        for i, fid in enumerate(ids):
            if fid == frame_id:
                ids.pop(i)
                vecs.pop(i)
                return
