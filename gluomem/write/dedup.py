"""gluomem.write.dedup — semantic (paraphrase) near-duplicate detection.

Exact content-hash dedup (``gate.admit`` → ``store.find_frame_by_hash``) misses
restated facts: "Alice got promoted to senior analyst" and "Alice was promoted at
Acme" hash differently and both survive in the ``{conv}:facts`` GOP, inflating the
retrieval pool. :class:`SemanticDeduper` closes that gap at admission with an
embedding-cosine gate, scoped to a single GOP.

Design:

- **In-memory, per-GOP.** For each GOP we keep the unit-normalized embedding of
  every frame admitted through the probe. A probe is ``candidate · matrix`` (both
  unit vectors → cosine), max over the GOP. Unlike a global vec0 KNN this can't
  miss the true same-GOP neighbour (no top-k truncation) and needs no vector
  read-back.
- **First-writer wins.** A candidate at or above the threshold is dropped; the
  matched existing frame is returned. No merge / access-count bump.
- **Fresh per conversation.** Built only when ``write.dedup.mode == 'semantic'``;
  the cache holds only this conversation's frames (independent ``.mind`` DBs), so
  it composes with the cross-conversation ingest concurrency.

The candidate vector computed for the probe is returned so the caller can hand it
to :meth:`record` after a real admit — the deduper never re-embeds a frame.
"""

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
    """Outcome of one probe. ``vector`` is the candidate's UNIT embedding, carried
    so the caller can :meth:`record` it after admit without re-embedding."""

    is_dup: bool
    matched_id: int | None
    cos: float
    vector: np.ndarray


class SemanticDeduper:
    """Per-GOP embedding-cosine near-duplicate gate for one conversation's ingest."""

    def __init__(self, embedder: Embedder, *, threshold: float) -> None:
        self._embedder = embedder
        self._threshold = float(threshold)
        # gop -> (frame_ids, unit vectors) in admission order.
        self._by_gop: dict[str, tuple[list[int], list[np.ndarray]]] = {}

    def probe(self, gop: str, content: str) -> DedupProbe:
        """Return whether ``content`` duplicates an existing frame in ``gop``.

        ``is_dup`` iff the max cosine to a recorded same-GOP vector is
        ``>= threshold``. Ties/empty GOP → not a duplicate.
        """
        v = _unit(self._embedder.embed(content))
        entry = self._by_gop.get(gop)
        if entry is None or not entry[0]:
            return DedupProbe(is_dup=False, matched_id=None, cos=0.0, vector=v)
        ids, vecs = entry
        sims = np.vstack(vecs) @ v  # (n,) cosine — both operands are unit vectors
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
        """Drop a cached entry whose frame no longer exists (compaction merges
        older P-frames into the I and referentially deletes them, so a recorded
        id can vanish mid-ingest). Removing it stops a dead vector from winning
        future probes and blocking a live match behind it."""
        entry = self._by_gop.get(gop)
        if entry is None:
            return
        ids, vecs = entry
        for i, fid in enumerate(ids):
            if fid == frame_id:
                ids.pop(i)
                vecs.pop(i)
                return


__all__ = ["DedupProbe", "SemanticDeduper"]
