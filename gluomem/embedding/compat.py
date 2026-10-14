"""gluomem.embedding.compat — one embedder per ``*.mind`` file.

The first embedder bound to a file with no vectors is recorded in the
``meta`` table under :data:`FINGERPRINT_META_KEY`. Every later bind
(``FrameStore`` or ``HybridSearch`` construction) must present the same
:class:`~gluomem.embedding.fingerprint.EmbedderFingerprint`, so vectors from
two encoders are never written to, or queried against, the same index.

A file that already holds vectors but has no recorded fingerprint (written by
0.0.1) is rejected too: its encoder is unknown. Two explicit ways out:

- ``FrameStore.rebuild_index(db, embedder)`` re-embeds every frame with the
  new embedder and records it, in one transaction.
- :func:`adopt_embedder` records the embedder without re-embedding, for an
  operator who knows the existing vectors came from it.
"""

from __future__ import annotations

from gluomem.embedding.fingerprint import EmbedderFingerprint, fingerprint_of
from gluomem.storage.db import MindDB

FINGERPRINT_META_KEY = "embedding_fingerprint"


class EmbeddingMismatchError(ValueError):
    """The embedder does not match the one that wrote this file's vectors."""

    def __init__(
        self,
        path: str,
        stored: EmbedderFingerprint | None,
        current: EmbedderFingerprint,
    ) -> None:
        self.stored = stored
        self.current = current
        if stored is None:
            self.fields: list[str] = []
            detail = (
                f"{path} holds vectors but records no embedder fingerprint "
                f"(written before fingerprints existed), so they cannot be "
                f"checked against {current.implementation}:{current.model}"
            )
        else:
            self.fields = stored.differences(current)
            diffs = ", ".join(
                f"{f}: stored={getattr(stored, f)!r} current={getattr(current, f)!r}"
                for f in self.fields
            )
            detail = f"{path} was indexed by a different embedder ({diffs})"
        super().__init__(
            f"{detail}. Re-embed with FrameStore.rebuild_index(db, embedder), or, "
            f"if the existing vectors are known to come from this embedder and "
            f"no fingerprint is recorded, adopt it with "
            f"gluomem.embedding.compat.adopt_embedder(db, embedder)."
        )


def stored_fingerprint(db: MindDB) -> EmbedderFingerprint | None:
    """The fingerprint recorded in ``db``, or ``None``."""
    raw = db.meta_get(FINGERPRINT_META_KEY)
    return None if raw is None else EmbedderFingerprint.from_json(raw)


def has_vectors(db: MindDB) -> bool:
    """True when any frame or chunk vector is stored."""
    for table in ("memory_frames_vec", "memory_frame_chunks_vec"):
        if db.conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None:
            return True
    return False


def write_fingerprint(db: MindDB, fp: EmbedderFingerprint) -> None:
    """Record ``fp`` inside the caller's transaction (no commit)."""
    db.conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (FINGERPRINT_META_KEY, fp.to_json()),
    )


def bind_embedder(db: MindDB, embedder: object) -> EmbedderFingerprint:
    """Check ``embedder`` against ``db`` and record it on first use.

    Raises :class:`EmbeddingMismatchError` before any vector is read or
    written when the file was indexed by another embedder, or holds vectors
    of unknown origin.
    """
    current = fingerprint_of(embedder)
    stored = stored_fingerprint(db)
    if stored is None:
        if has_vectors(db):
            raise EmbeddingMismatchError(str(db.path), None, current)
        write_fingerprint(db, current)
        db.conn.commit()
        return current
    if stored != current:
        raise EmbeddingMismatchError(str(db.path), stored, current)
    return current


def adopt_embedder(db: MindDB, embedder: object) -> EmbedderFingerprint:
    """Record ``embedder`` for a file without a fingerprint, without re-embedding.

    Refuses to replace a different recorded fingerprint; use
    ``FrameStore.rebuild_index`` for that.
    """
    current = fingerprint_of(embedder)
    stored = stored_fingerprint(db)
    if stored is not None and stored != current:
        raise EmbeddingMismatchError(str(db.path), stored, current)
    write_fingerprint(db, current)
    db.conn.commit()
    return current
