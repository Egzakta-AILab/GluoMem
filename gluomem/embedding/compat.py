"""One embedder per ``*.mind`` store: fingerprint binding and compatibility checks."""

from __future__ import annotations

from gluomem.embedding.fingerprint import EmbedderFingerprint, fingerprint_of
from gluomem.storage.db import MindDB

FINGERPRINT_META_KEY = "embedding_fingerprint"


class EmbeddingMismatchError(ValueError):
    """The embedder does not match the one that wrote this store's vectors."""

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
    """Return the fingerprint recorded in ``db``, or ``None``."""
    raw = db.meta_get(FINGERPRINT_META_KEY)
    return None if raw is None else EmbedderFingerprint.from_json(raw)


def has_vectors(db: MindDB) -> bool:
    """Return True when any frame or chunk vector is stored."""
    return db.frame_vec.count() > 0 or db.chunk_vec.count() > 0


def write_fingerprint(db: MindDB, fp: EmbedderFingerprint) -> None:
    """Record ``fp`` inside the caller's transaction (no commit)."""
    db.meta_set(FINGERPRINT_META_KEY, fp.to_json(), commit=False)


def bind_embedder(db: MindDB, embedder: object) -> EmbedderFingerprint:
    """Check ``embedder`` against ``db`` and record it on first use."""
    current = fingerprint_of(embedder)
    stored = stored_fingerprint(db)
    if stored is None:
        if has_vectors(db):
            raise EmbeddingMismatchError(str(db.path), None, current)
        db.meta_set(FINGERPRINT_META_KEY, current.to_json(), commit=True)
        return current
    if stored != current:
        raise EmbeddingMismatchError(str(db.path), stored, current)
    return current


def adopt_embedder(db: MindDB, embedder: object) -> EmbedderFingerprint:
    """Record ``embedder`` for a store without a fingerprint, without re-embedding."""
    current = fingerprint_of(embedder)
    stored = stored_fingerprint(db)
    if stored is not None and stored != current:
        raise EmbeddingMismatchError(str(db.path), stored, current)
    db.meta_set(FINGERPRINT_META_KEY, current.to_json(), commit=True)
    return current
