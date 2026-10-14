"""Content-hash deduplication key for frames."""

from __future__ import annotations

import hashlib


def content_hash(text: str) -> str:
    """Return the canonical SHA-256 hex digest for ``text``.

    Leading/trailing whitespace is stripped and internal whitespace runs are
    collapsed to single spaces before hashing.
    """
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
