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


def suppression_key(text: str) -> str:
    """Hash case/punctuation variants; paraphrases are not suppressed."""
    import unicodedata

    from gluomem.search.multilingual import fold_to_ascii

    folded = fold_to_ascii(text).casefold()
    folded = "".join(c for c in folded if not unicodedata.category(c).startswith("P"))
    return content_hash(folded)
