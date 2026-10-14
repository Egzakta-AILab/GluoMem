"""Token-preserving full-text query sanitizer."""

from __future__ import annotations

import re

LIKE_FALLBACK: str = ""

_HAS_ALNUM = re.compile(r"[^\W_]", re.UNICODE)


def sanitize(query: str) -> str:
    """Quote whitespace-delimited query tokens as FTS phrases."""
    if not query or not query.strip():
        return LIKE_FALLBACK

    phrases: list[str] = []
    for raw in query.split():
        cleaned = raw.replace('"', "")
        if _HAS_ALNUM.search(cleaned):
            phrases.append(f'"{cleaned}"')

    if not phrases:
        return LIKE_FALLBACK
    return " ".join(phrases)
