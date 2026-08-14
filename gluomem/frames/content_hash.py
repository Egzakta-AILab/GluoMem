"""gluomem.frames.content_hash — pure content-hash dedup key.

The dedup key for I-frames: a SHA-256 hex digest of the *normalized* content.
Normalization strips leading/trailing whitespace and collapses internal
whitespace runs to a single space, so ``"  remember the milk  "`` and
``"remember the milk"`` produce the same hash (and thus dedup to one frame).

This is a pure function — no DB, no side effects. The FrameStore calls it to
populate ``memory_frames.content_hash`` and to look up an existing frame with
the same hash within the same GOP before inserting.
"""

from __future__ import annotations

import hashlib


def content_hash(content: str) -> str:
    """Return the SHA-256 hex digest of the normalized ``content``.

    Normalization: ``" ".join(content.split())`` — this strips leading and
    trailing whitespace AND collapses every internal whitespace run (spaces,
    tabs, newlines) to a single space. The result is the canonical dedup key.

    Examples
    --------
    >>> content_hash("  remember the milk  ") == content_hash("remember the milk")
    True
    >>> content_hash("remember   the   milk") == content_hash("remember the milk")
    True
    """
    normalized = " ".join(content.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
