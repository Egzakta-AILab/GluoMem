"""gluomem.embedding.chunker — pure-function text chunker.

``chunk(text, cfg)`` is a PURE function: deterministic, no I/O, no globals.
Every size comes from the passed ``ChunkingConfig`` (max_chars=2000,
overlap_chars=200, min_chunk_chars=1500) — nothing is hardcoded here.

Splitting rule:

- If ``len(text) < min_chunk_chars`` → a SINGLE whole-frame chunk covering the
  full text (``char_start=0``, ``char_end=len(text)``).
- Otherwise → sliding windows of ``max_chars`` advancing by
  ``max_chars - overlap_chars`` so adjacent chunks overlap by exactly
  ``overlap_chars``. The final window is clipped at ``len(text)`` so no chunk
  runs past the end and the last chunk may be shorter than ``max_chars``.

The output is a list of :class:`Chunk` dataclasses, stable across calls for
identical input + config.
"""

from __future__ import annotations

from dataclasses import dataclass

from gluomem.config.app_config import ChunkingConfig


@dataclass(frozen=True)
class Chunk:
    """A single chunk of a frame's content.

    ``char_start`` / ``char_end`` are offsets into the ORIGINAL text such that
    ``text[char_start:char_end] == content``. Frozen for stable equality /
    hashability (the chunker's determinism guarantee).
    """

    content: str
    char_start: int
    char_end: int


def chunk(text: str, cfg: ChunkingConfig) -> list[Chunk]:
    """Split ``text`` into overlapping chunks per ``cfg``.

    Pure and deterministic: the same ``(text, cfg)`` always yields the same
    list of :class:`Chunk`. No I/O, no mutation of inputs.

    Parameters
    ----------
    text:
        The frame content to chunk.
    cfg:
        A ``ChunkingConfig`` with ``max_chars``, ``overlap_chars``,
        ``min_chunk_chars``. All sizes come from config — this function holds
        no literals beyond the documented algorithm shape.
    """
    max_chars = int(cfg.max_chars)
    overlap = int(cfg.overlap_chars)
    min_chunk = int(cfg.min_chunk_chars)

    if max_chars <= 0:
        raise ValueError(f"chunking.max_chars must be positive, got {max_chars}")
    if overlap < 0:
        raise ValueError(f"chunking.overlap_chars must be non-negative, got {overlap}")
    if overlap >= max_chars:
        raise ValueError(
            f"chunking.overlap_chars ({overlap}) must be < max_chars ({max_chars}) "
            f"or the window would not advance"
        )
    if min_chunk < 0:
        raise ValueError(
            f"chunking.min_chunk_chars must be non-negative, got {min_chunk}"
        )

    n = len(text)

    # Whole-frame shortcut: text shorter than the threshold is one chunk.
    if n < min_chunk:
        return [Chunk(content=text, char_start=0, char_end=n)]

    # Sliding windows of max_chars advancing by (max_chars - overlap). The
    # advance is > 0 because overlap < max_chars (validated above). The final
    # window is clipped at n so no chunk runs past the end of the text.
    advance = max_chars - overlap
    out: list[Chunk] = []
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        out.append(Chunk(content=text[start:end], char_start=start, char_end=end))
        if end >= n:
            break
        start += advance
    return out
