"""Pure-function sliding-window text chunker."""

from __future__ import annotations

from dataclasses import dataclass

from gluomem.config.app_config import ChunkingConfig


@dataclass(frozen=True)
class Chunk:
    """A single chunk of a frame's content with character offsets."""

    content: str
    char_start: int
    char_end: int


def chunk(text: str, cfg: ChunkingConfig) -> list[Chunk]:
    """Split ``text`` into overlapping chunks per ``cfg``."""
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
        raise ValueError(f"chunking.min_chunk_chars must be non-negative, got {min_chunk}")

    n = len(text)
    if n < min_chunk:
        return [Chunk(content=text, char_start=0, char_end=n)]

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
