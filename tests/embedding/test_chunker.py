"""Pure text chunker tests."""

from __future__ import annotations

from gluomem.config.app_config import AppConfig
from gluomem.embedding.chunker import Chunk, chunk


def test_chunker_split_overlap_and_wholeframe() -> None:
    cfg = AppConfig()
    text = "a" * 3000
    chunks = chunk(text, cfg.chunking)
    assert len(chunks) >= 2
    for c in chunks:
        assert isinstance(c, Chunk)
        assert len(c.content) <= cfg.chunking.max_chars
        assert text[c.char_start : c.char_end] == c.content
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a.char_end - b.char_start == cfg.chunking.overlap_chars
    assert chunk(text, cfg.chunking) == chunks

    short = "a" * 1499
    assert chunk(short, cfg.chunking) == [Chunk(content=short, char_start=0, char_end=1499)]
    boundary = "a" * 1500
    assert chunk(boundary, cfg.chunking) == [Chunk(content=boundary, char_start=0, char_end=1500)]
