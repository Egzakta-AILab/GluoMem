"""Chunker tests.

The chunker is a PURE function: deterministic, no I/O. Every size comes from
``cfg.chunking`` (max_chars=2000, overlap_chars=200, min_chunk_chars=1500).

- Split: a 3000-char text splits into >=2 chunks each <=2000, adjacent
  chunks overlap by 200, and a second identical call returns an identical list.
- Whole-frame: text shorter than min_chunk_chars (1499) returns a
  single whole-frame chunk equal to the full input.
"""

from __future__ import annotations

from gluomem.config.app_config import AppConfig
from gluomem.embedding.chunker import Chunk, chunk


def _cfg() -> AppConfig:
    return AppConfig()


# --- split ------------------------------------------------------------------


def test_chunker_split_overlap_and_stability() -> None:
    """3000 chars -> >=2 chunks, each <=2000, overlap==200, stable."""
    cfg = _cfg()
    max_chars = cfg.chunking.max_chars
    overlap = cfg.chunking.overlap_chars
    min_chunk = cfg.chunking.min_chunk_chars
    assert (max_chars, overlap, min_chunk) == (2000, 200, 1500), (
        "defaults must come from config"
    )

    text = "a" * 3000
    chunks = chunk(text, cfg.chunking)

    # At least 2 chunks for a >max_chars text.
    assert len(chunks) >= 2, f"expected >=2 chunks for 3000 chars, got {len(chunks)}"

    # Every chunk content fits within max_chars.
    for c in chunks:
        assert len(c.content) <= max_chars, (
            f"chunk len {len(c.content)} exceeds max_chars {max_chars}"
        )

    # char_start / char_end bound the content within the original text.
    for c in chunks:
        assert c.char_start == 0 or c.char_start < len(text)
        assert c.char_end <= len(text)
        assert c.char_end - c.char_start == len(c.content)
        assert text[c.char_start : c.char_end] == c.content

    # Adjacent chunks overlap by exactly overlap_chars.
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert b.char_start - a.char_start == max_chars - overlap, (
            f"adjacent advance must be {max_chars - overlap}; "
            f"got {b.char_start - a.char_start}"
        )
        assert a.char_end - b.char_start == overlap, (
            f"adjacent overlap must be {overlap}; got {a.char_end - b.char_start}"
        )

    # Stability: a second identical call returns an identical list.
    again = chunk(text, cfg.chunking)
    assert again == chunks, "chunk() must be deterministic across calls"


def test_chunker_returns_chunk_dataclass() -> None:
    """Each element is a Chunk with content/char_start/char_end."""
    cfg = _cfg()
    chunks = chunk("a" * 3000, cfg.chunking)
    for c in chunks:
        assert isinstance(c, Chunk)
        assert isinstance(c.content, str)
        assert isinstance(c.char_start, int)
        assert isinstance(c.char_end, int)


# --- whole-frame ------------------------------------------------------------


def test_chunker_wholeframe_short_text() -> None:
    """1499 chars (< min_chunk_chars 1500) -> 1 whole-frame chunk."""
    cfg = _cfg()
    text = "a" * 1499
    chunks = chunk(text, cfg.chunking)
    assert len(chunks) == 1, f"expected 1 whole-frame chunk, got {len(chunks)}"
    sole = chunks[0]
    assert sole.content == text
    assert sole.char_start == 0
    assert sole.char_end == len(text)


def test_chunker_wholeframe_boundary() -> None:
    """Exactly min_chunk_chars (1500) is NOT < min_chunk_chars -> it splits (boundary)."""
    cfg = _cfg()
    text = "a" * 1500
    # 1500 is not < 1500, so the split path is taken; max_chars=2000 > 1500,
    # so a single window still covers it. The whole-frame shortcut is strict <.
    chunks = chunk(text, cfg.chunking)
    # The window algorithm produces one chunk covering [0, 1500).
    assert len(chunks) == 1
    assert chunks[0].content == text


# --- purity -----------------------------------------------------------------


def test_chunker_is_pure_no_mutation() -> None:
    """chunk() must not mutate the input text and has no I/O."""
    cfg = _cfg()
    text = "abcdefghij" * 300  # 3000 chars
    original = text
    chunk(text, cfg.chunking)
    assert text == original, "chunk() must not mutate its input"
