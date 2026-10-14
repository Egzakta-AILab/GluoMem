"""Token-preserving FTS5 query sanitizer tests."""

from __future__ import annotations

import sqlite3

import pytest

from gluomem.search.sanitize import LIKE_FALLBACK, sanitize


def test_technical_tokens_survive_intact_and_unsplit() -> None:
    out = sanitize("error in GPT-4 and v1.2.3 about AI obj.method go")
    low = out.lower()
    for token in ("gpt-4", "v1.2.3", "ai", "obj.method", "go"):
        assert token in low
    for split in ("gpt 4", "obj method", "v1 2 3"):
        assert split not in low


@pytest.mark.parametrize(
    "query",
    [
        "error in GPT-4 and v1.2.3 about AI obj.method",
        'say "hello world" foo* bar:baz (a OR b) NEAR(x,y) AND NOT z',
    ],
)
def test_sanitized_output_is_valid_fts5_match(query: str) -> None:
    out = sanitize(query)
    assert out != LIKE_FALLBACK
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE t USING fts5(content, tokenize='porter unicode61')"
        )
        conn.execute("INSERT INTO t(content) VALUES (?)", (query,))
        rows = conn.execute("SELECT rowid FROM t WHERE t MATCH ?", (out,)).fetchall()
        assert isinstance(rows, list)
    finally:
        conn.close()


@pytest.mark.parametrize("raw", ["", "   \t\n  ", '""  ""'])
def test_empty_or_quote_only_input_yields_like_fallback(raw: str) -> None:
    assert sanitize(raw) == LIKE_FALLBACK == ""
