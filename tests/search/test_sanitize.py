"""Sanitizer tests.

The token-preserving FTS sanitizer is PURE: deterministic, no I/O, no config
required. It prepares a user query for the FTS5 MATCH path while preserving
technical tokens (version strings, product names with digits/hyphens, short
identifiers, dotted method names) by quoting each whitespace-delimited token
as an FTS5 phrase — neutralizing metacharacters without deleting punctuation.
Empty/whitespace-only input yields the LIKE-fallback marker so the HybridSearch
FTS lane can fall back to a LIKE scan.
"""

from __future__ import annotations

import sqlite3

from gluomem.search.sanitize import LIKE_FALLBACK, sanitize

# --- technical-token preservation -------------------------------------------


def test_technical_tokens_survive_intact() -> None:
    """sanitize('error in GPT-4 and v1.2.3 about AI obj.method') returns a
    string still containing 'GPT-4', 'v1.2.3', 'AI', and 'obj.method'
    (case-insensitive), none split into bare alphanumerics."""
    out = sanitize("error in GPT-4 and v1.2.3 about AI obj.method")
    low = out.lower()
    # Each technical token survives as a contiguous substring (not split into
    # bare alphanumerics like "gpt 4", "v1 2 3", "obj method").
    assert "gpt-4" in low
    assert "v1.2.3" in low
    assert "ai" in low
    assert "obj.method" in low


def test_technical_tokens_not_split_into_bare_alphanumerics() -> None:
    """The hyphenated/dotted tokens are NOT split into bare alphanumerics in the
    sanitized output: 'GPT-4' does not appear as 'GPT 4' (space-split), and
    'obj.method' does not appear as 'obj method'."""
    out = sanitize("error in GPT-4 and v1.2.3 about AI obj.method")
    low = out.lower()
    # The contiguous technical substrings survive (quoted phrases keep them).
    assert "gpt-4" in low
    assert "obj.method" in low
    assert "v1.2.3" in low
    # And they are not replaced by a space-split bare-alphanumeric form.
    assert "gpt 4" not in low
    assert "obj method" not in low
    assert "v1 2 3" not in low


def test_sanitized_output_is_valid_fts5_match() -> None:
    """The sanitized string is a VALID FTS5 MATCH query: running it against an
    FTS5 (porter unicode61) table does not raise and matches a document that
    contains the same technical tokens."""
    query = "error in GPT-4 and v1.2.3 about AI obj.method"
    out = sanitize(query)
    assert out  # non-empty -> not the LIKE-fallback marker

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE t USING fts5(content, tokenize='porter unicode61')"
        )
        conn.execute("INSERT INTO t(content) VALUES (?)", (query,))
        # Must not raise OperationalError — the sanitized string is valid FTS5.
        rows = conn.execute("SELECT rowid FROM t WHERE t MATCH ?", (out,)).fetchall()
        assert rows, "sanitized query must match a document containing the tokens"
    finally:
        conn.close()


def test_fts5_metacharacters_neutralized() -> None:
    """FTS5 syntax metacharacters (quotes, asterisk, parens, colon, NEAR/AND/OR
    keywords) are neutralized so the MATCH is valid and does not raise."""
    # A pathological query full of FTS5 metacharacters / operators.
    query = 'say "hello world" foo* bar:baz (a OR b) NEAR(x,y) AND NOT z'
    out = sanitize(query)
    assert out  # not the LIKE marker

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE t USING fts5(content, tokenize='porter unicode61')"
        )
        # The MATCH must execute without raising — that is the validity gate.
        conn.execute("SELECT * FROM t WHERE t MATCH ?", (out,)).fetchall()
    finally:
        conn.close()


def test_short_identifier_preserved() -> None:
    """Short technical identifiers like 'AI' and 'go' (length 2) are preserved,
    not dropped for being too short."""
    out = sanitize("AI and go are tokens")
    low = out.lower()
    assert "ai" in low
    assert "go" in low


# --- LIKE fallback (empty / too-short input) --------------------------------


def test_empty_input_yields_like_fallback() -> None:
    """Empty input yields the LIKE-fallback marker (empty string)."""
    assert sanitize("") == LIKE_FALLBACK
    assert LIKE_FALLBACK == ""


def test_whitespace_only_yields_like_fallback() -> None:
    """Whitespace-only input yields the LIKE-fallback marker."""
    assert sanitize("   \t\n  ") == LIKE_FALLBACK


def test_punctuation_only_yields_like_fallback() -> None:
    """Input that reduces to no usable tokens (only FTS metacharacters that are
    neutralized away) yields the LIKE-fallback marker."""
    # Double-quotes are stripped as the only FTS5 phrase-syntax metacharacter;
    # a query of only quotes/whitespace has no usable token.
    assert sanitize('""  ""') == LIKE_FALLBACK
