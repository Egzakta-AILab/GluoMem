"""gluomem.search.sanitize — token-preserving FTS5 query sanitizer.

A PURE function ``sanitize(query) -> str`` that prepares a user query for the
FTS5 MATCH path while PRESERVING technical tokens — version strings (v1.2.3),
product names with digits/hyphens (GPT-4), short identifiers (AI, go), dotted
method names (obj.method), error codes. It must NOT strip these into bare
alphanumerics or drop them.

Strategy: split the query on whitespace and quote each token as an FTS5 *phrase*
(double-quoted string). Inside a phrase, every FTS5 syntax metacharacter
(``*``, ``:``, ``(``, ``)``, ``-``, ``.``, the ``AND``/``OR``/``NOT``/``NEAR``
keywords) is literal text — the ``porter unicode61`` tokenizer handles them —
so quoting neutralizes the MATCH grammar without deleting punctuation. The only
character special inside a phrase is the double-quote itself, which we strip
(neutralize) so the phrase syntax stays valid.

Because every token is wrapped, technical substrings like ``GPT-4`` and
``obj.method`` survive verbatim in the sanitized output, and the FTS5 tokenizer
will match them against content tokenized the same way.

Empty / whitespace-only / no-alphanumeric input yields the LIKE-fallback marker
(an empty string) so the HybridSearch FTS lane can fall back to a LIKE scan.

This module is deterministic, offline, and reads NO config — quoting is a
universal, parameter-free neutralization (no magic numbers).
"""

from __future__ import annotations

import re

#: The marker returned when sanitized output is empty/too-short. The
#: HybridSearch FTS lane treats this as a signal to fall back to a LIKE scan.
#: An empty string is falsy, so ``if not sanitize(q):`` is the fallback check.
LIKE_FALLBACK: str = ""

# At least one ASCII alphanumeric must remain in a (quote-stripped) token for it
# to form a non-empty FTS5 phrase; tokens that reduce to nothing but punctuation
# (e.g. ``"*"``, ``":"``) would yield empty phrases and are dropped.
_HAS_ALNUM = re.compile(r"[A-Za-z0-9]")


def sanitize(query: str) -> str:
    """Prepare ``query`` for the FTS5 MATCH path, preserving technical tokens.

    Each whitespace-delimited token is quoted as an FTS5 phrase, which
    neutralizes all FTS5 metacharacters while keeping hyphenated/dotted/short
    technical tokens (``GPT-4``, ``v1.2.3``, ``AI``, ``obj.method``) intact —
    none split into bare alphanumerics or dropped.

    Returns ``LIKE_FALLBACK`` (``""``) for empty / whitespace-only / no-usable-
    token input, signalling the HybridSearch FTS lane to use a LIKE scan.
    """
    if not query or not query.strip():
        return LIKE_FALLBACK

    phrases: list[str] = []
    for raw in query.split():
        # Strip the only character special inside an FTS5 phrase so the phrase
        # syntax stays valid (neutralize, do not delete the rest).
        cleaned = raw.replace('"', "")
        if _HAS_ALNUM.search(cleaned):
            phrases.append(f'"{cleaned}"')

    if not phrases:
        return LIKE_FALLBACK
    return " ".join(phrases)
