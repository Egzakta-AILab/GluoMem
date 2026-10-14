"""Serbian (Cyrillic, Latin with diacritics, ASCII-folded) and English FTS normalization."""

from __future__ import annotations

import re
import unicodedata

# Serbian Cyrillic (Vuk Karadžić azbuka) + common Cyrillic fallback -> Serbian Latin (gajica)
_CYRILLIC_TO_LATIN_MAP: dict[str, str] = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "ђ": "đ",
    "е": "e",
    "ж": "ž",
    "з": "z",
    "и": "i",
    "ј": "j",
    "к": "k",
    "л": "l",
    "љ": "lj",
    "м": "m",
    "н": "n",
    "њ": "nj",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "ћ": "ć",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "c",
    "ч": "č",
    "џ": "dž",
    "ш": "š",
    # Uppercase
    "А": "A",
    "Б": "B",
    "В": "V",
    "Г": "G",
    "Д": "D",
    "Ђ": "Đ",
    "Е": "E",
    "Ж": "Ž",
    "З": "Z",
    "И": "I",
    "Ј": "J",
    "К": "K",
    "Л": "L",
    "Љ": "Lj",
    "М": "M",
    "Н": "N",
    "Њ": "Nj",
    "О": "O",
    "П": "P",
    "Р": "R",
    "С": "S",
    "Т": "T",
    "Ћ": "Ć",
    "У": "U",
    "Ф": "F",
    "Х": "H",
    "Ц": "C",
    "Ч": "Č",
    "Џ": "Dž",
    "Ш": "Š",
    # Extended Cyrillic fallback characters
    "ё": "jo",
    "й": "j",
    "щ": "šč",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "ju",
    "я": "ja",
    "Ё": "Jo",
    "Й": "J",
    "Щ": "Šč",
    "Ъ": "",
    "Ы": "Y",
    "Ь": "",
    "Э": "E",
    "Ю": "Ju",
    "Я": "Ja",
}

_CYRILLIC_TRANSLATION = str.maketrans(_CYRILLIC_TO_LATIN_MAP)

_LATIN_DIACRITIC_TO_ASCII: dict[str, str] = {
    "đ": "dj",
    "Đ": "Dj",
    "č": "c",
    "Č": "C",
    "ć": "c",
    "Ć": "C",
    "š": "s",
    "Š": "S",
    "ž": "z",
    "Ž": "Z",
}

_LATIN_ASCII_TRANSLATION = str.maketrans(_LATIN_DIACRITIC_TO_ASCII)

_WORD_RE = re.compile(r"[^\W_]+(?:[-.][^\W_]+)*", re.UNICODE)
_QUOTED_PHRASE_RE = re.compile(r'"([^"]+)"|(\S+)')

# Common Serbian noun/adjective case suffixes (ordered longest-first)
_SERBIAN_SUFFIXES: tuple[str, ...] = (
    "ovima",
    "evima",
    "ama",
    "ima",
    "om",
    "em",
    "og",
    "eg",
    "oj",
    "u",
    "a",
    "e",
    "i",
    "o",
)


def cyrillic_to_latin(text: str) -> str:
    """Transliterate Serbian Cyrillic characters to Serbian Latin (preserving Latin diacritics)."""
    if not text:
        return ""
    return text.translate(_CYRILLIC_TRANSLATION)


def fold_diacritics(text: str) -> str:
    """Fold Serbian Latin diacritics (``đ -> dj``, ``č/ć -> c``, ``š -> s``, ``ž -> z``)."""
    if not text:
        return ""
    step = text.translate(_LATIN_ASCII_TRANSLATION)
    decomposed = unicodedata.normalize("NFD", step)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def fold_to_ascii(text: str) -> str:
    """Convert Cyrillic or Latin text to lowercase ASCII-folded Latin."""
    if not text:
        return ""
    return fold_diacritics(cyrillic_to_latin(text)).lower()


def _serbian_stem(ascii_token: str) -> str | None:
    """Return a light stem for a lowercase ASCII token if a Serbian case suffix applies."""
    if len(ascii_token) < 5 or not ascii_token.isalpha():
        return None
    for suffix in _SERBIAN_SUFFIXES:
        if ascii_token.endswith(suffix) and len(ascii_token) - len(suffix) >= 4:
            return ascii_token[: -len(suffix)]
    return None


def fold_multilingual_text(text: str) -> str:
    """Build normalized ``folded_content`` for indexing across SR (Cyrillic/Latin/ASCII) & EN.

    For each token, emits (deduplicated, order-preserving):
    1. Lowercased original token (preserving Cyrillic or Latin diacritics).
    2. Lowercased Serbian Latin transliteration (if original had Cyrillic).
    3. Lowercased ASCII-folded form (``đ -> dj``, ``č/ć -> c``, ``š -> s``, ``ž -> z``).
    4. Secondary ``đ -> d`` folded variant if ``đ``/``ђ`` was present (e.g. ``dorde``).
    5. Light Serbian morphological stem (e.g. ``beogradu -> beograd``).
    """
    if not text or not text.strip():
        return ""

    seen: set[str] = set()
    out: list[str] = []

    def _emit(tok: str) -> None:
        if tok and tok not in seen:
            seen.add(tok)
            out.append(tok)

    for match in _WORD_RE.finditer(text):
        raw_tok = match.group(0).lower()
        latin_tok = cyrillic_to_latin(raw_tok).lower()
        ascii_tok = fold_diacritics(latin_tok).lower()
        _emit(raw_tok)
        _emit(latin_tok)
        _emit(ascii_tok)
        if "đ" in latin_tok:
            d_variant = fold_diacritics(latin_tok.replace("đ", "d")).lower()
            _emit(d_variant)
        stem = _serbian_stem(ascii_tok)
        if stem:
            _emit(stem)

    return " ".join(out)


def fold_query_text(query: str) -> str:
    """Normalize a raw search query into lowercase ASCII-folded tokens for ``'simple'`` tsquery."""
    if not query or not query.strip():
        return ""
    tokens: list[str] = []
    for raw in query.replace('"', " ").split():
        folded = fold_to_ascii(raw)
        if folded:
            tokens.append(folded)
    return " ".join(tokens)


def expand_fts5_query(sanitized_query: str) -> str:
    """Expand a sanitized FTS5 query so Cyrillic, Latin-diacritic, and ASCII-folded tokens match.

    Pure ASCII tokens without Cyrillic/diacritic differences remain single quoted phrases
    so FTS5 Porter stemming on ``content`` and exact matching on ``folded_content`` both work.
    Tokens with Cyrillic or Serbian diacritics expand to ``("raw" OR "latin" OR "ascii")``.
    """
    if not sanitized_query or not sanitized_query.strip():
        return ""

    groups: list[str] = []
    for match in _QUOTED_PHRASE_RE.finditer(sanitized_query):
        token = (match.group(1) or match.group(2) or "").replace('"', "").strip()
        if not token:
            continue
        raw_low = token.lower()
        latin_low = cyrillic_to_latin(raw_low).lower()
        ascii_low = fold_diacritics(latin_low).lower()

        variants: list[str] = []
        for candidate in (token, latin_low, ascii_low):
            if candidate and candidate.lower() not in {v.lower() for v in variants}:
                variants.append(candidate)

        stem = _serbian_stem(ascii_low)
        if stem and stem != ascii_low and raw_low != ascii_low:
            if stem not in {v.lower() for v in variants}:
                variants.append(stem)

        if len(variants) == 1:
            groups.append(f'"{variants[0]}"')
        else:
            or_expr = " OR ".join(f'"{v}"' for v in variants)
            groups.append(f"({or_expr})")

    if any(g.startswith("(") for g in groups):
        return " AND ".join(groups)
    return " ".join(groups)


__all__ = [
    "cyrillic_to_latin",
    "expand_fts5_query",
    "fold_diacritics",
    "fold_multilingual_text",
    "fold_query_text",
    "fold_to_ascii",
]
