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
}

_CYRILLIC_TO_LATIN_MAP.update(
    {k.upper(): v.capitalize() for k, v in _CYRILLIC_TO_LATIN_MAP.copy().items()}
)

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
    "ovi",
    "ova",
    "eve",
    "ima",
    "om",
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
    return unicodedata.normalize("NFKC", text).translate(_CYRILLIC_TRANSLATION)


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
    if ascii_token in {"program", "radio"} or len(ascii_token) < 5 or not ascii_token.isalpha():
        return None
    for suffix in _SERBIAN_SUFFIXES:
        if ascii_token.endswith(suffix) and len(ascii_token) - len(suffix) >= 4:
            return ascii_token[: -len(suffix)]
    return None


def _variants(token: str) -> list[str]:
    raw = token.lower()
    latin = cyrillic_to_latin(raw).lower()
    ascii_form = fold_diacritics(latin).lower()
    forms = list(dict.fromkeys((raw, latin, ascii_form, ascii_form.replace("dj", "d"))))
    return list(dict.fromkeys([*forms, *(stem for form in forms if (stem := _serbian_stem(form)))]))


def fold_multilingual_text(text: str) -> str:
    """Emit the same token variants used by multilingual queries."""
    return " ".join(
        dict.fromkeys(
            variant for match in _WORD_RE.finditer(text) for variant in _variants(match.group())
        )
    )


def fold_query_text(query: str) -> str:
    """Build a literal PostgreSQL ``to_tsquery('simple', ...)`` expression."""
    groups = []
    for match in _WORD_RE.finditer(query):
        forms = _variants(match.group())
        groups.append("(" + " | ".join("'" + form.replace("'", "''") + "'" for form in forms) + ")")
    return " & ".join(groups)


def expand_fts5_query(sanitized_query: str) -> str:
    """Expand each literal FTS5 term with symmetric script and stem variants."""
    groups = []
    for match in _QUOTED_PHRASE_RE.finditer(sanitized_query):
        token = (match.group(1) or match.group(2) or "").replace('"', "").strip()
        if not token:
            continue
        forms = _variants(token)
        expression = " OR ".join('"' + form + '"' for form in forms)
        groups.append("(" + expression + ")" if len(forms) > 1 else expression)
    return " AND ".join(groups)


__all__ = [
    "cyrillic_to_latin",
    "expand_fts5_query",
    "fold_diacritics",
    "fold_multilingual_text",
    "fold_query_text",
    "fold_to_ascii",
]
