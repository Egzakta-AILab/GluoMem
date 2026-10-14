"""Word-boundary memory-kind classification for English and Serbian."""

import re

from gluomem.search.multilingual import fold_to_ascii
from gluomem.storage.scope import MemoryKind

_PREFERENCE_CUES = (
    "prefer",
    "prefers",
    "preference",
    "always use",
    "never use",
    "favorite",
    "favourite",
    "like to",
    "likes to",
    "volim",
    "preferiram",
    "uvek koristim",
    "nikada ne koristim",
    "omiljen",
)
_DECISION_CUES = (
    "decided",
    "decision",
    "agreed to",
    "we chose",
    "selected",
    "odlucili",
    "odluka",
    "dogovorili",
    "izabrali smo",
    "usvojili",
)
_PROCEDURE_CUES = (
    "procedure",
    "runbook",
    "workflow",
    "step 1",
    "step 2",
    "first,",
    "how to",
    "procedura",
    "uputstvo",
    "korak 1",
    "proces",
)
_EPISODE_CUES = (
    "yesterday",
    "today",
    "last week",
    "meeting",
    "incident",
    "happened",
    "called",
    "visited",
    "juce",
    "danas",
    "prosle nedelje",
    "sastanak",
    "desilo se",
)
_PROFILE_CUES = (
    "my name is",
    "i work at",
    "works at",
    "my role is",
    "role is",
    "title is",
    "lives in",
    "based in",
    "zovem se",
    "radim u",
    "moja uloga",
    "inzenjer",
    "direktor",
    "zivi u",
)


_CUE_PATTERNS: tuple[tuple[MemoryKind, re.Pattern[str]], ...] = tuple(
    (kind, re.compile(r"(?<!\w)(?:" + "|".join(re.escape(cue) for cue in cues) + r")(?!\w)"))
    for kind, cues in (
        ("procedure", _PROCEDURE_CUES),
        ("decision", _DECISION_CUES),
        ("preference", _PREFERENCE_CUES),
        ("profile", _PROFILE_CUES),
        ("episode", _EPISODE_CUES),
    )
)


def classify_memory_kind(
    text: str,
    *,
    default: MemoryKind = "fact",
) -> MemoryKind:
    """Classify a statement into one of GluoMem's memory kinds (EN + SR)."""
    folded = fold_to_ascii(text).casefold()
    for kind, pattern in _CUE_PATTERNS:
        if pattern.search(folded):
            return kind
    return default
