"""gluomem.kg.temporal — deterministic relative-date resolver.

A DETERMINISTIC, rule-based resolver (NO LLM, no network) mapping relative time
cues to ISO dates against a fixed reference date. This feeds frame
``event_time`` at write time (``event_time`` is the WORLD/event time, distinct
from the ingest-time ``created_at``).

Handled cues (case-insensitive, whitespace-tolerant):
  - ``"today"``        -> reference
  - ``"yesterday"``    -> reference - 1 day
  - ``"tomorrow"``     -> reference + 1 day
  - ``"last week"``    -> reference - 7 days
  - ``"next week"``    -> reference + 7 days
  - ``"N days ago"``   -> reference - N days   (N a non-negative integer)
  - ``"N day ago"``    -> reference - N days   (singular form)
  - ``"in N days"``    -> reference + N days

An unrecognized cue raises a clear ``ValueError`` — never a silent guess. The
resolver is a pure function (modulo calendar arithmetic); the same inputs always
return the same output (repeatable, no randomness).
"""

from __future__ import annotations

import re
from datetime import date, timedelta

__all__ = ["resolve_relative_date"]

# A reference date may be supplied as a ``datetime.date`` or an ISO
# ``"YYYY-MM-DD"`` string.
DateLike = date | str


# --- cue patterns (compiled once) ------------------------------------------

_N_DAYS_AGO_RE = re.compile(r"^(\d+)\s+days?\s+ago$", re.IGNORECASE)
_IN_N_DAYS_RE = re.compile(r"^in\s+(\d+)\s+days?$", re.IGNORECASE)

# Fixed-word cues -> signed day delta from the reference.
_WORD_DELTAS: dict[str, int] = {
    "today": 0,
    "yesterday": -1,
    "tomorrow": 1,
    "last week": -7,
    "next week": 7,
}


def resolve_relative_date(cue: str, reference_date: DateLike) -> str:
    """Map a relative-date ``cue`` to an ISO ``"YYYY-MM-DD"`` date.

    Parameters
    ----------
    cue:
        The relative-time phrase (e.g. ``"yesterday"``, ``"last week"``,
        ``"3 days ago"``, ``"in 2 days"``). Case-insensitive; surrounding
        whitespace is tolerated.
    reference_date:
        The anchor date, as a :class:`datetime.date` or ISO ``"YYYY-MM-DD"``
        string.

    Returns
    -------
    The resolved date as an ISO ``"YYYY-MM-DD"`` string.

    Raises
    ------
    ValueError:
        If ``cue`` is empty or unrecognized (never a silent guess).
    """
    text = (cue or "").strip().lower()
    if not text:
        raise ValueError("empty relative-date cue")

    ref = _to_date(reference_date)

    # Fixed-word cues.
    if text in _WORD_DELTAS:
        return (ref + timedelta(days=_WORD_DELTAS[text])).isoformat()

    # "N days ago" / "N day ago".
    m = _N_DAYS_AGO_RE.match(text)
    if m:
        n = int(m.group(1))
        return (ref - timedelta(days=n)).isoformat()

    # "in N days" / "in N day".
    m = _IN_N_DAYS_RE.match(text)
    if m:
        n = int(m.group(1))
        return (ref + timedelta(days=n)).isoformat()

    raise ValueError(f"unrecognized relative-date cue: {cue!r}")


# --- helpers (pure) --------------------------------------------------------


def _to_date(value: DateLike) -> date:
    """Coerce a ``date`` or ISO ``"YYYY-MM-DD"`` string to a :class:`date`."""
    if isinstance(value, date):
        return value
    s = value.strip()
    if len(s) == 10 and s[4] == "-":
        return date.fromisoformat(s)
    # ``date.fromisoformat`` (3.11+) also accepts full datetimes; tolerate them
    # by taking the date portion.
    return date.fromisoformat(s[:10])
