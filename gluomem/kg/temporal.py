"""Deterministic rule-based relative-date resolver."""

from __future__ import annotations

import re
from datetime import date, timedelta

__all__ = ["resolve_relative_date"]

DateLike = date | str

_N_UNITS_AGO_RE = re.compile(
    r"^(\d+)\s+(day|week|month|year)s?\s+ago$", re.IGNORECASE
)
_IN_N_UNITS_RE = re.compile(
    r"^in\s+(\d+)\s+(day|week|month|year)s?$", re.IGNORECASE
)
_LAST_N_UNITS_RE = re.compile(
    r"^last\s+(\d+)\s+(day|week|month|year)s?$", re.IGNORECASE
)
_NEXT_N_UNITS_RE = re.compile(
    r"^next\s+(\d+)\s+(day|week|month|year)s?$", re.IGNORECASE
)

_UNIT_DAYS: dict[str, int] = {
    "day": 1,
    "week": 7,
    "month": 30,
    "year": 365,
}

_WORD_DELTAS: dict[str, int] = {
    "today": 0,
    "yesterday": -1,
    "tomorrow": 1,
    "last week": -7,
    "next week": 7,
    "last month": -30,
    "next month": 30,
    "last year": -365,
    "next year": 365,
}


def resolve_relative_date(cue: str, reference_date: DateLike) -> str:
    """Map a relative-date ``cue`` to an ISO ``"YYYY-MM-DD"`` date."""
    text = (cue or "").strip().lower()
    if not text:
        raise ValueError("empty relative-date cue")

    ref = _to_date(reference_date)

    if text in _WORD_DELTAS:
        return (ref + timedelta(days=_WORD_DELTAS[text])).isoformat()

    for pattern, sign in (
        (_N_UNITS_AGO_RE, -1),
        (_LAST_N_UNITS_RE, -1),
        (_IN_N_UNITS_RE, 1),
        (_NEXT_N_UNITS_RE, 1),
    ):
        m = pattern.match(text)
        if m:
            n = int(m.group(1))
            unit = m.group(2).lower()
            return (ref + timedelta(days=sign * n * _UNIT_DAYS[unit])).isoformat()

    raise ValueError(f"unrecognized relative-date cue: {cue!r}")


def _to_date(value: DateLike) -> date:
    if isinstance(value, date):
        return value
    s = value.strip()
    if len(s) == 10 and s[4] == "-":
        return date.fromisoformat(s)
    return date.fromisoformat(s[:10])
