"""gluomem.timeutil — one UTC convention for stored timestamps.

Rules (applied to ``created_at``, ``event_time``, KG ``valid_from`` /
``valid_to`` / ``recorded_at`` and search ``since`` / ``until``):

- **Naive values are UTC.** SQLite ``datetime('now')`` writes naive UTC, and
  every naive string GluoMem stores or receives is read the same way.
- **Aware values are converted to UTC.** On write they are stored as naive
  UTC ``YYYY-MM-DDTHH:MM:SS``; on read any stored offset is honoured, so
  ``10:00+02:00`` and ``08:00Z`` are the same instant.
- **Date-only values** (``YYYY-MM-DD``) mean midnight UTC of that day and are
  stored verbatim (day precision is kept).
- **Comparison** goes through :func:`utc_key`, a fixed-width UTC string, so
  lexicographic order equals chronological order. Sub-second precision is
  dropped.

The wall clock is only read through a :data:`Clock`, so evaluation can inject
a fixed one.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

__all__ = ["Clock", "parse_utc", "to_storage", "utc_key", "utc_now"]

Clock = Callable[[], datetime]

_KEY_FMT = "%Y-%m-%dT%H:%M:%S"


def utc_now() -> datetime:
    """The default :data:`Clock`: the current aware UTC time."""
    return datetime.now(UTC)


def parse_utc(value: str | datetime) -> datetime:
    """Parse an ISO / SQLite timestamp or date into an aware UTC datetime.

    Raises ``ValueError`` for strings that are not ISO-8601 dates or datetimes.
    """
    if isinstance(value, datetime):
        dt = value
    else:
        s = value.strip()
        if s.endswith(("Z", "z")):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def utc_key(value: str | datetime) -> str:
    """Fixed-width UTC sort key ``YYYY-MM-DDTHH:MM:SS`` for ``value``.

    A string that does not parse (only possible for rows written outside
    GluoMem) keeps the legacy padding rule so old files stay readable.
    """
    try:
        return parse_utc(value).strftime(_KEY_FMT)
    except ValueError:
        s = str(value).strip()
        if len(s) == 10 and s[4] == "-":
            return s + "T00:00:00"
        return s.replace(" ", "T", 1) if "T" not in s else s


def to_storage(value: str | datetime | None) -> str | None:
    """Normalize a timestamp for storage.

    ``None`` stays ``None``. Naive strings and date-only strings are stored
    verbatim (they are already UTC by convention). Aware values and
    ``datetime`` objects become naive UTC ``YYYY-MM-DDTHH:MM:SS``. Raises
    ``ValueError`` for unparsable input so bad timestamps fail at write time.
    """
    if value is None:
        return None
    dt = parse_utc(value)
    if isinstance(value, str) and _is_naive_string(value):
        return value.strip()
    return dt.strftime(_KEY_FMT)


def _is_naive_string(value: str) -> bool:
    s = value.strip()
    if s.endswith(("Z", "z")):
        return False
    return datetime.fromisoformat(s).tzinfo is None
