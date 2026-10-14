"""UTC timestamp conventions for stored and compared timestamps."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

__all__ = ["Clock", "now_utc", "parse_utc", "to_storage", "utc_key", "utc_now"]

Clock = Callable[[], datetime]

_KEY_FMT = "%Y-%m-%dT%H:%M:%S"


def utc_now() -> datetime:
    """Return the current aware UTC time."""
    return datetime.now(UTC)


now_utc = utc_now


def parse_utc(value: str | datetime) -> datetime:
    """Parse an ISO / SQLite timestamp or date into an aware UTC datetime."""
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
    """Return fixed-width UTC sort key ``YYYY-MM-DDTHH:MM:SS`` for ``value``."""
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
    verbatim (already UTC by convention). Aware values and ``datetime``
    objects become naive UTC ``YYYY-MM-DDTHH:MM:SS``.
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
