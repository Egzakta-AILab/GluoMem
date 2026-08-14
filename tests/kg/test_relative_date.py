"""Deterministic relative-date resolver tests.

``gluomem.kg.temporal.resolve_relative_date(cue, reference_date) -> iso`` is a
DETERMINISTIC, rule-based resolver (NO LLM, no network) mapping relative time
cues to ISO dates against a fixed reference date. This feeds frame
``event_time`` (backdating) at write time.

Pinned values (reference ``2025-01-15``):
  - "yesterday"   -> 2025-01-14
  - "today"       -> 2025-01-15
  - "tomorrow"    -> 2025-01-16
  - "last week"   -> 2025-01-08
  - "next week"   -> 2025-01-22
  - "3 days ago"  -> 2025-01-12
  - "in 2 days"   -> 2025-01-17
"""

from __future__ import annotations

from datetime import date

import pytest

from gluomem.kg.temporal import resolve_relative_date

REF = date(2025, 1, 15)


# --- pinned cues ------------------------------------------------------------


def test_yesterday() -> None:
    """Reference 2025-01-15, 'yesterday' == 2025-01-14."""
    assert resolve_relative_date("yesterday", REF) == "2025-01-14"


def test_today() -> None:
    """'today' maps to the reference date itself."""
    assert resolve_relative_date("today", REF) == "2025-01-15"


def test_tomorrow() -> None:
    """'tomorrow' maps to reference + 1 day."""
    assert resolve_relative_date("tomorrow", REF) == "2025-01-16"


def test_last_week() -> None:
    """Reference 2025-01-15, 'last week' == 2025-01-08."""
    assert resolve_relative_date("last week", REF) == "2025-01-08"


def test_next_week() -> None:
    """'next week' maps to reference + 7 days."""
    assert resolve_relative_date("next week", REF) == "2025-01-22"


def test_n_days_ago() -> None:
    """'N days ago' maps to reference - N days."""
    assert resolve_relative_date("3 days ago", REF) == "2025-01-12"
    assert resolve_relative_date("1 day ago", REF) == "2025-01-14"
    assert resolve_relative_date("10 days ago", REF) == "2025-01-05"


def test_in_n_days() -> None:
    """'in N days' maps to reference + N days."""
    assert resolve_relative_date("in 2 days", REF) == "2025-01-17"
    assert resolve_relative_date("in 0 days", REF) == "2025-01-15"


def test_repeatable_deterministic() -> None:
    """The resolver is repeatable: calling it twice with the same inputs
    returns the same output (no LLM, no randomness)."""
    a = resolve_relative_date("yesterday", REF)
    b = resolve_relative_date("yesterday", REF)
    assert a == b == "2025-01-14"
    assert resolve_relative_date("last week", REF) == resolve_relative_date(
        "last week", REF
    )


def test_reference_as_iso_string() -> None:
    """The reference date may be passed as an ISO string 'YYYY-MM-DD'."""
    assert resolve_relative_date("yesterday", "2025-01-15") == "2025-01-14"
    assert resolve_relative_date("last week", "2025-01-15") == "2025-01-08"


def test_case_insensitive_and_whitespace() -> None:
    """Cue matching is case-insensitive and tolerates surrounding whitespace."""
    assert resolve_relative_date("  Yesterday  ", REF) == "2025-01-14"
    assert resolve_relative_date("LAST WEEK", REF) == "2025-01-08"
    assert resolve_relative_date("  3 Days Ago ", REF) == "2025-01-12"


def test_unknown_cue_raises() -> None:
    """An unrecognized cue raises a clear ValueError (never a silent guess)."""
    with pytest.raises(ValueError):
        resolve_relative_date("sometime last month", REF)
    with pytest.raises(ValueError):
        resolve_relative_date("", REF)
