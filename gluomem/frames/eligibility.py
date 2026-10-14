"""Recall and consolidation eligibility rules for frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gluomem.frames.store import Frame, Trust

__all__ = [
    "ELIGIBLE_TRUST",
    "INELIGIBLE_PARAMS",
    "INELIGIBLE_PLACEHOLDERS",
    "RECALL_INELIGIBLE_TRUST",
    "TRUST_RANK",
    "can_merge_into",
    "is_recall_eligible",
]

TRUST_RANK: dict[str, int] = {
    "trusted": 3,
    "normal": 2,
    "low": 1,
    "quarantined": 0,
}

ELIGIBLE_TRUST: tuple[str, ...] = ("trusted", "normal", "low")
RECALL_INELIGIBLE_TRUST: frozenset[str] = frozenset({"quarantined"})

INELIGIBLE_PARAMS: tuple[str, ...] = tuple(sorted(RECALL_INELIGIBLE_TRUST))
INELIGIBLE_PLACEHOLDERS: str = ",".join("?" * len(INELIGIBLE_PARAMS))


def is_recall_eligible(frame_or_trust: Frame | Trust | str) -> bool:
    """Return True when a frame (or trust level) may feed usable recall."""
    trust = (
        frame_or_trust
        if isinstance(frame_or_trust, str)
        else frame_or_trust.trust
    )
    return trust not in RECALL_INELIGIBLE_TRUST


def can_merge_into(base: Frame, delta: Frame) -> bool:
    """Return True when ``delta`` may be merged into ``base`` without promoting it."""
    return (
        base.gop_id == delta.gop_id
        and base.source == delta.source
        and is_recall_eligible(base.trust)
        and is_recall_eligible(delta.trust)
        and TRUST_RANK[delta.trust] >= TRUST_RANK[base.trust]
    )
