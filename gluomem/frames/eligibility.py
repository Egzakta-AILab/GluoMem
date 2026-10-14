"""Recall and consolidation eligibility rules for frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gluomem.timeutil import now_utc_iso, utc_key

if TYPE_CHECKING:
    from datetime import datetime

    from gluomem.frames.store import Frame, Trust
    from gluomem.storage.scope import FrameStatus

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


def is_recall_eligible(
    frame_or_trust: Frame | Trust | str,
    *,
    status: FrameStatus | str = "active",
    expires_at: str | None = None,
    pinned: bool = False,
    now: datetime | str | None = None,
    now_iso: str | None = None,
) -> bool:
    """Return True when a frame (or trust level + lifecycle fields) may feed usable recall."""
    if isinstance(frame_or_trust, str):
        trust_val = frame_or_trust
        status_val = status
        exp_val = expires_at
        pinned_val = pinned
    else:
        frame = frame_or_trust
        trust_val = frame.trust
        status_val = getattr(frame, "status", "active")
        exp_val = getattr(frame, "expires_at", None)
        pinned_val = bool(getattr(frame, "pinned", False))

    if trust_val in RECALL_INELIGIBLE_TRUST:
        return False
    if status_val != "active":
        return False
    if exp_val is not None and not pinned_val:
        ref = now_iso if now_iso is not None else now
        eff_now = utc_key(ref) if ref is not None else now_utc_iso()
        if utc_key(exp_val) <= eff_now:
            return False
    return True


def can_merge_into(base: Frame, delta: Frame) -> bool:
    """Return True when ``delta`` may be merged into ``base`` without promoting it."""
    return (
        base.tenant_id == delta.tenant_id
        and base.principal_id == delta.principal_id
        and base.namespace == delta.namespace
        and base.gop_id == delta.gop_id
        and base.source == delta.source
        and is_recall_eligible(base)
        and is_recall_eligible(delta)
        and TRUST_RANK[delta.trust] >= TRUST_RANK[base.trust]
    )
