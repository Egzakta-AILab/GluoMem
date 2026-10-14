"""gluomem.frames.eligibility — the single recall/consolidation eligibility rule.

Trust is ordered ``trusted > normal > low > quarantined``. Two rules follow:

- **Recall eligibility.** A frame whose trust is in ``RECALL_INELIGIBLE_TRUST``
  never reaches usable recall: retrieval lanes, graph-contextual scoring,
  reranking and context assembly all skip it. It stays in the store and is
  readable by id (``FrameStore.get_frame``) for administrative inspection.
- **Merge compatibility.** Derived content (compaction) inherits the base
  frame's trust, so a delta may be merged into a base only when doing so
  cannot raise the trust of the delta's text: same GOP, same source, both
  recall-eligible, and the delta's trust at least the base's trust.

Source labels describe provenance; they are not tenant or user
authorization and are not used to grant recall.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gluomem.frames.store import Frame, Trust

__all__ = [
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

RECALL_INELIGIBLE_TRUST: frozenset[str] = frozenset({"quarantined"})

# For SQL filters: ``f"trust NOT IN ({INELIGIBLE_PLACEHOLDERS})"`` bound with
# ``*INELIGIBLE_PARAMS``.
INELIGIBLE_PARAMS: tuple[str, ...] = tuple(sorted(RECALL_INELIGIBLE_TRUST))
INELIGIBLE_PLACEHOLDERS: str = ",".join("?" * len(INELIGIBLE_PARAMS))


def is_recall_eligible(trust: Trust | str) -> bool:
    """True when a frame with this trust may feed usable recall."""
    return trust not in RECALL_INELIGIBLE_TRUST


def can_merge_into(base: Frame, delta: Frame) -> bool:
    """True when ``delta`` may be merged into ``base`` without promoting it."""
    return (
        base.gop_id == delta.gop_id
        and base.source == delta.source
        and is_recall_eligible(base.trust)
        and is_recall_eligible(delta.trust)
        and TRUST_RANK[delta.trust] >= TRUST_RANK[base.trust]
    )
