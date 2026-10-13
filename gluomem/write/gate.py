"""gluomem.write.gate — the single admission point for ALL memory writes.

Every write funnels through ``admit()`` — scan + dedup + index on
every path. No other write-path module may call ``store.create_*`` directly
(mechanically enforced by an architecture test).

Scan semantics: ``InjectionScanner`` is the cheap regex lane; its score is the
max weight among matched ``safety.scanner.patterns``. At or above
``safety.block_threshold`` the gate applies ``safety.fail_mode``:

- ``quarantine`` — demote ``trust`` to ``'quarantined'`` and admit anyway
  for inspection; the frame is recall-ineligible
  (:mod:`gluomem.frames.eligibility`), skips semantic dedup and KG linking,
  and extraction skips it;
- ``reject`` — drop the write entirely (no store row).

Dedup: the gate probes ``store.find_frame_by_hash`` for EVERY frame type, so
whole-conversation re-ingest is idempotent (the store itself only dedups
I-frames); a dedup hit whose index is missing or stale is re-indexed
(``FrameStore.ensure_indexed``). Indexing (FTS + vector + chunks) commits
with the frame row inside the store's ``create_*``; KG entity links are
written post-insert, and skipped
for quarantined content so flagged text never wires into the graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gluomem.frames.eligibility import is_recall_eligible
from gluomem.frames.store import (
    Frame,
    FrameSource,
    FrameType,
    Importance,
    Trust,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gluomem.config.app_config import AppConfig, SafetyConfig
    from gluomem.frames.store import FrameStore
    from gluomem.kg.graph import KnowledgeGraph
    from gluomem.write.dedup import SemanticDeduper


@dataclass
class ScanVerdict:
    """Outcome of one injection scan: score in [0,1] + the regexes that hit."""

    score: float
    matched: list[str] = field(default_factory=list)


class InjectionScanner:
    """Cheap regex injection scanner over ``safety.scanner.patterns``.

    Patterns are compiled once at construction (the config validator already
    guarantees they compile). The optional DSPy semantic classifier is a
    declared seam only — enabling it without an implementation fails loudly
    rather than silently scanning with regexes alone.
    """

    def __init__(self, cfg: SafetyConfig) -> None:
        if cfg.semantic_classifier.enabled:
            raise NotImplementedError(
                "safety.semantic_classifier.enabled=true but no semantic "
                "classifier is implemented (M4 ships the regex lane only); "
                "set safety.semantic_classifier.enabled=false"
            )
        self._patterns: list[tuple[re.Pattern[str], float]] = [
            (re.compile(p.regex), p.weight) for p in cfg.scanner.patterns
        ]

    def scan(self, content: str, *, context: str = "") -> ScanVerdict:  # noqa: ARG002
        """Score ``content``; ``context`` (the source tag) is reserved."""
        matched: list[str] = []
        score = 0.0
        for pattern, weight in self._patterns:
            if pattern.search(content) is not None:
                matched.append(pattern.pattern)
                score = max(score, weight)
        return ScanVerdict(score=score, matched=matched)


@dataclass
class AdmitResult:
    """Outcome of one admission: the frame (or None under reject) + verdict."""

    admitted: bool
    frame: Frame | None
    verdict: ScanVerdict
    trust: Trust
    deduped: bool = False
    semantic_deduped: bool = False
    # Set when dedup.on_hit='keep_longest' demoted a shorter stored near-dup
    # in favour of this (longer) admission.
    superseded_id: int | None = None


def admit(
    content: str,
    *,
    gop: str,
    scanner: InjectionScanner,
    store: FrameStore,
    kg: KnowledgeGraph | None,
    cfg: AppConfig,
    frame_type: FrameType = "I",
    base_id: int | None = None,
    references: Sequence[int] | None = None,
    importance: Importance | None = None,
    source: FrameSource = "user_stated",
    event_time: str | None = None,
    trust: Trust = "normal",
    entity_ids: Sequence[int] = (),
    created_at: str | None = None,
    deduper: SemanticDeduper | None = None,
) -> AdmitResult:
    """Admit one write: scan → (quarantine|reject) → dedup → create → KG-link.

    ``importance=None`` means the per-type default: ``'normal'`` for I/P and
    base-frame inheritance for B (the store implements the inheritance).
    For B-frames ``content`` is the correction *description* and
    ``references`` lists the frames it corrects. ``created_at`` backdates the
    recency-decay anchor (benchmark ingest); ``None`` keeps ingest time.
    """
    verdict = scanner.scan(content, context=source)
    if verdict.score >= cfg.safety.block_threshold:
        if cfg.safety.fail_mode == "reject":
            return AdmitResult(
                admitted=False, frame=None, verdict=verdict, trust=trust
            )
        trust = "quarantined"
    if not is_recall_eligible(trust):
        # An ineligible write must not supersede (keep_longest) an eligible
        # near-duplicate, nor be recorded as a dedup target that would absorb
        # later clean restatements.
        deduper = None

    existing = store.find_frame_by_hash(gop, content)
    if existing is not None:
        # A frame left unindexed by an older build is repaired, not absorbed.
        store.ensure_indexed(existing)
        return AdmitResult(
            admitted=True,
            frame=existing,
            verdict=verdict,
            trust=existing.trust,
            deduped=True,
        )

    # Semantic (paraphrase) dedup — only when the caller passes a deduper (i.e.
    # mode='semantic' AND this lane/frame is eligible). Runs after the exact-hash
    # miss; a candidate at/above the cosine threshold is dropped in favour of the
    # matched same-GOP frame. The probe vector is reused by record() below so an
    # admitted frame is never embedded twice by the deduper.
    probe = deduper.probe(gop, content) if deduper is not None else None
    superseded_id: int | None = None
    if deduper is not None and probe is not None and probe.is_dup and probe.matched_id is not None:
        try:
            matched = store.get_frame(probe.matched_id)
        except KeyError:
            # The matched frame was compacted/deleted since it was cached; drop
            # the stale entry and admit this candidate as the live target.
            deduper.forget(gop, probe.matched_id)
        else:
            if (
                cfg.write.dedup.on_hit == "keep_longest"
                and len(content) > len(matched.content)
            ):
                # The longer restatement supersedes the shorter stored
                # near-dup (longer text tends to carry detail — dates,
                # specifics — the short one drops): demote the old frame —
                # never delete, bitemporal history stays — and admit this.
                store.set_importance(matched.id, "deprecated")
                deduper.forget(gop, matched.id)
                superseded_id = matched.id
            else:
                store.ensure_indexed(matched)
                return AdmitResult(
                    admitted=True,
                    frame=matched,
                    verdict=verdict,
                    trust=matched.trust,
                    semantic_deduped=True,
                )

    if frame_type == "I":
        frame = store.create_i_frame(
            gop,
            content,
            importance=importance if importance is not None else "normal",
            source=source,
            event_time=event_time,
            trust=trust,
            created_at=created_at,
        )
    elif frame_type == "P":
        if base_id is None:
            raise ValueError("P-frame admission requires base_id")
        frame = store.create_p_frame(
            gop,
            content,
            base_id,
            importance=importance if importance is not None else "normal",
            source=source,
            event_time=event_time,
            trust=trust,
            created_at=created_at,
        )
    elif frame_type == "B":
        if base_id is None:
            raise ValueError("B-frame admission requires base_id")
        frame = store.create_b_frame(
            gop,
            content,
            base_id,
            list(references or []),
            importance=importance,
            event_time=event_time,
            trust=trust,
            created_at=created_at,
        )
    else:  # pragma: no cover - FrameType literal exhausted above
        raise ValueError(f"unknown frame_type {frame_type!r}")

    # Register the freshly-admitted frame so later same-GOP candidates can dedup
    # against it (reusing the probe's vector — no re-embed).
    if deduper is not None and probe is not None:
        deduper.record(gop, frame.id, probe.vector)

    if kg is not None and is_recall_eligible(trust):
        for eid in entity_ids:
            kg.link_entity_frame(eid, frame.id)

    return AdmitResult(
        admitted=True,
        frame=frame,
        verdict=verdict,
        trust=trust,
        superseded_id=superseded_id,
    )
