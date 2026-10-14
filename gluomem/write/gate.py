"""Single admission point for all memory writes."""

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
from gluomem.search.multilingual import fold_to_ascii
from gluomem.storage.scope import FrameStatus, MemoryKind, MemoryScope, resolve_scope

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gluomem.config.app_config import AppConfig, SafetyConfig
    from gluomem.frames.store import FrameStore, _PreparedIndex
    from gluomem.kg.graph import KnowledgeGraph
    from gluomem.write.dedup import SemanticDeduper


@dataclass
class ScanVerdict:
    """Outcome of one injection scan: score in [0, 1] and the regexes that hit."""

    score: float
    matched: list[str] = field(default_factory=list)


class InjectionScanner:
    """Regex injection scanner over ``safety.scanner.patterns``."""

    def __init__(self, cfg: SafetyConfig) -> None:
        self._patterns: list[tuple[re.Pattern[str], float]] = [
            (re.compile(p.regex), p.weight) for p in cfg.scanner.patterns
        ]

    def scan(self, content: str, *, context: str = "") -> ScanVerdict:  # noqa: ARG002
        matched: list[str] = []
        score = 0.0
        folded = fold_to_ascii(content)
        needs_folded_pass = folded != content.lower()
        for pattern, weight in self._patterns:
            if (
                pattern.search(content) is not None
                or (needs_folded_pass and pattern.search(folded) is not None)
            ):
                matched.append(pattern.pattern)
                score = max(score, weight)
        return ScanVerdict(score=score, matched=matched)


@dataclass
class AdmitResult:
    """Outcome of one admission: the frame (or None under reject) and verdict."""

    admitted: bool
    frame: Frame | None
    verdict: ScanVerdict
    trust: Trust
    deduped: bool = False
    semantic_deduped: bool = False
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
    relation_ids: Sequence[int] = (),
    created_at: str | None = None,
    deduper: SemanticDeduper | None = None,
    scope: MemoryScope | None = None,
    kind: MemoryKind = "fact",
    status: FrameStatus = "active",
    confidence: float = 1.0,
    expires_at: str | None = None,
    pinned: bool = False,
    verified_by_user: bool = False,
    scope_wide: bool = False,
    prepared: _PreparedIndex | None = None,
) -> AdmitResult:
    """Admit one write: scan, dedup, create frame, and link KG entities."""
    s = resolve_scope(scope, default=store.scope)
    verdict = scanner.scan(content, context=source)
    if verdict.score >= cfg.safety.block_threshold:
        if cfg.safety.fail_mode == "reject":
            return AdmitResult(admitted=False, frame=None, verdict=verdict, trust=trust)
        trust = "quarantined"
    if not is_recall_eligible(trust, status=status, expires_at=expires_at):
        deduper = None

    def link_graph(frame: Frame) -> None:
        if kg is not None and is_recall_eligible(frame):
            for eid in entity_ids:
                kg.link_entity_frame(eid, frame.id, scope=s)
            for rid in relation_ids:
                kg.link_relation_frame(rid, frame.id, scope=s)

    existing = store.find_frame_by_hash(None if scope_wide else gop, content, scope=s)
    if existing is not None:
        if (
            trust != "quarantined"
            and existing.trust != "quarantined"
            and (verified_by_user or pinned)
        ):
            from gluomem.frames.generation import bump_scope_generation

            with store.db.transaction():
                store._update(
                    existing.id,
                    s,
                    status="active",
                    pinned=int(existing.pinned or pinned),
                    verified_by_user=int(existing.verified_by_user or verified_by_user),
                    expires_at=None if pinned else existing.expires_at,
                )
                bump_scope_generation(store._backend, s)
            existing = store.get_frame(existing.id, scope=s)
        store.ensure_indexed(existing, prepared=prepared)
        link_graph(existing)
        return AdmitResult(
            admitted=True,
            frame=existing,
            verdict=verdict,
            trust=existing.trust,
            deduped=True,
        )

    probe = deduper.probe(gop, content, scope=s) if deduper is not None else None
    superseded_id: int | None = None
    if deduper is not None and probe is not None and probe.is_dup and probe.matched_id is not None:
        try:
            matched = store.get_frame(probe.matched_id, scope=s)
        except KeyError:
            deduper.forget(gop, probe.matched_id, scope=s)
        else:
            if cfg.write.dedup.on_hit == "keep_longest" and len(content) > len(matched.content):
                store.set_importance(matched.id, "deprecated", scope=s)
                deduper.forget(gop, matched.id, scope=s)
                superseded_id = matched.id
            else:
                store.ensure_indexed(matched)
                link_graph(matched)
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
            prepared=prepared,
            scope_wide=scope_wide,
            importance=importance if importance is not None else "normal",
            source=source,
            event_time=event_time,
            trust=trust,
            created_at=created_at,
            scope=s,
            kind=kind,
            status=status,
            confidence=confidence,
            expires_at=expires_at,
            pinned=pinned,
            verified_by_user=verified_by_user,
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
            scope=s,
            kind=kind,
            status=status,
            confidence=confidence,
            expires_at=expires_at,
            pinned=pinned,
            verified_by_user=verified_by_user,
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
            scope=s,
            kind=kind,
            status=status,
            confidence=confidence,
            expires_at=expires_at,
            pinned=pinned,
            verified_by_user=verified_by_user,
        )
    else:  # pragma: no cover
        raise ValueError(f"unknown frame_type {frame_type!r}")

    if deduper is not None and probe is not None:
        deduper.record(gop, frame.id, probe.vector, scope=s)

    link_graph(frame)

    return AdmitResult(
        admitted=True,
        frame=frame,
        verdict=verdict,
        trust=trust,
        superseded_id=superseded_id,
    )
