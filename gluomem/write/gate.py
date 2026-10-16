"""Single admission point for all memory writes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from gluomem.components import ProviderUnavailableError, require_fake_allowed
from gluomem.config.registry import ModelRegistry, UnknownModelError
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

CLASSIFIER_PROVIDERS: tuple[str, ...] = ("fake", "inprocess")


@dataclass
class ScanVerdict:
    """Outcome of one injection scan: regex score, matched regexes, and optional model verdict."""

    score: float
    matched: list[str] = field(default_factory=list)
    classifier_score: float = 0.0
    classifier_labels: list[str] = field(default_factory=list)


@runtime_checkable
class Scanner(Protocol):
    """Protocol for write-gate prompt-injection / safety scanners."""

    def scan(self, content: str, *, context: str = "") -> ScanVerdict: ...


@runtime_checkable
class InjectionClassifier(Protocol):
    """Protocol for zero-shot multi-label prompt-injection classifiers."""

    def score(self, content: str) -> tuple[float, list[str]]: ...


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
            if pattern.search(content) is not None or (
                needs_folded_pass and pattern.search(folded) is not None
            ):
                matched.append(pattern.pattern)
                score = max(score, weight)
        return ScanVerdict(score=score, matched=matched)


class FakeInjectionClassifier:
    """Deterministic offline test double for the zero-shot injection classifier."""

    def __init__(
        self,
        *,
        attack_labels: Sequence[str],
        review_threshold: float = 0.4,
        scripted_scores: dict[str, float] | None = None,
        default_score: float = 0.0,
    ) -> None:
        self._attack_labels = tuple(attack_labels)
        self._review_threshold = float(review_threshold)
        self._scripted: dict[str, float] = dict(scripted_scores or {})
        self._default_score = float(default_score)

    def score(self, content: str) -> tuple[float, list[str]]:
        val = float(self._scripted.get(content, self._default_score))
        labels = (
            [self._attack_labels[0]]
            if val >= self._review_threshold and self._attack_labels
            else []
        )
        return val, labels


class GlinerInjectionClassifier:
    """Lazy-loaded GLiNER2.5 zero-shot multi-label prompt-injection classifier."""

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        attack_labels: Sequence[str],
        benign_label: str,
        review_threshold: float = 0.4,
    ) -> None:
        self._model_name = model_name
        self._revision = revision
        self._attack_labels = tuple(attack_labels)
        self._benign_label = benign_label
        self._review_threshold = float(review_threshold)
        self._clf: Any = None
        self._schema: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._clf is None:
            from gluomem.extras import require

            require("classifier", f"injection classifier '{self._model_name}'")
            from gliner2.classification import (  # type: ignore[import-not-found]
                ClassificationSchema,
                Classifier,
            )

            self._clf = Classifier.from_pretrained(self._model_name, revision=self._revision)
            labels = [*self._attack_labels, self._benign_label]
            self._schema = ClassificationSchema().multi("risk", labels, min_labels=0)
        return self._clf, self._schema

    def score(self, content: str) -> tuple[float, list[str]]:
        clf, schema = self._load()
        raw = clf.classify(content, schema)
        data = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw)
        probs = data.get("risk", {}).get("probabilities", {})
        prob_map: dict[str, float] = {}
        if isinstance(probs, dict):
            prob_map = {str(k): float(v) for k, v in probs.items()}
        elif isinstance(probs, list):
            for item in probs:
                if isinstance(item, dict) and "label" in item:
                    raw_val = item.get("probability", item.get("score", 0.0))
                    prob_map[str(item["label"])] = (
                        float(raw_val) if isinstance(raw_val, (int, float, str)) else 0.0
                    )
        best = 0.0
        matched: list[str] = []
        for label in self._attack_labels:
            p = prob_map.get(label, 0.0)
            if p > best:
                best = p
            if p >= self._review_threshold:
                matched.append(label)
        return best, matched


class CompositeScanner:
    """Regex scanner composed with an optional zero-shot injection classifier."""

    def __init__(self, regex_scanner: InjectionScanner, classifier: InjectionClassifier) -> None:
        self._regex = regex_scanner
        self._classifier = classifier

    def scan(self, content: str, *, context: str = "") -> ScanVerdict:
        base = self._regex.scan(content, context=context)
        clf_score, clf_labels = self._classifier.score(content)
        return ScanVerdict(
            score=base.score,
            matched=base.matched,
            classifier_score=clf_score,
            classifier_labels=clf_labels,
        )


def make_scanner(cfg: AppConfig) -> Scanner:
    """Build the configured write-gate scanner from ``cfg``."""
    regex_scanner = InjectionScanner(cfg.safety)
    clf_cfg = cfg.safety.scanner.classifier
    if not clf_cfg.enabled:
        return regex_scanner

    name = clf_cfg.model
    try:
        entry = ModelRegistry(cfg).resolve(name)
    except UnknownModelError as exc:
        raise ProviderUnavailableError("classifier", name, str(exc)) from None

    if entry.provider == "fake":
        require_fake_allowed(cfg, f"safety.scanner.classifier.model={name!r} (provider: fake)")
        raw_scores = entry.extra.get("scripted_scores")
        scripted_scores = (
            {str(k): float(v) for k, v in raw_scores.items()}
            if isinstance(raw_scores, dict)
            else None
        )
        default_score = float(entry.extra.get("default_score", 0.0))
        classifier: InjectionClassifier = FakeInjectionClassifier(
            attack_labels=clf_cfg.attack_labels,
            review_threshold=clf_cfg.review_threshold,
            scripted_scores=scripted_scores,
            default_score=default_score,
        )
        return CompositeScanner(regex_scanner, classifier)

    if entry.provider != "inprocess":
        raise ProviderUnavailableError(
            "classifier",
            name,
            f"provider {entry.provider!r} has no classifier implementation; "
            f"supported: {', '.join(CLASSIFIER_PROVIDERS)}",
        )
    from gluomem.extras import require

    require("classifier", f"injection classifier '{entry.provider_model}'")
    classifier = GlinerInjectionClassifier(
        entry.provider_model,
        revision=entry.revision,
        attack_labels=clf_cfg.attack_labels,
        benign_label=clf_cfg.benign_label,
        review_threshold=clf_cfg.review_threshold,
    )
    return CompositeScanner(regex_scanner, classifier)


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
    conflicts_with: int | None = None


def admit(
    content: str,
    *,
    gop: str,
    scanner: Scanner,
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
    verdict: ScanVerdict | None = None,
) -> AdmitResult:
    """Admit one write: scan, dedup, create frame, and link KG entities."""
    s = resolve_scope(scope, default=store.scope)
    verdict = verdict if verdict is not None else scanner.scan(content, context=source)
    if verdict.score >= cfg.safety.block_threshold:
        if cfg.safety.fail_mode == "reject":
            return AdmitResult(admitted=False, frame=None, verdict=verdict, trust=trust)
        trust = "quarantined"
    if not is_recall_eligible(trust, status=status, expires_at=expires_at):
        deduper = None

    def link_graph(frame: Frame) -> Frame | None:
        from gluomem.frames.generation import lock_scope_generation

        with store.db.transaction():
            lock_scope_generation(store._backend, s)
            current = store.get_frame(frame.id, scope=s)
            if current.status != frame.status or current.content_hash != frame.content_hash:
                return None
            if kg is not None and is_recall_eligible(current):
                try:
                    for eid in entity_ids:
                        kg.link_entity_frame(eid, current.id, scope=s)
                    for rid in relation_ids:
                        kg.link_relation_frame(rid, current.id, scope=s)
                except KeyError:
                    # Reentrant erasure may also delete the graph target. Do not
                    # unwind its savepoint; unrelated missing IDs still raise.
                    current = store.get_frame(frame.id, scope=s)
                    if current.status != frame.status or current.content_hash != frame.content_hash:
                        return None
                    raise
            current = store.get_frame(frame.id, scope=s)
            if current.status != frame.status or current.content_hash != frame.content_hash:
                return None
            return current

    def reuse_frame(
        frame: Frame, *, semantic: bool = False, prepared: _PreparedIndex | None = None
    ) -> AdmitResult:
        # The caller holds generation protection through repair and graph publication.
        store.ensure_indexed(frame, prepared=prepared)
        current = store.get_frame(frame.id, scope=s)
        if current.status in ("active", "candidate") and current.content_hash == frame.content_hash:
            current = link_graph(current)
            if current is not None:
                return AdmitResult(
                    admitted=True,
                    frame=current,
                    verdict=verdict,
                    trust=current.trust,
                    deduped=not semantic,
                    semantic_deduped=semantic,
                )
        # Reject without unwinding a successful reentrant erasure's savepoint.
        return AdmitResult(admitted=False, frame=None, verdict=verdict, trust=trust)

    existing = store.find_frame_by_hash(None if scope_wide else gop, content, scope=s)
    if existing is not None:
        from gluomem.frames.generation import bump_scope_generation, lock_scope_generation

        # Standalone admission owns this transaction too: flags, repair and graph
        # projections must not be separated by a retraction's generation lock.
        with store.db.transaction():
            lock_scope_generation(store._backend, s)
            current = store.get_frame(existing.id, scope=s)
            if (
                current.status not in ("active", "candidate")
                or current.content_hash != existing.content_hash
            ):
                raise ValueError("Memory changed before admission; retry ingestion")
            existing = current
            if (
                trust != "quarantined"
                and existing.trust != "quarantined"
                and (verified_by_user or pinned)
            ):
                store._update(
                    existing.id,
                    s,
                    status="active",
                    pinned=int(existing.pinned or pinned),
                    verified_by_user=int(existing.verified_by_user or verified_by_user),
                    expires_at=None if pinned else existing.expires_at,
                )
                bump_scope_generation(store._backend, s)
            return reuse_frame(existing, prepared=prepared)

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
                from gluomem.frames.generation import lock_scope_generation

                with store.db.transaction():
                    lock_scope_generation(store._backend, s)
                    current = store.get_frame(matched.id, scope=s)
                    if (
                        current.status not in ("active", "candidate")
                        or current.content_hash != matched.content_hash
                    ):
                        raise ValueError("Memory changed before admission; retry ingestion")
                    return reuse_frame(current, semantic=True)

    clf_cfg = cfg.safety.scanner.classifier
    if (
        clf_cfg.enabled
        and frame_type != "B"
        and source in clf_cfg.reviewed_sources
        and status == "active"
        and not (pinned or verified_by_user)
        and is_recall_eligible(trust, status=status, expires_at=expires_at)
        and verdict.classifier_score >= clf_cfg.review_threshold
    ):
        status = "candidate"

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

    current = link_graph(frame)
    if current is None:
        return AdmitResult(admitted=False, frame=None, verdict=verdict, trust=trust)

    return AdmitResult(
        admitted=True,
        frame=current,
        verdict=verdict,
        trust=trust,
        superseded_id=superseded_id,
    )
