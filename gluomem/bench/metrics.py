"""gluomem.bench.metrics — MetricRegistry + the metric functions (dspy-free).

Metric discipline is a RUNTIME invariant, not a convention: ``aggregate``
raises ``IncompatibleMetricsError`` when scores from different metric/judge
families (or different backbones/datasets) would be averaged together.

``qa_scores_from_records`` is the single scoring path shared by the runner
and ``bench rescore`` — both derive headlines from the same JudgeRecord
rows, so the live run and the offline reproduction cannot diverge.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from gluomem.bench.types import (
    UNREGISTERED,
    JudgeRecord,
    Prediction,
    ProtocolTag,
    Sample,
    Score,
)


class IncompatibleMetricsError(ValueError):
    """Raised when scores with different ProtocolTags would be aggregated."""


class MetricRegistry:
    """Aggregates per-sample scores into one Score; refuses mixed families."""

    def aggregate(self, scores: Iterable[Score]) -> Score:
        items = list(scores)
        if not items:
            raise IncompatibleMetricsError("cannot aggregate zero scores")
        tags = {s.tag for s in items}
        if len(tags) > 1:
            fams = sorted(
                f"{t.protocol}/{t.metric_family}/{t.judge_family}/{t.backbone}/"
                f"{t.dataset_sha[:8]}"
                for t in tags
            )
            raise IncompatibleMetricsError(
                f"refusing to aggregate {len(tags)} incompatible protocol "
                f"tags: {fams}"
            )
        total_n = sum(s.n for s in items)
        value = sum(s.value * s.n for s in items) / total_n
        return Score(value=value, tag=items[0].tag, n=total_n)


def objective_tag(
    metric_family: str, dataset_sha: str, protocol: str = UNREGISTERED
) -> ProtocolTag:
    return ProtocolTag(
        metric_family=metric_family,
        judge_family="none",
        judge_count=0,
        backbone="none",
        dataset_sha=dataset_sha,
        protocol=protocol,
    )


NO_EVIDENCE_GOLD = "no_evidence_gold"
RETRIEVER_WITHOUT_RETRIEVAL = "retriever_without_retrieval"


def retrieval_not_applicable(s: Sample, *, retrieves: bool) -> str | None:
    """Why retrieval metrics do not apply to ``s``, or None when they do.

    Only two reasons qualify: the sample has no evidence gold (e.g. an
    abstention question), or the retriever never retrieves (full context).
    An empty result from a retrieving system is a failure, not a reason.
    """
    if not retrieves:
        return RETRIEVER_WITHOUT_RETRIEVAL
    if not s.evidence:
        return NO_EVIDENCE_GOLD
    return None


def retrieval_scores(
    p: Prediction,
    s: Sample,
    *,
    ks: Sequence[int],
    dataset_sha: str,
    protocol: str = UNREGISTERED,
) -> list[Score]:
    """Objective retrieval diagnostics against the sample's evidence dia_ids.

    precision@k       = |top-k dia_ids ∩ evidence| / k
    evidence_recall@k = |top-k dia_ids ∩ evidence| / |evidence|
    evidence_hit@k    = 1.0 if any evidence dia_id in top-k else 0.0

    An answerable sample (non-empty evidence) always scores: empty
    ``p.retrieved`` (no hits, abstention, answer failure) contributes zero.
    A sample without evidence gold returns no scores; callers count it under
    ``retrieval_not_applicable`` instead.
    """
    if not s.evidence:
        return []
    gold = set(s.evidence)
    out: list[Score] = []
    for k in ks:
        top = {d.dia_id for d in p.retrieved[:k] if d.dia_id is not None}
        inter = len(top & gold)
        out.extend(
            (
                Score(
                    value=inter / k,
                    tag=objective_tag(f"precision@{k}", dataset_sha, protocol),
                    n=1,
                ),
                Score(
                    value=inter / len(gold),
                    tag=objective_tag(f"evidence_recall@{k}", dataset_sha, protocol),
                    n=1,
                ),
                Score(
                    value=1.0 if inter else 0.0,
                    tag=objective_tag(f"evidence_hit@{k}", dataset_sha, protocol),
                    n=1,
                ),
            )
        )
    return out


def qa_scores_from_records(
    records: Sequence[JudgeRecord],
    *,
    backbones: Sequence[str],
    dataset_sha: str,
    sample_ids: Iterable[str] = (),
    protocol: str = UNREGISTERED,
) -> tuple[dict[str, Score], dict[str, dict[str, Score]]]:
    """(trio per-sample scores, per-backbone single per-sample scores),
    both keyed by sample_id (sorted) for category splits.

    Trio-strict: a sample is correct iff EVERY backbone's record says
    correct. A sample missing any backbone's record is scored incorrect
    (parse/API failure counts against, denominator unchanged — matches the
    hive-mind protocol). ``sample_ids`` names the full denominator, so a
    sample whose every verdict failed still scores zero instead of
    vanishing. Singles come from the same rows, giving the
    self-judge-inflation measurement for free.
    """
    expected = sorted(backbones)
    by_sample: dict[str, dict[str, JudgeRecord]] = {sid: {} for sid in sample_ids}
    for rec in records:
        by_sample.setdefault(rec.sample_id, {})[rec.backbone] = rec

    trio_tag = ProtocolTag(
        metric_family="qa_accuracy",
        judge_family="trio_strict",
        judge_count=len(expected),
        backbone="+".join(expected),
        dataset_sha=dataset_sha,
        protocol=protocol,
    )
    single_tags = {
        b: ProtocolTag(
            metric_family="qa_accuracy",
            judge_family="single",
            judge_count=1,
            backbone=b,
            dataset_sha=dataset_sha,
            protocol=protocol,
        )
        for b in expected
    }

    trio: dict[str, Score] = {}
    singles: dict[str, dict[str, Score]] = {b: {} for b in expected}
    for sample_id, verdicts in sorted(by_sample.items()):
        unanimous = all(
            b in verdicts and verdicts[b].correct for b in expected
        )
        trio[sample_id] = Score(
            value=1.0 if unanimous else 0.0, tag=trio_tag, n=1
        )
        for b in expected:
            rec = verdicts.get(b)
            correct = rec is not None and rec.correct
            singles[b][sample_id] = Score(
                value=1.0 if correct else 0.0, tag=single_tags[b], n=1
            )
    return trio, singles


def by_category(
    per_sample: Mapping[str, Score], samples: Mapping[str, Sample]
) -> dict[str, Score]:
    """Aggregate per-sample scores into one Score per category_name."""
    registry = MetricRegistry()
    buckets: dict[str, list[Score]] = {}
    for sample_id, score in per_sample.items():
        sample = samples[sample_id]
        buckets.setdefault(sample.category_name, []).append(score)
    return {name: registry.aggregate(group) for name, group in buckets.items()}


def compute_denominators(
    predictions: Mapping[str, Prediction],
    records: Sequence[JudgeRecord],
    samples: Mapping[str, Sample],
    *,
    backbones: Sequence[str],
    retrieves: bool,
    excluded: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Every count behind the headline numbers, so none is implicit.

    ``predictions`` partitions into answered, abstained (by reason) and failed
    (by kind); each failed or abstained sample still sits in the QA and
    retrieval denominators. ``excluded`` counts rows the adapter removed from
    the sampling pool before any prediction.
    """
    failed: Counter[str] = Counter()
    abstained: Counter[str] = Counter()
    answered = 0
    for p in predictions.values():
        if p.failure is not None:
            failed[p.failure] += 1
        elif p.abstained:
            reason = (p.abstention or {}).get("reason", "unspecified")
            abstained[str(reason)] += 1
        else:
            answered += 1

    have = Counter(rec.backbone for rec in records)
    n = len(predictions)
    missing = {b: n - have.get(b, 0) for b in sorted(backbones)}

    not_applicable: Counter[str] = Counter()
    scored = zero_retrieved = 0
    for sid, p in predictions.items():
        reason = retrieval_not_applicable(samples[sid], retrieves=retrieves)
        if reason is not None:
            not_applicable[reason] += 1
            continue
        scored += 1
        if not p.retrieved:
            zero_retrieved += 1

    return {
        "samples": n,
        "excluded_from_pool": dict(sorted((excluded or {}).items())),
        "predictions": {
            "answered": answered,
            "abstained": dict(sorted(abstained.items())),
            "failed": dict(sorted(failed.items())),
        },
        "judge": {
            "expected_records": n * len(backbones),
            "records": len(records),
            "missing_by_backbone": missing,
        },
        "retrieval": {
            "scored": scored,
            "scored_with_empty_retrieval": zero_retrieved,
            "not_applicable": dict(sorted(not_applicable.items())),
        },
    }


__all__ = [
    "NO_EVIDENCE_GOLD",
    "RETRIEVER_WITHOUT_RETRIEVAL",
    "IncompatibleMetricsError",
    "MetricRegistry",
    "by_category",
    "compute_denominators",
    "objective_tag",
    "qa_scores_from_records",
    "retrieval_not_applicable",
    "retrieval_scores",
]
