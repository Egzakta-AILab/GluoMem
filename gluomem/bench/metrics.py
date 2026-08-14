"""gluomem.bench.metrics — MetricRegistry + the metric functions (dspy-free).

Metric discipline is a RUNTIME invariant, not a convention: ``aggregate``
raises ``IncompatibleMetricsError`` when scores from different metric/judge
families (or different backbones/datasets) would be averaged together.

``qa_scores_from_records`` is the single scoring path shared by the runner
and ``bench rescore`` — both derive headlines from the same JudgeRecord
rows, so the live run and the offline reproduction cannot diverge.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from gluomem.bench.types import JudgeRecord, Prediction, ProtocolTag, Sample, Score


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
                f"{t.metric_family}/{t.judge_family}/{t.backbone}/{t.dataset_sha[:8]}"
                for t in tags
            )
            raise IncompatibleMetricsError(
                f"refusing to aggregate {len(tags)} incompatible protocol "
                f"tags: {fams}"
            )
        total_n = sum(s.n for s in items)
        value = sum(s.value * s.n for s in items) / total_n
        return Score(value=value, tag=items[0].tag, n=total_n)


def _objective_tag(metric_family: str, dataset_sha: str) -> ProtocolTag:
    return ProtocolTag(
        metric_family=metric_family,
        judge_family="none",
        judge_count=0,
        backbone="none",
        dataset_sha=dataset_sha,
    )


def retrieval_scores(
    p: Prediction, s: Sample, *, ks: Sequence[int], dataset_sha: str
) -> list[Score]:
    """Objective retrieval diagnostics against the sample's evidence dia_ids.

    precision@k       = |top-k dia_ids ∩ evidence| / k
    evidence_recall@k = |top-k dia_ids ∩ evidence| / |evidence|
    evidence_hit@k    = 1.0 if any evidence dia_id in top-k else 0.0

    Empty ``p.retrieved`` (full-context baseline, abstention) produces no
    scores — absence is visible in the report, never a silent zero.
    """
    if not p.retrieved or not s.evidence:
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
                    tag=_objective_tag(f"precision@{k}", dataset_sha),
                    n=1,
                ),
                Score(
                    value=inter / len(gold),
                    tag=_objective_tag(f"evidence_recall@{k}", dataset_sha),
                    n=1,
                ),
                Score(
                    value=1.0 if inter else 0.0,
                    tag=_objective_tag(f"evidence_hit@{k}", dataset_sha),
                    n=1,
                ),
            )
        )
    return out


def qa_scores_from_records(
    records: Sequence[JudgeRecord], *, backbones: Sequence[str], dataset_sha: str
) -> tuple[dict[str, Score], dict[str, dict[str, Score]]]:
    """(trio per-sample scores, per-backbone single per-sample scores),
    both keyed by sample_id (sorted) for category splits.

    Trio-strict: a sample is correct iff EVERY backbone's record says
    correct. A sample missing any backbone's record is scored incorrect
    (parse/API failure counts against, denominator unchanged — matches the
    hive-mind protocol). Singles come from the same rows, giving the
    self-judge-inflation measurement for free.
    """
    expected = sorted(backbones)
    by_sample: dict[str, dict[str, JudgeRecord]] = {}
    for rec in records:
        by_sample.setdefault(rec.sample_id, {})[rec.backbone] = rec

    trio_tag = ProtocolTag(
        metric_family="qa_accuracy",
        judge_family="trio_strict",
        judge_count=len(expected),
        backbone="+".join(expected),
        dataset_sha=dataset_sha,
    )
    single_tags = {
        b: ProtocolTag(
            metric_family="qa_accuracy",
            judge_family="single",
            judge_count=1,
            backbone=b,
            dataset_sha=dataset_sha,
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


__all__ = [
    "IncompatibleMetricsError",
    "MetricRegistry",
    "by_category",
    "qa_scores_from_records",
    "retrieval_scores",
]
