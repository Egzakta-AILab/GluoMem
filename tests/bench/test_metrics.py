"""MetricRegistry — family refusal, metric math, per-category splits."""

from __future__ import annotations

import pytest

from gluomem.bench.metrics import (
    IncompatibleMetricsError,
    MetricRegistry,
    by_category,
    qa_scores_from_records,
    retrieval_scores,
)
from gluomem.bench.types import (
    JudgeRecord,
    Prediction,
    ProtocolTag,
    RetrievedDoc,
    Sample,
    Score,
)

SHA = "d" * 64


def _tag(**kw: object) -> ProtocolTag:
    base: dict[str, object] = {
        "metric_family": "qa_accuracy",
        "judge_family": "trio_strict",
        "judge_count": 3,
        "backbone": "gpt5-mini+haiku+qwen3",
        "dataset_sha": SHA,
    }
    base.update(kw)
    return ProtocolTag(**base)  # type: ignore[arg-type]


def _score(value: float, tag: ProtocolTag | None = None, n: int = 1) -> Score:
    return Score(value=value, tag=tag or _tag(), n=n)


def _sample(idx: int, category_name: str = "single-hop", **kw: object) -> Sample:
    base: dict[str, object] = {
        "id": f"c1:qa:{idx}",
        "conv_id": "c1",
        "question": "q",
        "gold": "g",
        "category": 4,
        "category_name": category_name,
        "evidence": ("D1:1", "D2:3"),
    }
    base.update(kw)
    return Sample(**base)  # type: ignore[arg-type]


def _record(sample_id: str, backbone: str, correct: bool) -> JudgeRecord:
    return JudgeRecord(
        sample_id=sample_id,
        backbone=backbone,
        model_slug=f"openrouter/x/{backbone}",
        correct=correct,
        score=1.0 if correct else 0.0,
        rationale="r",
        question="q",
        gold="g",
        prediction="p",
    )


# --- aggregate -------------------------------------------------------------


def test_aggregate_weighted_mean() -> None:
    agg = MetricRegistry().aggregate([_score(1.0, n=3), _score(0.0, n=1)])
    assert agg.value == pytest.approx(0.75)
    assert agg.n == 4


def test_aggregate_refuses_mixed_metric_family() -> None:
    with pytest.raises(IncompatibleMetricsError):
        MetricRegistry().aggregate(
            [_score(1.0), _score(1.0, _tag(metric_family="precision@5"))]
        )


def test_aggregate_refuses_mixed_judge_family() -> None:
    with pytest.raises(IncompatibleMetricsError):
        MetricRegistry().aggregate(
            [_score(1.0), _score(1.0, _tag(judge_family="single", judge_count=1))]
        )


def test_aggregate_refuses_mixed_backbone() -> None:
    with pytest.raises(IncompatibleMetricsError):
        MetricRegistry().aggregate([_score(1.0), _score(1.0, _tag(backbone="haiku"))])


def test_aggregate_refuses_mixed_dataset_sha() -> None:
    with pytest.raises(IncompatibleMetricsError):
        MetricRegistry().aggregate(
            [_score(1.0), _score(1.0, _tag(dataset_sha="e" * 64))]
        )


def test_aggregate_empty_raises() -> None:
    with pytest.raises(IncompatibleMetricsError):
        MetricRegistry().aggregate([])


# --- retrieval metrics -----------------------------------------------------


def _prediction(dia_ids: list[str | None]) -> Prediction:
    docs = tuple(
        RetrievedDoc(rank=i, text="t", score=1.0, dia_id=d)
        for i, d in enumerate(dia_ids)
    )
    return Prediction(sample_id="c1:qa:0", answer="a", retrieved=docs)


def test_retrieval_metric_math() -> None:
    s = _sample(0)  # evidence = {D1:1, D2:3}
    p = _prediction(["D1:1", "D9:9", None, "D2:3", "D8:8"])
    scores = {
        sc.tag.metric_family: sc.value
        for sc in retrieval_scores(p, s, ks=[5], dataset_sha=SHA)
    }
    # Both evidence dia_ids inside top-5: precision 2/5, recall 2/2, hit 1.
    assert scores["precision@5"] == pytest.approx(0.4)
    assert scores["evidence_recall@5"] == pytest.approx(1.0)
    assert scores["evidence_hit@5"] == 1.0


def test_precision_at_k_ceiling_demonstrated() -> None:
    """LoCoMo evidence sets are 1-3 dia_ids: even a PERFECT top-5 caps
    precision@5 at |evidence|/5 — why the CI gate uses evidence_hit@5."""
    s = _sample(0, evidence=("D1:1",))
    p = _prediction(["D1:1", "D2:2", "D3:3", "D4:4", "D5:5"])
    scores = {
        sc.tag.metric_family: sc.value
        for sc in retrieval_scores(p, s, ks=[5], dataset_sha=SHA)
    }
    assert scores["precision@5"] == pytest.approx(0.2)  # perfect retrieval!
    assert scores["evidence_hit@5"] == 1.0


def test_retrieval_scores_multiple_ks() -> None:
    s = _sample(0, evidence=("D2:3",))
    p = _prediction(["D9:9", "D8:8", "D7:7", "D6:6", "D5:5", "D2:3"])
    by_family = {
        sc.tag.metric_family: sc.value
        for sc in retrieval_scores(p, s, ks=[5, 10], dataset_sha=SHA)
    }
    assert by_family["evidence_hit@5"] == 0.0
    assert by_family["evidence_hit@10"] == 1.0


def test_retrieval_scores_empty_retrieved_or_evidence() -> None:
    s = _sample(0)
    assert retrieval_scores(
        Prediction(sample_id=s.id, answer="a"), s, ks=[5], dataset_sha=SHA
    ) == []
    no_gold = _sample(1, evidence=())
    assert (
        retrieval_scores(_prediction(["D1:1"]), no_gold, ks=[5], dataset_sha=SHA)
        == []
    )


def test_retrieval_tags_are_objective() -> None:
    s = _sample(0)
    for sc in retrieval_scores(_prediction(["D1:1"]), s, ks=[5], dataset_sha=SHA):
        assert sc.tag.judge_family == "none"
        assert sc.tag.judge_count == 0
        assert sc.tag.backbone == "none"


# --- qa scores from judge records -------------------------------------------


BACKBONES = ["haiku", "gpt5-mini", "qwen3"]


def test_trio_strict_unanimity() -> None:
    records = [
        _record("c1:qa:0", b, True) for b in BACKBONES
    ] + [
        _record("c1:qa:1", "haiku", True),
        _record("c1:qa:1", "gpt5-mini", False),
        _record("c1:qa:1", "qwen3", True),
    ]
    trio, singles = qa_scores_from_records(
        records, backbones=BACKBONES, dataset_sha=SHA
    )
    assert [s.value for s in trio.values()] == [1.0, 0.0]
    assert trio["c1:qa:0"].tag.backbone == "gpt5-mini+haiku+qwen3"
    # Singles derived from the SAME rows: gpt5-mini disagrees on qa:1.
    assert [s.value for s in singles["gpt5-mini"].values()] == [1.0, 0.0]
    assert [s.value for s in singles["haiku"].values()] == [1.0, 1.0]


def test_missing_backbone_record_scores_incorrect() -> None:
    """A sample missing one backbone's verdict is not-correct; denominator
    stays (parse/API failure counts against, matching hive-mind)."""
    records = [
        _record("c1:qa:0", "haiku", True),
        _record("c1:qa:0", "qwen3", True),
        # gpt5-mini row absent
    ]
    trio, singles = qa_scores_from_records(
        records, backbones=BACKBONES, dataset_sha=SHA
    )
    assert [s.value for s in trio.values()] == [0.0]
    assert [s.value for s in singles["gpt5-mini"].values()] == [0.0]
    assert len(singles["haiku"]) == 1


# --- per-category ------------------------------------------------------------


def test_by_category_splits_and_aggregates() -> None:
    samples = {
        "c1:qa:0": _sample(0, "temporal"),
        "c1:qa:1": _sample(1, "temporal"),
        "c1:qa:2": _sample(2, "multi-hop"),
    }
    per_sample = {
        "c1:qa:0": _score(1.0),
        "c1:qa:1": _score(0.0),
        "c1:qa:2": _score(1.0),
    }
    cats = by_category(per_sample, samples)
    assert cats["temporal"].value == pytest.approx(0.5)
    assert cats["temporal"].n == 2
    assert cats["multi-hop"].value == 1.0
