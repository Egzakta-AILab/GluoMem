"""SingleJudge, fake_judge_fn, MetricRegistry, retrieval/QA scoring, and denominators."""

from __future__ import annotations

import pytest

from gluomem.bench.judge import SingleJudge, fake_judge_fn
from gluomem.bench.metrics import (
    NO_EVIDENCE_GOLD,
    RETRIEVER_WITHOUT_RETRIEVAL,
    IncompatibleMetricsError,
    MetricRegistry,
    by_category,
    compute_denominators,
    qa_scores_from_records,
    retrieval_not_applicable,
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
from gluomem.llm.outputs import Verdict

SHA = "d" * 64
BACKBONES = ["haiku", "gpt5-mini", "qwen3"]


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


def _sample(idx: int = 0, category_name: str = "single-hop", **kw: object) -> Sample:
    base: dict[str, object] = {
        "id": f"c1:qa:{idx}",
        "conv_id": "c1",
        "question": "Where does Bob live?",
        "gold": "Seattle",
        "category": 4,
        "category_name": category_name,
        "evidence": ("D1:1", "D2:3"),
    }
    base.update(kw)
    return Sample(**base)  # type: ignore[arg-type]


def _prediction(
    dia_ids: list[str | None] | None = None, answer: str = "Bob lives in Seattle"
) -> Prediction:
    docs = (
        tuple(RetrievedDoc(rank=i, text="t", score=1.0, dia_id=d) for i, d in enumerate(dia_ids))
        if dia_ids is not None
        else ()
    )
    return Prediction(sample_id="c1:qa:0", answer=answer, retrieved=docs)


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


def test_single_judge_score_sink_and_rubric() -> None:
    records: list[JudgeRecord] = []
    seen_args: list[list[object]] = []

    def fn(*args: object) -> Verdict:
        seen_args.append(list(args))
        return Verdict(correct=True, score=1.0, rationale="r")

    judge = SingleJudge(
        "haiku",
        fn,
        model_slug="openrouter/x/haiku",
        dataset_sha=SHA,
        sink=records.append,
    )
    score = judge.score(_sample(), _prediction())
    assert score.value == 1.0 and score.n == 1
    assert score.tag.judge_family == "single"
    assert score.tag.backbone == "haiku"
    assert len(records) == 1 and records[0].model_slug == "openrouter/x/haiku"
    assert len(seen_args[-1]) == 3

    judge_rubric = SingleJudge(
        "haiku",
        fn,
        model_slug="m",
        dataset_sha=SHA,
        rubric_for=lambda s: f"rubric-for-{s.category_name}",
    )
    judge_rubric.score(_sample(), _prediction())
    assert seen_args[-1][3] == "rubric-for-single-hop"

    judge_none = SingleJudge(
        "haiku", fn, model_slug="m", dataset_sha=SHA, rubric_for=lambda s: None
    )
    judge_none.score(_sample(), _prediction())
    assert len(seen_args[-1]) == 3


def test_fake_judge_fn_deterministic(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import gluomem.bench.judge as judge_mod

    monkeypatch.setattr(judge_mod, "_fake_banner_emitted", False)
    assert fake_judge_fn("q", "Seattle", "He moved to seattle last week").correct is True
    assert fake_judge_fn("q", "Seattle", "Boston").correct is False
    assert fake_judge_fn("q", "Seattle", "Bob lives in Seattle", "some rubric").correct is True
    assert capsys.readouterr().err.count("FAKE JUDGE") == 1


def test_metric_registry_aggregate_and_refusals() -> None:
    reg = MetricRegistry()
    agg = reg.aggregate([_score(1.0, n=3), _score(0.0, n=1)])
    assert agg.value == pytest.approx(0.75) and agg.n == 4

    with pytest.raises(IncompatibleMetricsError):
        reg.aggregate([])
    with pytest.raises(IncompatibleMetricsError):
        reg.aggregate([_score(1.0), _score(1.0, _tag(metric_family="precision@5"))])
    with pytest.raises(IncompatibleMetricsError):
        reg.aggregate([_score(1.0), _score(1.0, _tag(judge_family="single", judge_count=1))])
    with pytest.raises(IncompatibleMetricsError):
        reg.aggregate([_score(1.0), _score(1.0, _tag(backbone="haiku"))])
    with pytest.raises(IncompatibleMetricsError):
        reg.aggregate([_score(1.0), _score(1.0, _tag(dataset_sha="e" * 64))])
    with pytest.raises(IncompatibleMetricsError, match="adapted:x@v1"):
        reg.aggregate(
            [
                _score(1.0, _tag(protocol="native:x@v1")),
                _score(1.0, _tag(protocol="adapted:x@v1")),
            ]
        )


def test_retrieval_metrics_math_and_applicability() -> None:
    s = _sample(0)
    p = _prediction(["D1:1", "D9:9", None, "D2:3", "D8:8"])
    scores = {
        sc.tag.metric_family: sc.value
        for sc in retrieval_scores(p, s, ks=[5], dataset_sha=SHA)
    }
    assert scores["precision@5"] == pytest.approx(0.4)
    assert scores["evidence_recall@5"] == pytest.approx(1.0)
    assert scores["evidence_hit@5"] == 1.0

    for sc in retrieval_scores(_prediction(["D1:1"]), s, ks=[5], dataset_sha=SHA):
        assert sc.tag.judge_family == "none" and sc.tag.judge_count == 0

    s_one = _sample(0, evidence=("D2:3",))
    p_late = _prediction(["D9:9", "D8:8", "D7:7", "D6:6", "D5:5", "D2:3"])
    by_family = {
        sc.tag.metric_family: sc.value
        for sc in retrieval_scores(p_late, s_one, ks=[5, 10], dataset_sha=SHA)
    }
    assert by_family["evidence_hit@5"] == 0.0
    assert by_family["evidence_hit@10"] == 1.0

    empty_scores = retrieval_scores(
        Prediction(sample_id=s.id, answer="a"), s, ks=[5, 10], dataset_sha=SHA
    )
    assert all(sc.value == 0.0 and sc.n == 1 for sc in empty_scores)

    no_gold = _sample(1, evidence=())
    assert retrieval_scores(_prediction(["D1:1"]), no_gold, ks=[5], dataset_sha=SHA) == []
    assert retrieval_not_applicable(no_gold, retrieves=True) == NO_EVIDENCE_GOLD
    assert retrieval_not_applicable(s, retrieves=True) is None
    assert retrieval_not_applicable(s, retrieves=False) == RETRIEVER_WITHOUT_RETRIEVAL


def test_qa_scores_from_records_and_by_category() -> None:
    records = [
        *[_record("c1:qa:0", b, True) for b in BACKBONES],
        _record("c1:qa:1", "haiku", True),
        _record("c1:qa:1", "gpt5-mini", False),
        _record("c1:qa:1", "qwen3", True),
    ]
    trio, singles = qa_scores_from_records(
        records,
        backbones=BACKBONES,
        dataset_sha=SHA,
        sample_ids=["c1:qa:0", "c1:qa:1", "c1:qa:2"],
    )
    assert [s.value for s in trio.values()] == [1.0, 0.0, 0.0]
    assert trio["c1:qa:0"].tag.backbone == "gpt5-mini+haiku+qwen3"
    assert [s.value for s in singles["gpt5-mini"].values()] == [1.0, 0.0, 0.0]
    assert [s.value for s in singles["haiku"].values()] == [1.0, 1.0, 0.0]

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
    assert cats["temporal"].value == pytest.approx(0.5) and cats["temporal"].n == 2
    assert cats["multi-hop"].value == 1.0


def test_compute_denominators_partitions_every_sample() -> None:
    samples = {s.id: s for s in (_sample(0), _sample(1), _sample(2), _sample(3, evidence=()))}
    predictions = {
        "c1:qa:0": _prediction(["D1:1"]),
        "c1:qa:1": Prediction(
            sample_id="c1:qa:1", answer="x", abstained=True, abstention={"reason": "no_hits"}
        ),
        "c1:qa:2": Prediction(sample_id="c1:qa:2", answer="", failure="timeout"),
        "c1:qa:3": _prediction(["D1:1"]),
    }
    records = [_record("c1:qa:0", b, True) for b in BACKBONES]
    d = compute_denominators(
        predictions,
        records,
        samples,
        backbones=BACKBONES,
        retrieves=True,
        excluded={"adversarial_category": 7},
    )
    assert d["samples"] == 4
    assert d["excluded_from_pool"] == {"adversarial_category": 7}
    assert d["predictions"] == {
        "answered": 2,
        "abstained": {"no_hits": 1},
        "failed": {"timeout": 1},
    }
    assert d["judge"]["expected_records"] == 12
    assert d["judge"]["missing_by_backbone"] == {b: 3 for b in sorted(BACKBONES)}
    assert d["retrieval"] == {
        "scored": 3,
        "scored_with_empty_retrieval": 2,
        "not_applicable": {NO_EVIDENCE_GOLD: 1},
    }
    full_context = compute_denominators(
        predictions, records, samples, backbones=BACKBONES, retrieves=False
    )
    assert full_context["retrieval"]["not_applicable"] == {RETRIEVER_WITHOUT_RETRIEVAL: 4}
