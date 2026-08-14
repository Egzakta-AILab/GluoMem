"""Judges — unanimity, arity enforcement, record emission, fake determinism."""

from __future__ import annotations

import pytest

from gluomem.bench.judge import SingleJudge, TrioStrictJudge, fake_judge_fn
from gluomem.bench.types import JudgeRecord, Prediction, Sample
from gluomem.llm.outputs import Verdict

SHA = "d" * 64


def _sample() -> Sample:
    return Sample(
        id="c1:qa:0",
        conv_id="c1",
        question="Where does Bob live?",
        gold="Seattle",
        category=4,
        category_name="single-hop",
    )


def _prediction(answer: str = "Bob lives in Seattle") -> Prediction:
    return Prediction(sample_id="c1:qa:0", answer=answer)


def _fixed_judge_fn(correct: bool):
    def fn(question: str, gold: str, prediction: str) -> Verdict:  # noqa: ARG001
        return Verdict(correct=correct, score=1.0 if correct else 0.0, rationale="r")

    return fn


def _single(backbone: str, correct: bool, sink=None) -> SingleJudge:
    return SingleJudge(
        backbone,
        _fixed_judge_fn(correct),
        model_slug=f"openrouter/x/{backbone}",
        dataset_sha=SHA,
        sink=sink,
    )


def test_single_judge_score_and_tag() -> None:
    records: list[JudgeRecord] = []
    judge = _single("haiku", True, records.append)
    score = judge.score(_sample(), _prediction())
    assert score.value == 1.0 and score.n == 1
    assert score.tag.judge_family == "single"
    assert score.tag.backbone == "haiku"
    (rec,) = records
    assert rec.sample_id == "c1:qa:0"
    assert rec.model_slug == "openrouter/x/haiku"
    assert rec.prediction == "Bob lives in Seattle"


def test_trio_strict_unanimity() -> None:
    trio_all = TrioStrictJudge(
        [_single("haiku", True), _single("gpt5-mini", True), _single("qwen3", True)]
    )
    assert trio_all.score(_sample(), _prediction()).value == 1.0
    trio_split = TrioStrictJudge(
        [_single("haiku", True), _single("gpt5-mini", False), _single("qwen3", True)]
    )
    assert trio_split.score(_sample(), _prediction()).value == 0.0


def test_trio_tag_backbone_sorted_join() -> None:
    trio = TrioStrictJudge(
        [_single("qwen3", True), _single("haiku", True), _single("gpt5-mini", True)]
    )
    score = trio.score(_sample(), _prediction())
    assert score.tag.backbone == "gpt5-mini+haiku+qwen3"
    assert score.tag.judge_count == 3


def test_trio_requires_exactly_three() -> None:
    with pytest.raises(ValueError, match="exactly 3"):
        TrioStrictJudge([_single("haiku", True), _single("qwen3", True)])


def test_trio_members_emit_records() -> None:
    records: list[JudgeRecord] = []
    trio = TrioStrictJudge(
        [
            _single("haiku", True, records.append),
            _single("gpt5-mini", False, records.append),
            _single("qwen3", True, records.append),
        ]
    )
    trio.score(_sample(), _prediction())
    assert [r.backbone for r in records] == ["haiku", "gpt5-mini", "qwen3"]
    assert [r.correct for r in records] == [True, False, True]


def test_fake_judge_fn_deterministic(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import gluomem.bench.judge as judge_mod

    monkeypatch.setattr(judge_mod, "_fake_banner_emitted", False)
    v1 = fake_judge_fn("q", "Seattle", "He moved to seattle last week")
    assert v1.correct is True
    v2 = fake_judge_fn("q", "Seattle", "Boston")
    assert v2.correct is False
    # Banner reaches stderr exactly once (module-global once-guard).
    assert capsys.readouterr().err.count("FAKE JUDGE") == 1


# --- rubric-aware judging (LongMemEval per-type grading) ----------------------


def test_single_judge_without_rubric_calls_three_arg() -> None:
    """Without a rubric, judge_fn sees exactly (q, gold, pred)."""
    seen: list[tuple] = []

    def fn(*args) -> Verdict:
        seen.append(args)
        return Verdict(correct=True, score=1.0, rationale="r")

    SingleJudge("haiku", fn, model_slug="m", dataset_sha=SHA).score(
        _sample(), _prediction()
    )
    assert len(seen) == 1 and len(seen[0]) == 3


def test_single_judge_with_rubric_passes_fourth_arg() -> None:
    seen: list[tuple] = []

    def fn(*args) -> Verdict:
        seen.append(args)
        return Verdict(correct=True, score=1.0, rationale="r")

    judge = SingleJudge(
        "haiku",
        fn,
        model_slug="m",
        dataset_sha=SHA,
        rubric_for=lambda s: f"rubric-for-{s.category_name}",
    )
    judge.score(_sample(), _prediction())
    assert seen[0][3] == "rubric-for-single-hop"


def test_rubric_for_returning_none_stays_three_arg() -> None:
    seen: list[tuple] = []

    def fn(*args) -> Verdict:
        seen.append(args)
        return Verdict(correct=True, score=1.0, rationale="r")

    SingleJudge(
        "haiku", fn, model_slug="m", dataset_sha=SHA, rubric_for=lambda s: None
    ).score(_sample(), _prediction())
    assert len(seen[0]) == 3


def test_fake_judge_fn_accepts_rubric_arg() -> None:
    v = fake_judge_fn("q", "Seattle", "Bob lives in Seattle", "some rubric")
    assert v.correct
