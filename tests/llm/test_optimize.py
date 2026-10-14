"""Optimize seam tests — trainset loading, metrics, fake-teleprompter compile.

MIPROv2 and GEPA are NEVER invoked here; a fake teleprompter proves the
offline flow (examples -> compile -> save_artifact with manifest) end-to-end,
and the extraction metric is the frozen source-backed evaluator.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import dspy
import pytest

from gluomem.config.app_config import load_config
from gluomem.llm.extraction_eval import SealedSplitError
from gluomem.llm.optimize import (
    check_trainset,
    fixture_examples,
    load_trainset,
    main,
    metric_for,
    optimize_task,
)
from gluomem.llm.outputs import ContradictionVerdict, ExtractedEvent, ExtractedFact

_FACT_ROW = {
    "text": "Alice lives in Boston. It rains.",
    "facts": [{"text": "Alice lives in Boston.", "confidence": 1.0}],
}


def _trainset(tmp_path: Path, rows: list[dict], name: str = "fact.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_load_trainset_marks_inputs(tmp_path: Path) -> None:
    path = _trainset(tmp_path, [_FACT_ROW])
    examples = load_trainset(path, "fact_extraction")
    assert len(examples) == 1
    assert set(examples[0].inputs().keys()) == {"text"}
    assert examples[0].facts == _FACT_ROW["facts"]


def test_load_trainset_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing input fields"):
        load_trainset(_trainset(tmp_path, [{"facts": []}]), "fact_extraction")
    with pytest.raises(ValueError, match="invalid JSON"):
        path = tmp_path / "bad.jsonl"
        path.write_text("{not json\n", encoding="utf-8")
        load_trainset(path, "fact_extraction")
    with pytest.raises(ValueError, match="empty trainset"):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        load_trainset(path, "fact_extraction")


def _ana_example() -> dspy.Example:
    return next(
        ex for ex in fixture_examples("fact_extraction", splits=("dev",))
        if ex.case_id == "en-ana-move"
    )


def test_unsupported_but_valid_facts_score_zero_not_half() -> None:
    # The structural metric gave any parsed output 0.5; the evidence metric
    # gives an unsupported fact nothing and a supported paraphrase full credit.
    metric = metric_for("fact_extraction")
    example = _ana_example()
    unsupported = dspy.Prediction(facts=[ExtractedFact(text="Ana lives in Paris.")])
    assert metric(example, unsupported) == 0.0
    paraphrase = dspy.Prediction(
        facts=[
            ExtractedFact(text="Ana relocated to Novi Sad."),
            ExtractedFact(text="Ana's brother Marko is based in Belgrade."),
            ExtractedFact(text="Ana doesn't have a car."),
            ExtractedFact(text="Ana is considering getting a bike."),
        ]
    )
    assert metric(example, paraphrase) == 1.0


def test_gepa_call_returns_score_with_feedback() -> None:
    metric = metric_for("fact_extraction")
    pred = dspy.Prediction(facts=[ExtractedFact(text="Ana bought a bicycle.")])
    out = metric(_ana_example(), pred, None, "extract", None)
    assert isinstance(out, dspy.Prediction)
    assert out.score == metric(_ana_example(), pred)
    assert "plan or possibility as done" in out.feedback
    assert "Missed:" in out.feedback


def test_event_metric_checks_dates() -> None:
    metric = metric_for("event_extraction")
    example = fixture_examples("event_extraction", splits=("dev",))[0]
    assert set(example.inputs().keys()) == {"source_text", "reference_date"}
    good = dspy.Prediction(
        events=[
            ExtractedEvent(text="Project kickoff", event_date=datetime(2026, 2, 3)),
            ExtractedEvent(text="Client demo", event_date=datetime(2026, 3, 24)),
        ]
    )
    stale = dspy.Prediction(
        events=[
            ExtractedEvent(text="Project kickoff", event_date=datetime(2026, 2, 3)),
            ExtractedEvent(text="Client demo", event_date=datetime(2026, 3, 10)),
        ]
    )
    assert metric(example, good) == 1.0
    assert metric(example, stale) == 0.5


def test_structural_metric_still_covers_other_tasks(tmp_path: Path) -> None:
    row = {"new_statement": "a", "stored_statement": "b",
           "verdict": {"is_contradiction": False, "description": "", "confidence": 1.0}}
    example = load_trainset(_trainset(tmp_path, [row]), "contradiction")[0]
    metric = metric_for("contradiction")
    exact = dspy.Prediction(verdict=ContradictionVerdict(is_contradiction=False))
    assert metric(example, exact) == 1.0
    wrong = dspy.Prediction(verdict=ContradictionVerdict(is_contradiction=True))
    assert metric(example, wrong) == 0.5


def test_fixture_examples_carry_inputs_and_case_id_only() -> None:
    for task in ("fact_extraction", "event_extraction"):
        for ex in fixture_examples(task, splits=("train", "dev")):
            assert set(ex.keys()) - set(ex.inputs().keys()) == {"case_id"}


def test_sealed_cases_never_reach_the_optimizer(tmp_path: Path) -> None:
    with pytest.raises(SealedSplitError):
        fixture_examples("fact_extraction", splits=("train", "sealed_test"))
    sealed_row = {"text": "Jelena leads payments.", "case_id": "en-jelena-role"}
    examples = load_trainset(_trainset(tmp_path, [sealed_row]), "fact_extraction")
    with pytest.raises(SealedSplitError):
        check_trainset("fact_extraction", examples)
    sealed_example = dspy.Example(text="x", case_id="en-jelena-role").with_inputs("text")
    with pytest.raises(SealedSplitError):
        metric_for("fact_extraction")(sealed_example, dspy.Prediction(facts=[]))


def test_unannotated_extraction_trainset_is_refused(tmp_path: Path) -> None:
    examples = load_trainset(_trainset(tmp_path, [_FACT_ROW]), "fact_extraction")
    with pytest.raises(ValueError, match="annotated extraction fixture"):
        check_trainset("fact_extraction", examples)
    events = fixture_examples("event_extraction", splits=("dev",))
    with pytest.raises(ValueError, match="is a 'event_extraction' case"):
        check_trainset("fact_extraction", events)


def test_optimize_task_with_fake_teleprompter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # router.bind preflights the key; the fake teleprompter never calls a provider.
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-placeholder")
    cfg = load_config(env_file=tmp_path / "absent.env").with_overrides(
        {"dspy.artifacts.dir": str(tmp_path / "artifacts")}
    )
    trainset_path = fixture_examples("fact_extraction", splits=("train",))
    seen: dict = {}

    class FakeTeleprompter:
        def __init__(self, metric) -> None:
            self.metric = metric

        def compile(self, student, *, trainset):
            seen["student"] = student
            seen["trainset"] = trainset
            seen["empty_score"] = self.metric(trainset[0], dspy.Prediction(facts=[]))
            return student  # "compiled" = unchanged; artifact flow is the point

    target = optimize_task(
        "fact_extraction",
        cfg,
        trainset_path,
        teleprompter_factory=lambda metric: FakeTeleprompter(metric),
    )
    assert (target / "program.pkl").exists()
    manifest = json.loads((target / "manifest.gluomem.json").read_text())
    assert manifest["model_ref"] == "deepseek-flash"
    assert [ex.case_id for ex in seen["trainset"]] == ["en-plant-migration"]
    assert seen["empty_score"] == 0.0  # over-abstention on a fact-bearing source
    assert isinstance(seen["student"], dspy.Predict)


def test_missing_key_fails_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Point load_config at a nonexistent .env so a developer's real repo .env
    # can't repopulate the key this test removes.
    cfg = load_config(env_file=tmp_path / "absent.env")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        optimize_task("fact_extraction", cfg, fixture_examples("fact_extraction"))


def test_main_requires_task_and_trainset() -> None:
    with pytest.raises(SystemExit):
        main([])
    with pytest.raises(SystemExit):
        main(["--task", "unknown-task", "--trainset", "x.jsonl"])
