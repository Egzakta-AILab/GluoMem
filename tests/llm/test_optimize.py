"""Optimize seam tests — trainset loading, metric, fake-teleprompter compile.

MIPROv2 is NEVER invoked here; a fake teleprompter proves the offline flow
(load trainset -> compile -> save_artifact with manifest) end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import dspy
import pytest

from gluomem.config.app_config import load_config
from gluomem.llm.optimize import load_trainset, main, metric_for, optimize_task

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


def test_metric_scores_validity_floor_and_gold_agreement(tmp_path: Path) -> None:
    path = _trainset(tmp_path, [_FACT_ROW])
    example = load_trainset(path, "fact_extraction")[0]
    metric = metric_for("fact_extraction")

    from gluomem.llm.outputs import ExtractedFact

    exact = dspy.Prediction(
        facts=[ExtractedFact(text="Alice lives in Boston.", confidence=1.0)]
    )
    assert metric(example, exact) == 1.0
    divergent = dspy.Prediction(facts=[ExtractedFact(text="Wrong fact.")])
    assert metric(example, divergent) == 0.5


def test_optimize_task_with_fake_teleprompter(tmp_path: Path) -> None:
    cfg = load_config().with_overrides(
        {"dspy.artifacts.dir": str(tmp_path / "artifacts")}
    )
    trainset_path = _trainset(tmp_path, [_FACT_ROW])
    seen: dict = {}

    class FakeTeleprompter:
        def __init__(self, metric) -> None:
            self.metric = metric

        def compile(self, student, *, trainset):
            seen["student"] = student
            seen["trainset"] = trainset
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
    assert len(seen["trainset"]) == 1
    assert isinstance(seen["student"], dspy.Predict)


def test_missing_key_fails_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Point load_config at a nonexistent .env so a developer's real repo .env
    # can't repopulate the key this test removes.
    cfg = load_config(env_file=tmp_path / "absent.env")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        optimize_task("fact_extraction", cfg, _trainset(tmp_path, [_FACT_ROW]))


def test_main_requires_task_and_trainset() -> None:
    with pytest.raises(SystemExit):
        main([])
    with pytest.raises(SystemExit):
        main(["--task", "unknown-task", "--trainset", "x.jsonl"])
