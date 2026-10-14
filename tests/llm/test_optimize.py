"""DSPy artifact bootstrap, prompt optimizer seam, and campaign contract tests."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from gluomem.config.app_config import load_config
from gluomem.llm.bootstrap import (
    ArtifactError,
    artifact_dir,
    load_artifact,
    load_or_build,
    save_artifact,
)
from gluomem.llm.campaign import (
    CAMPAIGN_PATH,
    BudgetExceeded,
    BudgetLedger,
    CampaignError,
    CampaignNotAuthorized,
    Scorecard,
    assess,
    load_campaign,
    parse_campaign,
    require_authorized,
    select,
)
from gluomem.llm.extraction_eval import SealedSplitError, evaluator_fingerprint
from gluomem.llm.modules import build_module
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


def test_artifact_save_load_tamper_pin_and_require(tmp_path: Path) -> None:
    assert artifact_dir("artifacts/dspy", "cognify") == Path("artifacts/dspy/cognify")
    cfg = load_config().with_overrides({"dspy.artifacts.dir": str(tmp_path / "artifacts")})
    module = build_module("fact_extraction", cfg)
    target = save_artifact(
        module, cfg.dspy.artifacts.dir, "fact_extraction", model_ref="deepseek-flash"
    )
    assert (target / "program.pkl").exists()
    manifest = json.loads((target / "manifest.gluomem.json").read_text())
    assert manifest["task"] == "fact_extraction" and len(manifest["sha256"]) == 64

    loaded = load_artifact(target.parent, "fact_extraction", pins={})
    lm = DummyLM([{"facts": [{"text": "Alice lives in Boston.", "confidence": 1.0}]}])
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        assert loaded(text="Alice lives in Boston.").facts[0].text == "Alice lives in Boston."

    with pytest.raises(ArtifactError, match="pin mismatch"):
        load_artifact(target.parent, "fact_extraction", pins={"fact_extraction": "0" * 64})

    assert isinstance(load_or_build("fact_extraction", cfg), dspy.Predict)
    empty_cfg = load_config().with_overrides({"dspy.artifacts.dir": str(tmp_path / "empty")})
    assert isinstance(load_or_build("cognify", empty_cfg), dspy.Predict)

    req_cfg = empty_cfg.with_overrides({"dspy.artifacts.require": True})
    with pytest.raises(ArtifactError, match="optimize-prompts"):
        load_or_build("fact_extraction", req_cfg)

    program = target / "program.pkl"
    program.write_bytes(program.read_bytes() + b"tamper")
    with pytest.raises(ArtifactError, match="hash mismatch"):
        load_artifact(target.parent, "fact_extraction", pins={})


def test_trainset_loading_and_metrics(tmp_path: Path) -> None:
    examples = load_trainset(_trainset(tmp_path, [_FACT_ROW]), "fact_extraction")
    assert len(examples) == 1 and set(examples[0].inputs().keys()) == {"text"}

    with pytest.raises(ValueError, match="missing input fields"):
        load_trainset(_trainset(tmp_path, [{"facts": []}], "bad1.jsonl"), "fact_extraction")
    with pytest.raises(ValueError, match="empty trainset"):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        load_trainset(empty, "fact_extraction")

    ana_ex = next(
        ex
        for ex in fixture_examples("fact_extraction", splits=("dev",))
        if ex.case_id == "en-ana-move"
    )
    f_metric = metric_for("fact_extraction")
    wrong_pred = dspy.Prediction(facts=[ExtractedFact(text="Ana lives in Paris.")])
    assert f_metric(ana_ex, wrong_pred) == 0.0

    gepa_out = f_metric(
        ana_ex,
        dspy.Prediction(facts=[ExtractedFact(text="Ana bought a bicycle.")]),
        None,
        "extract",
        None,
    )
    assert isinstance(gepa_out, dspy.Prediction)
    assert "plan or possibility as done" in gepa_out.feedback

    e_metric = metric_for("event_extraction")
    ev_ex = fixture_examples("event_extraction", splits=("dev",))[0]
    good_ev = dspy.Prediction(
        events=[
            ExtractedEvent(text="Project kickoff", event_date=datetime(2026, 2, 3)),
            ExtractedEvent(text="Client demo", event_date=datetime(2026, 3, 24)),
        ]
    )
    assert e_metric(ev_ex, good_ev) == 1.0

    c_row = {
        "new_statement": "a",
        "stored_statement": "b",
        "verdict": {"is_contradiction": False, "description": "", "confidence": 1.0},
    }
    c_ex = load_trainset(_trainset(tmp_path, [c_row], "c.jsonl"), "contradiction")[0]
    c_metric = metric_for("contradiction")
    c_false = dspy.Prediction(verdict=ContradictionVerdict(is_contradiction=False))
    c_true = dspy.Prediction(verdict=ContradictionVerdict(is_contradiction=True))
    assert c_metric(c_ex, c_false) == 1.0
    assert c_metric(c_ex, c_true) == 0.5


def test_sealed_split_and_optimize_task_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SealedSplitError):
        fixture_examples("fact_extraction", splits=("train", "sealed_test"))
    with pytest.raises(ValueError, match="annotated extraction fixture"):
        raw_ex = load_trainset(_trainset(tmp_path, [_FACT_ROW]), "fact_extraction")
        check_trainset("fact_extraction", raw_ex)

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-placeholder")
    cfg = load_config(env_file=tmp_path / "absent.env").with_overrides(
        {"dspy.artifacts.dir": str(tmp_path / "artifacts")}
    )

    class FakeTeleprompter:
        def __init__(self, metric) -> None:
            self.metric = metric

        def compile(self, student, *, trainset):
            assert self.metric(trainset[0], dspy.Prediction(facts=[])) == 0.0
            return student

    target = optimize_task(
        "fact_extraction",
        cfg,
        fixture_examples("fact_extraction", splits=("train",)),
        teleprompter_factory=FakeTeleprompter,
    )
    assert (target / "program.pkl").exists()

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        optimize_task("fact_extraction", cfg, fixture_examples("fact_extraction"))

    with pytest.raises(SystemExit):
        main([])


def test_campaign_contract_ledger_and_selector() -> None:
    campaign = load_campaign()
    assert campaign.ceiling_usd == 200 and campaign.reserved_usd + campaign.contingency_usd <= 200
    assert campaign.evaluator_fingerprint == evaluator_fingerprint()
    for name, stage in campaign.stages.items():
        assert stage.status == "unrun" and not stage.authorized
        with pytest.raises(CampaignNotAuthorized):
            require_authorized(campaign, name)

    doc = copy.deepcopy(json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8")))
    doc["contingency_usd"] = 100
    with pytest.raises(CampaignError, match="exceed the \\$200 ceiling"):
        parse_campaign(doc)

    ledger = BudgetLedger(campaign)
    ledger.charge("gepa_focused", "proposal", calls=10, tokens=1000, usd=14.0)
    with pytest.raises(BudgetExceeded, match="usd"):
        ledger.charge("gepa_focused", "proposal", calls=1, tokens=10, usd=1.5)

    def _card(cand: str, stage: str, val: float, **over: Any) -> Scorecard:
        fields: dict[str, Any] = {
            "candidate": cand,
            "stage": stage,
            "evaluator": campaign.evaluator_fingerprint,
            "split": "dev",
            "capabilities": dict.fromkeys(campaign.capabilities, val),
            "families": dict.fromkeys(campaign.families, val),
            "spend_usd": 1.0,
        }
        fields.update(over)
        return Scorecard(**fields)

    base = _card("baseline", "reviewed_baseline", 0.70)
    mipro = _card("mipro", "miprov2", 0.74)
    gepa = _card("gepa", "gepa_focused", 0.78)
    winner, decisions = select(campaign, base, [mipro, gepa])
    assert winner == "gepa" and all(d.eligible for d in decisions)

    with pytest.raises(CampaignError, match="not comparable"):
        assess(campaign, base, _card("x", "gepa_focused", 0.9, evaluator="0" * 64))
    with pytest.raises(CampaignError, match="sealed-test"):
        assess(campaign, base, _card("x", "gepa_focused", 0.9, split="sealed_test"))
