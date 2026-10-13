"""Future optimization campaign contract: roles, balance, floors, budget, selection.

Nothing here runs an optimizer or calls a model. The tests prove the
contract is internally consistent, fits the $200 ceiling, stays unrun, and
that its selector refuses results it cannot compare fairly.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from gluomem.config.app_config import load_config
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
from gluomem.llm.extraction_eval import evaluator_fingerprint


def _doc() -> dict[str, Any]:
    return json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))


def test_contract_fits_the_ceiling_and_reserves_final_validation() -> None:
    campaign = load_campaign()
    assert campaign.ceiling_usd == 200
    assert campaign.daily_cap_usd == 10
    assert campaign.reserved_usd + campaign.contingency_usd <= 200
    assert campaign.stage("sealed_validation").usd > 0
    plan = campaign.plan()
    assert plan["stages"]["gepa_joint"]["min_days_at_daily_cap"] == 7


def test_every_stage_is_unrun_and_unauthorized() -> None:
    campaign = load_campaign()
    assert {"reviewed_baseline", "miprov2", "gepa_focused", "gepa_joint"} <= set(
        campaign.stages
    )
    for name, stage in campaign.stages.items():
        assert stage.status == "unrun", name
        assert stage.authorized is False, name
        with pytest.raises(CampaignNotAuthorized):
            require_authorized(campaign, name)


def test_model_roles_are_fixed_and_configured() -> None:
    campaign = load_campaign()
    models = set(load_config().models)
    assert set(campaign.model_roles) == {"extraction", "proposal", "reader", "judge", "embedding"}
    for role, spec in campaign.model_roles.items():
        assert spec["fixed"] is True
        assert set(spec["models"]) <= models, role
    assert campaign.model_roles["reader"]["optimized"] == "never"
    assert campaign.model_roles["judge"]["optimized"] == "never"


def test_contract_pins_the_current_evaluator() -> None:
    assert load_campaign().evaluator_fingerprint == evaluator_fingerprint()


def test_capabilities_are_balanced_and_families_have_floors() -> None:
    campaign = load_campaign()
    weights = campaign.capabilities.values()
    assert abs(sum(weights) - 1) < 1e-9
    assert min(weights) >= 0.5 * max(weights)
    for fam in campaign.families.values():
        assert 0 <= fam.floor <= 1
        assert fam.max_regression <= 0.2


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(contingency_usd=100), "exceed the \\$200 ceiling"),
        (lambda d: d["stages"][1]["reservations"].pop("proposal"), "reserve the proposal"),
        (lambda d: d["stages"][3]["reservations"].pop("judge"), "reader and judge"),
        (lambda d: d["stages"][0]["reservations"].pop("extraction"), "must reserve it"),
        (lambda d: d["model_roles"]["judge"].update(optimized="prompt"), "only the extraction"),
        (lambda d: d["model_roles"]["reader"].update(fixed=False), "fixed models"),
        (lambda d: d["capabilities"].update(support=0.5, subject=0.01), "sum to 1|unbalanced"),
        (lambda d: d["capabilities"].pop("abstention"), "cover exactly"),
        (lambda d: d["families"][1].update(family="nonexistent"), "unknown families|registry"),
        (lambda d: d["stages"][4]["reservations"].clear(), "must reserve"),
    ],
)
def test_inconsistent_contracts_are_rejected(mutate: Any, message: str) -> None:
    doc = copy.deepcopy(_doc())
    mutate(doc)
    with pytest.raises(CampaignError, match=message):
        parse_campaign(doc)


def test_ledger_counts_every_role_against_its_reservation() -> None:
    ledger = BudgetLedger(load_campaign())
    ledger.charge("gepa_focused", "proposal", calls=10, tokens=1000, usd=14.0)
    with pytest.raises(BudgetExceeded, match="usd"):
        ledger.charge("gepa_focused", "proposal", calls=1, tokens=10, usd=1.5)
    with pytest.raises(BudgetExceeded, match="no reservation"):
        ledger.charge("gepa_focused", "judge", calls=1, tokens=1, usd=0.0)
    with pytest.raises(BudgetExceeded, match="calls"):
        ledger.charge("miprov2", "proposal", calls=301, tokens=1, usd=0.0)
    assert ledger.stage_usd("gepa_focused") == 14.0


def _card(candidate: str, stage: str, value: float, **over: Any) -> Scorecard:
    campaign = load_campaign()
    fields: dict[str, Any] = {
        "candidate": candidate,
        "stage": stage,
        "evaluator": campaign.evaluator_fingerprint,
        "split": "dev",
        "capabilities": dict.fromkeys(campaign.capabilities, value),
        "families": {"extraction_v1": value},
        "spend_usd": 1.0,
    }
    fields.update(over)
    return Scorecard(**fields)


def test_same_evaluator_compares_baseline_miprov2_and_gepa() -> None:
    campaign = load_campaign()
    baseline = _card("baseline", "reviewed_baseline", 0.70,
                     families=dict.fromkeys(campaign.families, 0.7))
    mipro = _card("mipro", "miprov2", 0.74)
    gepa = _card("gepa", "gepa_focused", 0.78)
    winner, decisions = select(campaign, baseline, [mipro, gepa])
    assert winner == "gepa"
    assert all(d.eligible for d in decisions)


def test_selector_enforces_floors_regression_and_spend() -> None:
    campaign = load_campaign()
    baseline = _card("baseline", "reviewed_baseline", 0.70,
                     families=dict.fromkeys(campaign.families, 0.7))
    below_floor = _card("floor", "gepa_focused", 0.9, families={"extraction_v1": 0.5})
    caps = dict.fromkeys(campaign.capabilities, 0.95)
    caps["negation"] = 0.60
    regressed = _card("negation-regressed", "gepa_focused", 0.9, capabilities=caps)
    overspent = _card("overspent", "gepa_focused", 0.9, spend_usd=31.0)
    winner, decisions = select(campaign, baseline, [below_floor, regressed, overspent])
    assert winner == "baseline"
    reasons = {d.candidate: " ".join(d.reasons) for d in decisions}
    assert "below floor" in reasons["floor"]
    assert "capability negation regressed" in reasons["negation-regressed"]
    assert "exceeds" in reasons["overspent"]


def test_selector_refuses_foreign_evaluator_and_sealed_results() -> None:
    campaign = load_campaign()
    baseline = _card("baseline", "reviewed_baseline", 0.7)
    with pytest.raises(CampaignError, match="not comparable"):
        assess(campaign, baseline, _card("x", "gepa_focused", 0.9, evaluator="0" * 64))
    with pytest.raises(CampaignError, match="sealed-test"):
        assess(campaign, baseline, _card("x", "gepa_focused", 0.9, split="sealed_test"))


def test_changed_evaluator_invalidates_the_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gluomem.llm.campaign as campaign_mod

    load_campaign.cache_clear()
    monkeypatch.setattr(campaign_mod, "evaluator_fingerprint", lambda: "f" * 64)
    try:
        with pytest.raises(CampaignError, match="needs a new contract"):
            campaign_mod.load_campaign()
    finally:
        load_campaign.cache_clear()
