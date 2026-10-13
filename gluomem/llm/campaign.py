"""gluomem.llm.campaign — frozen contract for a future prompt-optimization campaign.

The contract (``fixtures/campaign_v1.json``) fixes the model for every role,
the capability weights and family floors used for selection, and a
call/token/dollar reservation for every role in every stage, all inside the
existing $200 experiment ceiling. It names one frozen evaluator by
fingerprint, so the reviewed baseline, MIPROv2 and GEPA candidates are scored
by the same rules.

Nothing here runs an optimizer. Every stage in v1 is ``unrun`` and
``authorized: false``; ``require_authorized`` refuses to start one, and
``BudgetLedger`` refuses spend beyond a reservation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from gluomem.llm.extraction_eval import CAPABILITIES, SEALED_SPLIT, evaluator_fingerprint

CAMPAIGN_PATH = Path(__file__).parent / "fixtures" / "campaign_v1.json"
OPTIMIZERS = ("none", "miprov2", "gepa")
STAGE_STATUSES = ("unrun", "running", "complete")
SELECTION_SPLIT = "dev"
_FLOAT_TOL = 1e-9


class CampaignError(ValueError):
    """The campaign contract is inconsistent or a comparison breaks it."""


class CampaignNotAuthorized(PermissionError):
    """A stage was started without a separate authorization."""


class BudgetExceeded(RuntimeError):
    """A charge would exceed a role reservation."""


@dataclass(frozen=True)
class Reservation:
    calls: int
    tokens: int
    usd: float


@dataclass(frozen=True)
class Stage:
    stage: str
    optimizer: str
    status: str
    authorized: bool
    families: tuple[str, ...]
    reservations: Mapping[str, Reservation]

    @property
    def usd(self) -> float:
        return sum(r.usd for r in self.reservations.values())


@dataclass(frozen=True)
class Family:
    family: str
    kind: str
    weight: float
    floor: float
    max_regression: float


@dataclass(frozen=True)
class Campaign:
    name: str
    version: int
    ceiling_usd: float
    daily_cap_usd: float
    evaluator_fingerprint: str
    model_roles: Mapping[str, Mapping[str, Any]]
    capabilities: Mapping[str, float]
    capability_max_regression: float
    families: Mapping[str, Family]
    stages: Mapping[str, Stage]
    contingency_usd: float

    @property
    def reserved_usd(self) -> float:
        return sum(s.usd for s in self.stages.values())

    def stage(self, name: str) -> Stage:
        if name not in self.stages:
            raise CampaignError(f"unknown stage {name!r}; contract has {sorted(self.stages)}")
        return self.stages[name]

    def plan(self) -> dict[str, Any]:
        """Budget plan for a receipt: per-stage dollars and minimum calendar days."""
        return {
            "campaign": self.name,
            "version": self.version,
            "evaluator": self.evaluator_fingerprint,
            "ceiling_usd": self.ceiling_usd,
            "reserved_usd": self.reserved_usd,
            "contingency_usd": self.contingency_usd,
            "stages": {
                s.stage: {
                    "optimizer": s.optimizer,
                    "status": s.status,
                    "authorized": s.authorized,
                    "usd": s.usd,
                    "min_days_at_daily_cap": math.ceil(s.usd / self.daily_cap_usd),
                }
                for s in self.stages.values()
            },
        }


def _reservation(raw: Mapping[str, Any], where: str) -> Reservation:
    try:
        res = Reservation(int(raw["calls"]), int(raw["tokens"]), float(raw["usd"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise CampaignError(f"{where}: reservation needs calls, tokens and usd") from exc
    if res.calls <= 0 or res.tokens <= 0 or res.usd < 0:
        raise CampaignError(f"{where}: calls and tokens must be positive, usd non-negative")
    return res


def parse_campaign(data: Mapping[str, Any]) -> Campaign:
    roles = data.get("model_roles") or {}
    for role, spec in roles.items():
        if not spec.get("fixed") or not spec.get("models"):
            raise CampaignError(f"model role {role!r} must name fixed models")
        if role != "extraction" and spec.get("optimized") != "never":
            raise CampaignError(f"only the extraction prompt may be optimized, not {role!r}")
    stages: dict[str, Stage] = {}
    for raw in data.get("stages") or []:
        name = str(raw["stage"])
        reservations = {
            role: _reservation(r, f"{name}/{role}")
            for role, r in (raw.get("reservations") or {}).items()
        }
        unknown_roles = set(reservations) - set(roles)
        if unknown_roles:
            raise CampaignError(f"{name}: reserves undeclared roles {sorted(unknown_roles)}")
        if "extraction" not in reservations:
            raise CampaignError(f"{name}: every stage runs extraction and must reserve it")
        stage = Stage(
            stage=name,
            optimizer=str(raw["optimizer"]),
            status=str(raw["status"]),
            authorized=bool(raw["authorized"]),
            families=tuple(raw["families"]),
            reservations=reservations,
        )
        if stage.optimizer not in OPTIMIZERS or stage.status not in STAGE_STATUSES:
            raise CampaignError(f"{name}: bad optimizer {stage.optimizer!r} or status")
        if stage.optimizer != "none" and "proposal" not in reservations:
            raise CampaignError(f"{name}: an optimizer stage must reserve the proposal role")
        stages[name] = stage
    for required in ("reviewed_baseline", "sealed_validation"):
        if required not in stages or stages[required].usd <= 0:
            raise CampaignError(f"contract must reserve dollars for {required!r}")

    families = {
        str(f["family"]): Family(
            family=str(f["family"]),
            kind=str(f["kind"]),
            weight=float(f["weight"]),
            floor=float(f["floor"]),
            max_regression=float(f["max_regression"]),
        )
        for f in data.get("families") or []
    }
    _check_families(families, stages)

    capabilities = {str(k): float(v) for k, v in (data.get("capabilities") or {}).items()}
    expected = {*CAPABILITIES, "abstention"}
    if set(capabilities) != expected:
        raise CampaignError(f"capability weights must cover exactly {sorted(expected)}")
    _check_weights(capabilities, "capability")
    if min(capabilities.values()) < 0.5 * max(capabilities.values()):
        raise CampaignError("capability weights are unbalanced (min < half of max)")

    campaign = Campaign(
        name=str(data["campaign"]),
        version=int(data["version"]),
        ceiling_usd=float(data["ceiling_usd"]),
        daily_cap_usd=float(data["daily_cap_usd"]),
        evaluator_fingerprint=str(data["evaluator"]["fingerprint"]),
        model_roles=roles,
        capabilities=capabilities,
        capability_max_regression=float(data["capability_max_regression"]),
        families=families,
        stages=stages,
        contingency_usd=float(data["contingency_usd"]),
    )
    total = campaign.reserved_usd + campaign.contingency_usd
    if total > campaign.ceiling_usd + _FLOAT_TOL:
        raise CampaignError(
            f"reservations ${campaign.reserved_usd:g} + contingency "
            f"${campaign.contingency_usd:g} exceed the ${campaign.ceiling_usd:g} ceiling"
        )
    return campaign


def _check_weights(weights: Mapping[str, float], what: str) -> None:
    if any(w <= 0 for w in weights.values()) or abs(sum(weights.values()) - 1) > 1e-6:
        raise CampaignError(f"{what} weights must be positive and sum to 1")


def _check_families(families: Mapping[str, Family], stages: Mapping[str, Stage]) -> None:
    from gluomem.bench.registry import REGISTRY

    _check_weights({k: f.weight for k, f in families.items()}, "family")
    for fam in families.values():
        if fam.kind == "registry" and fam.family not in REGISTRY:
            raise CampaignError(f"family {fam.family!r} is not in the benchmark registry")
        if fam.kind not in ("registry", "extraction_fixture"):
            raise CampaignError(f"family {fam.family!r}: unknown kind {fam.kind!r}")
        if not (0 <= fam.floor <= 1 and 0 <= fam.max_regression <= 0.2):
            raise CampaignError(f"family {fam.family!r}: floor or max_regression out of range")
    for stage in stages.values():
        unknown = set(stage.families) - set(families)
        if unknown:
            raise CampaignError(f"{stage.stage}: unknown families {sorted(unknown)}")
        needs_reader = any(families[f].kind == "registry" for f in stage.families)
        if needs_reader and not {"reader", "judge"} <= set(stage.reservations):
            raise CampaignError(
                f"{stage.stage}: benchmark families need reader and judge reservations"
            )


@cache
def load_campaign(path: Path = CAMPAIGN_PATH, *, check_evaluator: bool = True) -> Campaign:
    campaign = parse_campaign(json.loads(Path(path).read_text(encoding="utf-8")))
    if check_evaluator and campaign.evaluator_fingerprint != evaluator_fingerprint():
        raise CampaignError(
            "the campaign pins evaluator "
            f"{campaign.evaluator_fingerprint[:12]} but the current evaluator is "
            f"{evaluator_fingerprint()[:12]}; a changed evaluator needs a new contract"
        )
    return campaign


def require_authorized(campaign: Campaign, stage: str) -> Stage:
    found = campaign.stage(stage)
    if not found.authorized:
        raise CampaignNotAuthorized(
            f"stage {stage!r} of {campaign.name} v{campaign.version} is not authorized; "
            f"running it needs a separate decision and a new contract version"
        )
    return found


@dataclass
class BudgetLedger:
    """Running spend per (stage, role); refuses charges beyond a reservation."""

    campaign: Campaign
    spent: dict[tuple[str, str], Reservation] = field(default_factory=dict)

    def charge(self, stage: str, role: str, *, calls: int, tokens: int, usd: float) -> None:
        cap = self.campaign.stage(stage).reservations.get(role)
        if cap is None:
            raise BudgetExceeded(f"{stage}/{role}: no reservation for this role")
        prev = self.spent.get((stage, role), Reservation(0, 0, 0.0))
        new = Reservation(prev.calls + calls, prev.tokens + tokens, prev.usd + usd)
        over = [
            name
            for name, used, limit in (
                ("calls", new.calls, cap.calls),
                ("tokens", new.tokens, cap.tokens),
                ("usd", new.usd, cap.usd),
            )
            if used > limit + _FLOAT_TOL
        ]
        if over:
            raise BudgetExceeded(f"{stage}/{role}: charge exceeds reserved {', '.join(over)}")
        self.spent[(stage, role)] = new

    def stage_usd(self, stage: str) -> float:
        return sum(r.usd for (s, _), r in self.spent.items() if s == stage)


@dataclass(frozen=True)
class Scorecard:
    """One candidate's development results under the frozen evaluator."""

    candidate: str
    stage: str
    evaluator: str
    split: str
    capabilities: Mapping[str, float | None]
    families: Mapping[str, float | None]
    spend_usd: float = 0.0


@dataclass(frozen=True)
class Decision:
    candidate: str
    eligible: bool
    utility: float | None
    reasons: tuple[str, ...]


def _utility(campaign: Campaign, card: Scorecard) -> float | None:
    values = [card.capabilities.get(c) for c in campaign.capabilities]
    if any(v is None for v in values):
        return None
    return round(
        sum(w * (card.capabilities[c] or 0.0) for c, w in campaign.capabilities.items()), 4
    )


def assess(campaign: Campaign, baseline: Scorecard, card: Scorecard) -> Decision:
    """Eligibility and utility of one candidate against the reviewed baseline."""
    for c in (baseline, card):
        if c.evaluator != campaign.evaluator_fingerprint:
            raise CampaignError(
                f"{c.candidate}: scored by evaluator {c.evaluator[:12]}, not the pinned "
                f"{campaign.evaluator_fingerprint[:12]}; results are not comparable"
            )
        if c.split == SEALED_SPLIT:
            raise CampaignError(f"{c.candidate}: sealed-test results cannot drive selection")
        if c.split != SELECTION_SPLIT:
            raise CampaignError(f"{c.candidate}: selection uses {SELECTION_SPLIT!r} results")
    stage = campaign.stage(card.stage)
    reasons: list[str] = []
    for name in stage.families:
        fam = campaign.families[name]
        got, base = card.families.get(name), baseline.families.get(name)
        if got is None:
            reasons.append(f"family {name} not measured")
            continue
        if got < fam.floor:
            reasons.append(f"family {name} {got:.3f} below floor {fam.floor:.3f}")
        if base is not None and got < base - fam.max_regression:
            reasons.append(f"family {name} regressed {base - got:.3f} vs baseline")
    for cap in campaign.capabilities:
        got, base = card.capabilities.get(cap), baseline.capabilities.get(cap)
        if got is None:
            reasons.append(f"capability {cap} not measured")
        elif base is not None and got < base - campaign.capability_max_regression:
            reasons.append(f"capability {cap} regressed {base - got:.3f} vs baseline")
    if card.spend_usd > stage.usd + _FLOAT_TOL:
        reasons.append(f"spend ${card.spend_usd:g} exceeds ${stage.usd:g} reservation")
    return Decision(card.candidate, not reasons, _utility(campaign, card), tuple(reasons))


def select(
    campaign: Campaign, baseline: Scorecard, candidates: Sequence[Scorecard]
) -> tuple[str, list[Decision]]:
    """Pick the eligible candidate with the highest utility above the baseline.

    Returns ``(winner, decisions)``; the winner is the baseline's candidate id
    when no challenger is eligible and better.
    """
    decisions = [assess(campaign, baseline, c) for c in candidates]
    base_utility = _utility(campaign, baseline) or 0.0
    best = max(
        (d for d in decisions if d.eligible and d.utility is not None and d.utility > base_utility),
        key=lambda d: (d.utility, -decisions.index(d)),
        default=None,
    )
    return (best.candidate if best else baseline.candidate), decisions
