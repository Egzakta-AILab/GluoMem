"""Opt-in source-act leaves and codecs; proposals are not entailment authority."""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict

from gluomem.llm.observed_identity import (
    IdentityContext,
    IdentityTarget,
    ObservedSemanticProjection,
    PropertyIdentityProposal,
    observed_claim_type,
)
from gluomem.llm.semantic import (
    ClaimMeaning,
    SemanticClaimContext,
    SemanticClaimSpec,
    SemanticSnapshot,
    SemanticStoredCandidate,
    TargetMeaning,
)

ASSERTION_SCOPE_CONTRACT = "observed-assertion-scope-v4"
ASSERTION_SCOPE_INSTRUCTIONS = (
    "For every claim supply source-only meaning.assertion_act explicitly: direct_assertion "
    "when the source asserts the object-level proposition (including a corrected replacement "
    "or its dependent detail); prior_mistake_report when it asserts that an earlier assertion "
    "was mistaken, not that a value merely mentioned currently holds; unresolved when this "
    "narrow distinction is unknown. Keys/value describe what is actually asserted. Never infer "
    "the act from candidates, chosen targets, actions, rationale or expected answers. Preserve "
    "truthful negative/error reports and their literals. These labels are fallible source "
    "proposals, not entailment certificates. Dates retain their separate world/observation roles."
)


class AssertionScopedMeaning(ClaimMeaning):
    assertion_act: Literal["direct_assertion", "prior_mistake_report", "unresolved"]


class AssertionScopedClaimSpec(SemanticClaimSpec[AssertionScopedMeaning]):
    meaning: AssertionScopedMeaning


class AssertionScopedSnapshot(SemanticSnapshot):
    meaning: AssertionScopedMeaning


class AssertionScopedCandidate(SemanticStoredCandidate[AssertionScopedSnapshot]):
    semantic: AssertionScopedSnapshot | None = None


class AssertionScopedContext(SemanticClaimContext):
    meaning: AssertionScopedMeaning


class AssertionScopedTargetMeaning(TargetMeaning):
    meaning: AssertionScopedMeaning | None
    semantic: AssertionScopedSnapshot | None = None


class AssertionScopedIdentityTarget(IdentityTarget):
    meaning: AssertionScopedMeaning


def compatible_acts(new, old):
    return (
        type(new) is AssertionScopedMeaning
        and type(old) is AssertionScopedMeaning
        and new.assertion_act != "unresolved"
        and new.assertion_act == old.assertion_act
    )


class AssertionScopedIdentityContext(IdentityContext):
    targets: tuple[AssertionScopedIdentityTarget, ...] = ()

    def validate_proposal(self, proposal, context):
        checked = super().validate_proposal(proposal, context)
        if checked.relation not in {"independent", "unresolved"}:
            selected = next(
                t
                for t in self.targets
                if (t.target.frame_id, t.target.claim_index)
                == (checked.target_frame_id, checked.target_claim_index)
            )
            if not compatible_acts(context.meaning, selected.meaning):
                raise ValueError("Linked identity requires known equal source assertion acts")
        return checked


class AssertionScopedProjection(ObservedSemanticProjection[Literal["observed-assertion-scope-v4"]]):
    producer_contract: Literal["observed-assertion-scope-v4"] = ASSERTION_SCOPE_CONTRACT
    meaning: AssertionScopedMeaning
    context: AssertionScopedContext
    target_meanings: tuple[AssertionScopedTargetMeaning, ...]
    identity_context: AssertionScopedIdentityContext
    proposal: PropertyIdentityProposal

    @property
    def untargeted_actions(self) -> tuple[Literal["ADD", "CANDIDATE"], ...]:
        if self.meaning.assertion_act == "unresolved":
            return ("CANDIDATE",)
        return super().untargeted_actions

    def permits_relation(self, action, target, binding):
        return compatible_acts(self.meaning, binding.meaning) and super().permits_relation(
            action, target, binding
        )


class AssertionScopedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    producer_contract: Literal["observed-assertion-scope-v4"]
    meaning: AssertionScopedMeaning


def scoped_claim_type():
    """Required date as well as act at native return/cache/envelope boundaries."""
    return observed_claim_type(AssertionScopedClaimSpec)


def meaning_contract(meaning):
    """No duck typing or subclass-based upgrades of durable semantic authority."""
    if type(meaning) is AssertionScopedMeaning:
        return ASSERTION_SCOPE_CONTRACT
    if type(meaning) is ClaimMeaning:
        return None
    raise ValueError("Unrecognized exact semantic meaning type")


def spec_type(producer_contract):
    return (
        AssertionScopedClaimSpec
        if producer_contract == ASSERTION_SCOPE_CONTRACT
        else SemanticClaimSpec
    )


def snapshot_type(producer_contract):
    return (
        AssertionScopedSnapshot
        if producer_contract == ASSERTION_SCOPE_CONTRACT
        else SemanticSnapshot
    )


def encode_meaning(meaning):
    if meaning_contract(meaning) == ASSERTION_SCOPE_CONTRACT:
        return AssertionScopedPayload(
            producer_contract=ASSERTION_SCOPE_CONTRACT, meaning=meaning
        ).model_dump_json()
    return meaning.model_dump_json()


def decode_binding(raw):
    """Validate an exact known family before treating it as foreign/unknown."""
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Invalid semantic payload")
    if "producer_contract" in payload:
        meaning = AssertionScopedPayload.model_validate_json(raw).meaning
    else:
        meaning = ClaimMeaning.model_validate_json(raw)
    return meaning_contract(meaning), meaning
