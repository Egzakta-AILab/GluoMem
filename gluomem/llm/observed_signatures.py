"""Invocation-local identity schemas finalized after artifact loading."""

from copy import deepcopy

import dspy

from gluomem.llm.observed_identity import (
    PRODUCER_CONTRACT,
    PROPERTY_IDENTITY_CONTRACT,
    IdentityContext,
    IdentityProposal,
    PropertyIdentityProposal,
    identity_proposal_model,
)
from gluomem.llm.property_identity import PROPERTY_IDENTITY_INSTRUCTIONS, PropertySlotContext
from gluomem.llm.semantic import SemanticClaimContext
from gluomem.llm.signatures import _require_signature, _signature_fields

IDENTITY_INSTRUCTIONS = (
    "Propose an explicit relation to exactly one observed assertion, or independent/unresolved "
    "with no reference. same_assertion means the same asserted property and perspective, not "
    "merely a shared subject/topic or contradictory words. Distinguish negated error reports "
    "from the value assertion they discuss. refines_assertion requires genuinely dependent "
    "detail and its observed independent anchor. Never copy candidate keys into the source-only "
    "new meaning. Candidate receipts are revocable observations, not semantic entailment. "
    "Use only exact supplied stored/prior-publication references; no guesses or forward links. "
    "World-effective and source-observation dates are distinct. Rationale grants no authority."
)


class IdentityAlignment(dspy.Signature):
    new_claim: str = dspy.InputField()
    new_claim_context: SemanticClaimContext = dspy.InputField()
    identity_context: IdentityContext = dspy.InputField()
    proposal: IdentityProposal = dspy.OutputField()


class PropertyIdentityAlignment(dspy.Signature):
    # A distinct output contract is not a mutable-field subtype. Retain the
    # predecessor's inherited instructions and field order exactly.
    __doc__ = IdentityAlignment.__doc__
    new_claim: str = dspy.InputField()
    new_claim_context: SemanticClaimContext = dspy.InputField()
    identity_context: IdentityContext = dspy.InputField()
    proposal: PropertyIdentityProposal = dspy.OutputField()
    property_slot_context: PropertySlotContext = dspy.InputField()


def assertion_scope_identity_signature():
    from gluomem.llm.assertion_scope import AssertionScopedContext, AssertionScopedIdentityContext

    fields = _signature_fields(PropertyIdentityAlignment)
    fields["new_claim_context"] = (AssertionScopedContext, deepcopy(fields["new_claim_context"][1]))
    fields["identity_context"] = (
        AssertionScopedIdentityContext,
        deepcopy(fields["identity_context"][1]),
    )
    return dspy.make_signature(
        fields,
        PropertyIdentityAlignment.instructions,
        signature_name="AssertionScopeIdentityAlignment",
    )


def identity_alignment_module(base, projection, context, *, producer_contract=PRODUCER_CONTRACT):
    module = base.deepcopy()
    scoped = producer_contract == "observed-assertion-scope-v4"
    property_identity = producer_contract == PROPERTY_IDENTITY_CONTRACT or scoped
    scope_instructions = ""
    for predictor in module.predictors():
        signature = _require_signature(predictor)
        if not IdentityAlignment.fields.keys() <= signature.fields.keys():
            raise ValueError("Identity artifact lacks required native fields")
        fields = _signature_fields(signature)
        proposal_type = signature.fields["proposal"].annotation
        if proposal_type is None:
            raise ValueError("Identity artifact lacks native proposal type")
        rationale = proposal_type.model_fields["rationale"]
        limit = min([512, *[m.max_length for m in rationale.metadata if hasattr(m, "max_length")]])
        fields["proposal"] = (
            identity_proposal_model(
                projection, context, max_rationale_length=limit, producer_contract=producer_contract
            ),
            deepcopy(signature.fields["proposal"]),
        )
        for n in IdentityAlignment.input_fields:
            fields[n] = (
                IdentityAlignment.fields[n].annotation or str,
                deepcopy(signature.fields[n]),
            )
        if property_identity:
            fields["property_slot_context"] = (
                PropertySlotContext,
                deepcopy(PropertyIdentityAlignment.fields["property_slot_context"]),
            )
        if scoped:
            from gluomem.llm.assertion_scope import (
                ASSERTION_SCOPE_INSTRUCTIONS,
                AssertionScopedContext,
                AssertionScopedIdentityContext,
            )

            scope_instructions = "\n\n" + ASSERTION_SCOPE_INSTRUCTIONS
            fields["new_claim_context"] = (
                AssertionScopedContext,
                deepcopy(signature.fields["new_claim_context"]),
            )
            fields["identity_context"] = (
                AssertionScopedIdentityContext,
                deepcopy(signature.fields["identity_context"]),
            )
        predictor.signature = dspy.make_signature(
            fields,
            signature.instructions
            + "\n\n"
            + (PROPERTY_IDENTITY_INSTRUCTIONS if property_identity else IDENTITY_INSTRUCTIONS)
            + scope_instructions,
            signature_name=signature.__name__,
        )
    return module
