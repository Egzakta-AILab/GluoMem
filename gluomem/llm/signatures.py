"""gluomem.llm.signatures — DSPy 3.x signatures for the four extraction tasks.

Typed outputs go DIRECTLY on ``dspy.OutputField`` (DSPy 3.x removed
``TypedPredictor``/``Assert``/``Suggest``; output-validity retries are
``dspy.Refine``'s job, wired in ``modules.py``). The Pydantic output models
live in the dspy-free seam (``gluomem/llm/outputs.py``) so the write path can
consume them without importing dspy.

Signature class docstrings and field ``desc`` strings are LIVE PROMPT text
(hashed into run manifests) — any edit changes model behaviour.
"""

from __future__ import annotations

import re
from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Literal

import dspy
from pydantic import Field, create_model
from pydantic_core import PydanticUndefined

from gluomem.llm.outputs import (
    ACTION_TARGET_WRITER_CONTRACT,
    SUPPORTED_WRITER_CONTRACTS,
    ClaimConsolidationContext,
    ConsolidationDecision,
    ConsolidationProjection,
    ContradictionVerdict,
    ExtractedClaimSpec,
    ExtractedEvent,
    ExtractedFact,
    KGEntity,
    KGRelation,
    StoredCandidate,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _require_signature(predictor: dspy.Predict) -> type[dspy.Signature]:
    signature = predictor.signature
    if signature is None:
        raise ValueError("native typed predictor requires a signature")
    return signature


def _signature_fields(signature: type[dspy.Signature]):
    # DSPy make_signature uses str for a missing annotation. Resolve that same
    # fallback here so the copied field map contains only type forms.
    return {name: (f.annotation or str, deepcopy(f)) for name, f in signature.fields.items()}


class FactExtraction(dspy.Signature):
    """Extract atomic, self-contained facts from a conversation turn; each
    fact must stand alone without pronoun references."""

    text: str = dspy.InputField()
    facts: list[ExtractedFact] = dspy.OutputField()


class EventExtraction(dspy.Signature):
    """Extract episodic events; resolve relative time cues to ISO dates."""

    source_text: str = dspy.InputField()
    reference_date: datetime = dspy.InputField()
    events: list[ExtractedEvent] = dspy.OutputField()


class CognifyExtraction(dspy.Signature):
    """Extract typed entities + relations from a frame."""

    text: str = dspy.InputField()
    entities: list[KGEntity] = dspy.OutputField()
    relations: list[KGRelation] = dspy.OutputField()


class ContradictionCheck(dspy.Signature):
    """Judge whether a new statement contradicts a stored statement;
    describe the contradiction if so."""

    new_statement: str = dspy.InputField()
    stored_statement: str = dspy.InputField()
    verdict: ContradictionVerdict = dspy.OutputField()


class EventClaimExtraction(dspy.Signature):
    """Extract atomic, self-contained memory claims from a source event; classify
    each claim's kind, resolve relative dates against reference_date, attach the
    supporting verbatim quote, and extract typed entities and relations."""

    source_text: str = dspy.InputField()
    reference_date: str = dspy.InputField()
    claims: list[ExtractedClaimSpec] = dspy.OutputField()


SOURCE_ALIGNED_QUOTE_INSTRUCTIONS = (
    "For every claim capture sufficient exact source wording to support the whole claim, "
    "including who/what and any asserted value or world-effective date. Copy a contiguous "
    "verbatim_span from the actual source message, with its supplied message_id. Do not use "
    "interior characters, partial words or partial codes as evidence. Short complete tokens "
    "or codes are permitted when they are sufficient evidence. Keep literal whitespace; "
    "never fabricate, expand or replace a quote. Prefer the narrow supporting wording, not "
    "an entire message containing unrelated or superseded/erased facts. Omit a claim if no "
    "sufficient exact quote is available. Lexical boundary validation is not semantic entailment."
)


WRITER_RATIONALE_INSTRUCTIONS = (
    "The rationale is diagnostic prose only: it may be omitted and defaults to the empty "
    "string. If supplied, it must be a string within its declared character bound. Always "
    "supply the structured action and both explicit nullable target fields; prose never "
    "selects, repairs or substitutes an action or target."
)


SEMANTIC_EXTRACTION_INSTRUCTIONS = (
    "Semantic-state-writer-v4: propose a meaning for EVERY source-bound claim. These labels "
    "are untrusted source-derived proposals, not deterministic semantic entailment. Use the "
    "supplied source_context scope and message roles; key.perspective is the exact bound role. "
    "key.subject identifies the asserted subject, key.predicate the independently changeable "
    "property, and value_key its normalized assertion value. Preserve distinct predicates and "
    "perspectives, including reports about errors versus the corrected value assertion. Every "
    "independently versioned state has independently_versioned=true and refines=null, even "
    "when it shares a person, topic, kind or date with another claim. Only a genuinely dependent "
    "detail has independently_versioned=false and an explicit assertion-key refines anchor. "
    "Derive identity and anchor from the source, never from a chosen action, target or codec. "
    "Do not omit claims, expand quotes, or use kind/date/GOP similarity as semantic identity."
)


def source_quote_signature(
    signature: type[dspy.Signature],
    capture_policy: str | None,
    *,
    writer_contract: str | None = None,
):
    """Require a nonblank literal quote only for explicit source-bound extraction.

    Preserve loaded instructions, demos (on the predictor), other output fields
    and constraints. This task-local type never changes the permissive public
    ExtractedClaimSpec. Literal membership remains the writer's responsibility.
    """
    if writer_contract is not None and writer_contract not in SUPPORTED_WRITER_CONTRACTS:
        raise ValueError("unknown writer contract")
    if writer_contract and capture_policy is None:
        raise ValueError("writer contract requires an explicit source capture policy")
    if capture_policy is None:
        return signature
    if capture_policy not in {"trusted-only", "attributed-interaction"}:
        raise ValueError("unknown source capture policy")
    from typing import get_args, get_origin

    if not EventClaimExtraction.fields.keys() <= signature.fields.keys():
        raise ValueError("source-bound extraction lacks required task fields")
    annotation = signature.fields["claims"].annotation
    args = get_args(annotation)
    if (
        get_origin(annotation) is not list
        or len(args) != 1
        or not isinstance(args[0], type)
        or not issubclass(args[0], ExtractedClaimSpec)
    ):
        raise ValueError("source-bound extraction requires native typed claims")
    claim_type = args[0]
    quote = deepcopy(claim_type.model_fields["verbatim_span"])
    quote.default = PydanticUndefined
    quote.default_factory = None
    # Compiled Python regex keeps parser whitespace semantics aligned with
    # JSON-schema validation and the writer's str.strip check, without trimming.
    pattern = r"^[\s\S]*\S[\s\S]*$" if writer_contract else r"\S"
    quote.metadata.extend(Field(min_length=1, pattern=re.compile(pattern)).metadata)
    strict_claim = create_model(
        claim_type.__name__,
        __base__=claim_type,
        __doc__=claim_type.__doc__,
        verbatim_span=(str, quote),
    )
    if writer_contract == "semantic-state-writer-v4":
        from gluomem.llm.semantic import ClaimMeaning, SemanticClaimSpec

        strict_claim = create_model(
            "SemanticClaimSpec",
            __base__=strict_claim,
            __config__={"extra": "forbid"},
            __doc__=SemanticClaimSpec.__doc__,
            meaning=(ClaimMeaning, Field()),
        )
    fields = _signature_fields(signature)
    fields["claims"] = (list[strict_claim], deepcopy(signature.fields["claims"]))
    instructions = signature.instructions
    if writer_contract and not instructions.endswith(SOURCE_ALIGNED_QUOTE_INSTRUCTIONS):
        instructions += "\n\n" + SOURCE_ALIGNED_QUOTE_INSTRUCTIONS
    if writer_contract == "semantic-state-writer-v4":
        fields["source_context"] = (dict, dspy.InputField())
        instructions += "\n\n" + SEMANTIC_EXTRACTION_INSTRUCTIONS
    return dspy.make_signature(fields, instructions, signature_name=signature.__name__)


DATE_PRESENCE_INSTRUCTIONS = (
    "Supply event_date explicitly for EVERY claim, including facts, as an explicit "
    "world-effective date or null. Preserve absolute on/since dates and supported relative "
    "dates against reference_date. Source-observation timestamps do not establish world "
    "validity. Unknown dates remain null; do not infer dates from storage or receipt time. "
    "Use value-neutral stable assertion keys derived only from this source."
)


def enforce_observed_date_contract(module):
    """Task-local post-artifact overlay; retain demos, CoT and intrinsic metadata."""
    for predictor in module.predictors():
        signature = _require_signature(predictor)
        annotation = signature.fields["claims"].annotation
        if annotation is None:
            raise ValueError("observed extraction requires native typed claims")
        claim_type = annotation.__args__[0]
        from gluomem.llm.observed_identity import observed_claim_type

        dated = observed_claim_type(claim_type)
        fields = _signature_fields(signature)
        fields["claims"] = (list[dated], deepcopy(signature.fields["claims"]))
        predictor.signature = dspy.make_signature(
            fields,
            signature.instructions + "\n\n" + DATE_PRESENCE_INSTRUCTIONS,
            signature_name=signature.__name__,
        )
    return module


def enforce_assertion_scope_contract(module):
    """Source-only act overlay after quote/date restoration, before limits."""
    from gluomem.llm.assertion_scope import ASSERTION_SCOPE_INSTRUCTIONS, AssertionScopedMeaning

    for predictor in module.predictors():
        signature = _require_signature(predictor)
        annotation = signature.fields["claims"].annotation
        if annotation is None:
            raise ValueError("assertion scope requires native typed claims")
        claim_type = annotation.__args__[0]
        scoped = create_model(
            "AssertionScopedClaimSpec",
            __base__=claim_type,
            meaning=(AssertionScopedMeaning, deepcopy(claim_type.model_fields["meaning"])),
        )
        fields = _signature_fields(signature)
        fields["claims"] = (list[scoped], deepcopy(signature.fields["claims"]))
        predictor.signature = dspy.make_signature(
            fields,
            signature.instructions + "\n\n" + ASSERTION_SCOPE_INSTRUCTIONS,
            signature_name=signature.__name__,
        )
    return module


def enforce_source_quote_contract(
    module: dspy.Module,
    capture_policy: str | None,
    *,
    writer_contract: str | None = None,
):
    """Finalize source-mode types AFTER loading compiled state, without output repair."""
    if capture_policy is not None:
        predictors = module.predictors()
        if not predictors:
            raise ValueError("source-bound extraction requires native typed predictors")
        for predictor in predictors:
            predictor.signature = source_quote_signature(
                _require_signature(predictor), capture_policy, writer_contract=writer_contract
            )
    return module


class ClaimConsolidation(dspy.Signature):
    """Decide whether a new claim is independent (ADD), adds compatible detail to an
    observed I-frame (REFINE), is already stored (NOOP), records an ordinary dated
    state change preserving the prior I (TRANSITION), corrects/replaces an existing
    stored candidate (SUPERSEDE), or requires review (CANDIDATE)."""

    new_claim: str = dspy.InputField()
    candidates: list[StoredCandidate] = dspy.InputField()
    decision: ConsolidationDecision = dspy.OutputField()


CONSOLIDATION_POLICY_INSTRUCTIONS = (
    "Mandatory action/target policy: return an explicit action. NOOP and SUPERSEDE require "
    "a positive integer target_frame_id selected from the supplied observed candidates. "
    "REFINE requires a positive integer target_frame_id selecting an eligible observed I-frame "
    "by its frame_type; it adds compatible detail as a P-frame in that I-frame's GOP. "
    "Never REFINE a P-frame or B-frame, or silently rebase onto another I-frame. "
    "TRANSITION requires an eligible observed I-frame with temporal_operation and observed_at "
    "metadata, plus an explicit world-effective date in the new claim and trusted source "
    "observation time. It creates a new independent I-frame without declaring the prior state "
    "wrong or destroying its authorized history. Use effective_at/observed_at and explicit "
    "previous_frame_id, never frame-ID ordering, to distinguish ordinary changes from corrections. "
    "ADD and an untargeted CANDIDATE review proposal may use null. Never infer a target "
    "from rationale, invent an ID, silently retarget, or substitute ADD for an invalid "
    "correction, refinement or transition."
)


def enforce_consolidation_contract(module: dspy.Module) -> dspy.Module:
    """Restore the task's typed safety boundary after loading compiled program state.

    Retain compiled instructions as a legacy prefix, demos and extra CoT fields;
    they cannot replace the mandatory policy or the native decision parser.
    """
    predictors = module.predictors()
    if not predictors:
        raise ValueError("consolidation requires native typed predictors")
    for predictor in predictors:
        signature = predictor.signature
        if signature is None or not ClaimConsolidation.fields.keys() <= signature.fields.keys():
            raise ValueError("compiled consolidation lacks required task fields")
        fields = _signature_fields(signature)
        for name, canonical in ClaimConsolidation.fields.items():
            fields[name] = (canonical.annotation or str, deepcopy(canonical))
        instructions = signature.instructions
        if not instructions.endswith(CONSOLIDATION_POLICY_INSTRUCTIONS):
            instructions += "\n\n" + CONSOLIDATION_POLICY_INSTRUCTIONS
        predictor.signature = dspy.make_signature(
            fields, instructions, signature_name=signature.__name__
        )
    return module


CONSOLIDATION_CONTEXT_INSTRUCTIONS = (
    "Authoritative new-claim context: use new_claim_context.kind, source_role, observed_at "
    "and effective_at exactly as supplied. effective_at is world-validity time; observed_at "
    "is trusted source-observation time, not a replacement for world time. A null effective_at "
    "means unknown world time and does not permit TRANSITION, even with a nonnull observed_at "
    "or a source timestamp. TRANSITION also requires nonnull observed_at. An ordinary "
    "historical reaffirmation is not a new state change or an error correction, even if its "
    "extracted kind is correction: use NOOP with the explicit observed prior target when "
    "already captured. Do not SUPERSEDE true prior history merely because a later state is "
    "true. Preserve source_role attribution; an assistant statement is not a user statement."
)


def contextual_consolidation_module(module: dspy.Module) -> dspy.Module:
    """Add one typed context input to an invocation-owned compiled-program copy.

    Do not rebind output fields here: the loaded native parser, bounds, CoT
    outputs, compiled instructions and demos have already been finalized.
    """
    contextual = module.deepcopy()
    for predictor in contextual.predictors():
        signature = _require_signature(predictor)
        fields = _signature_fields(signature)
        fields["new_claim_context"] = (
            ClaimConsolidationContext,
            dspy.InputField(
                desc=(
                    "Authoritative source-bound metadata for new_claim; null values remain unknown."
                )
            ),
        )
        predictor.signature = dspy.make_signature(
            fields,
            signature.instructions + "\n\n" + CONSOLIDATION_CONTEXT_INSTRUCTIONS,
            signature_name=signature.__name__,
        )
    return contextual


CONSOLIDATION_PROJECTION_INSTRUCTIONS = (
    "Evolving per-claim projection: projection.targets is the legal target/action menu for this "
    "claim. A conditional target names an earlier proposed new independent I publication via "
    "target_claim_index (zero-based), NOT stored truth or a guessed frame ID. Its dependency "
    "must actually publish an eligible new I or the event will refuse atomically. For each "
    "targeted action select exactly one target_frame_id or target_claim_index explicitly from "
    "that menu; set the other to null. ADD/CANDIDATE use both null. Prior proposed effects "
    "consume destructive/branching actions but do not establish world chronology: use explicit "
    "effective_at, observed_at and previous links, never extraction order. A historical prior "
    "state can remain a legal NOOP/REFINE target after TRANSITION, not another branching "
    "TRANSITION. Never reinterpret an invalid stored selection as a publication reference."
)


ACTION_TARGET_PROJECTION_INSTRUCTIONS = (
    "Evolving per-claim projection: projection.targets is the legal target/action menu for this "
    "claim. A conditional target names an earlier proposed new independent I publication via "
    "target_claim_index (zero-based), NOT stored truth or a guessed frame ID. Its dependency "
    "must actually publish an eligible new I or the event will refuse atomically. Always supply "
    "an explicit action. NOOP/REFINE/SUPERSEDE/TRANSITION require exactly one nonnull legal "
    "target: a positive target_frame_id or a backward target_claim_index from that menu. The "
    "nonselected target may be omitted or null. ADD/CANDIDATE may omit both targets or use null; "
    "neither may select a target. Prior proposed effects consume destructive/branching actions "
    "but do not establish world chronology: use explicit effective_at, observed_at and previous "
    "links, never extraction order. A historical prior state can remain a legal NOOP/REFINE "
    "target after TRANSITION, not another branching TRANSITION. Never reinterpret an invalid "
    "stored selection as a publication reference."
)

ACTION_TARGET_RATIONALE_INSTRUCTIONS = (
    "The rationale is diagnostic prose only: it may be omitted and defaults to the empty "
    "string. If supplied, it must be a string within its declared character bound. Prose never "
    "selects, repairs or substitutes an action or target."
)


def projected_consolidation_module(
    module: dspy.Module,
    projection: ConsolidationProjection,
    *,
    writer_contract: str | None = None,
    semantic_producer_contract: str | None = None,
) -> dspy.Module:
    """Finalize an invocation-local, action-dependent native schema after loading."""
    from gluomem.llm.outputs import projected_decision_model

    action_target_policy = writer_contract in (
        ACTION_TARGET_WRITER_CONTRACT,
        "semantic-state-writer-v4",
    )
    projected = contextual_consolidation_module(module)
    for predictor in projected.predictors():
        signature = _require_signature(predictor)
        fields = _signature_fields(signature)
        decision_type = signature.fields["decision"].annotation
        if decision_type is None:
            raise ValueError("projected consolidation requires a native typed decision")
        rationale = decision_type.model_fields["rationale"]
        limit = min([512, *[m.max_length for m in rationale.metadata if hasattr(m, "max_length")]])
        decision = projected_decision_model(
            projection, max_rationale_length=limit, writer_contract=writer_contract
        )
        fields["decision"] = (decision, deepcopy(signature.fields["decision"]))
        fields["projection"] = (ConsolidationProjection, dspy.InputField())
        if writer_contract == "semantic-state-writer-v4":
            from gluomem.llm.semantic import (
                SemanticClaimContext,
                SemanticProjection,
                SemanticStoredCandidate,
            )

            fields["projection"] = (SemanticProjection, dspy.InputField())
            fields["new_claim_context"] = (
                SemanticClaimContext,
                deepcopy(signature.fields["new_claim_context"]),
            )
            fields["candidates"] = (
                list[SemanticStoredCandidate],
                deepcopy(signature.fields["candidates"]),
            )
        if semantic_producer_contract in (
            "observed-assertion-date-v1",
            "observed-assertion-date-v2",
            "observed-property-identity-v3",
            "observed-assertion-scope-v4",
        ):
            fields["projection"] = (dict, dspy.InputField())
            fields["candidates"] = (list[StoredCandidate], deepcopy(signature.fields["candidates"]))
        property_instructions = ""
        if semantic_producer_contract in (
            "observed-property-identity-v3",
            "observed-assertion-scope-v4",
        ):
            from gluomem.llm.property_identity import PROPERTY_ACTION_INSTRUCTIONS

            property_instructions = "\n\n" + PROPERTY_ACTION_INSTRUCTIONS
        if semantic_producer_contract == "observed-assertion-scope-v4":
            from gluomem.llm.assertion_scope import (
                ASSERTION_SCOPE_INSTRUCTIONS,
                AssertionScopedContext,
            )

            fields["new_claim_context"] = (
                AssertionScopedContext,
                deepcopy(signature.fields["new_claim_context"]),
            )
            property_instructions += "\n\n" + ASSERTION_SCOPE_INSTRUCTIONS
        predictor.signature = dspy.make_signature(
            fields,
            signature.instructions
            + "\n\n"
            + (
                ACTION_TARGET_PROJECTION_INSTRUCTIONS
                if action_target_policy
                else CONSOLIDATION_PROJECTION_INSTRUCTIONS
            )
            + (
                "\n\n"
                + (
                    ACTION_TARGET_RATIONALE_INSTRUCTIONS
                    if action_target_policy
                    else WRITER_RATIONALE_INSTRUCTIONS
                )
                if writer_contract
                else ""
            )
            + (
                "\n\n" + SEMANTIC_EXTRACTION_INSTRUCTIONS
                if writer_contract == "semantic-state-writer-v4"
                else ""
            )
            + property_instructions,
            signature_name=signature.__name__,
        )
    return projected


def make_cognify_signature(allowed_types: Sequence[str]) -> type[dspy.Signature]:
    """CognifyExtraction with ``entity_type`` constrained to a closed enum.

    Prompt-side entity-type gate (``kg.entity_type_prompt_enum``): the
    allowlist becomes a ``Literal`` on a KGEntity subclass, which dspy renders
    as a JSON-schema enum — the extractor cannot invent types at all, instead
    of resolution retyping them after the fact. Built at call time because the
    allowlist is config; the returned signature carries the same instructions
    as CognifyExtraction so only the schema differs.
    """
    literal = Literal[tuple(allowed_types)]  # type: ignore[valid-type]
    constrained_entity = create_model(
        "KGEntity",
        __base__=KGEntity,
        entity_type=(
            literal,
            Field(description="Entity category; one of the allowed values."),
        ),
    )
    # Built via make_signature's dict form: with `from __future__ import
    # annotations` a class-statement signature would stringify the annotation
    # and dspy could not resolve the closure-local constrained model at
    # get_type_hints time.
    return dspy.make_signature(
        {
            "text": (str, dspy.InputField()),
            "entities": (list[constrained_entity], dspy.OutputField()),
            "relations": (list[KGRelation], dspy.OutputField()),
        },
        CognifyExtraction.instructions,
        signature_name="CognifyExtraction",
    )


# Aliases matching Iteration 6 naming conventions.
ExtractEventClaimsSignature = EventClaimExtraction
ConsolidateClaimSignature = ClaimConsolidation


# Task-name -> signature map (config ``dspy.tasks`` keys).
TASK_SIGNATURES: dict[str, type[dspy.Signature]] = {
    "fact_extraction": FactExtraction,
    "event_extraction": EventExtraction,
    "cognify": CognifyExtraction,
    "contradiction": ContradictionCheck,
    "event_claims": EventClaimExtraction,
    "consolidate": ClaimConsolidation,
}

__all__ = [
    "TASK_SIGNATURES",
    "ClaimConsolidation",
    "CognifyExtraction",
    "ConsolidateClaimSignature",
    "ContradictionCheck",
    "EventClaimExtraction",
    "EventExtraction",
    "ExtractEventClaimsSignature",
    "FactExtraction",
    "make_cognify_signature",
]
