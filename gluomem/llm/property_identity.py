"""Invocation-local property slots; labels and aliases remain source proposals."""

from pydantic import BaseModel, ConfigDict

from gluomem.llm.outputs import ProjectedCandidate
from gluomem.llm.semantic import AssertionKey, SemanticSourceWitness

PROPERTY_IDENTITY_INSTRUCTIONS = (
    "Property-slot alignment is value-independent and date-independent: identify the subject, "
    "property and perspective before comparing values or world-effective/source-observation dates. "
    "A changed value, later date or different source wording does not by itself create a different "
    "property. The source-proposed labels are fallible, not canonical identifiers. source_alias "
    "means a fallible invocation-local proposal that different subject/property labels denote "
    "the same slot, supported by the supplied source claims/spans; it is not stored entity "
    "identity, transitive alias entailment, or permission to merge merely similar topics. "
    "Never rewrite the source-only new meaning or copy a selected candidate's labels into it. "
    "Choose assertion_relation before action selection: state_observation is the same property "
    "and perspective observed in a state (possibly a new value/effective date), not a declaration "
    "that the earlier assertion was mistaken; mistaken_assertion explicitly corrects an erroneous "
    "earlier assertion, not an ordinary change over time; dependent_refinement is genuinely "
    "dependent detail with its exact observed independent anchor; unrelated is a different "
    "subject/property/perspective or an independent error report about an assertion; unresolved "
    "means insufficient evidence or ambiguous targets. A negative report mentioning an old value "
    "is not automatically a correction of that value assertion. Use same_assertion for "
    "state_observation/mistaken_assertion, refines_assertion for dependent_refinement, independent "
    "for unrelated and unresolved for unresolved. For linked proposals, same_label denotes exact "
    "source-label equality and source_alias denotes a proposed equivalence despite different "
    "labels; for refinement compare the proposed refinement anchor to the observed slot. "
    "Nonlinked proposals use not_aligned and no target. Never select the first or nearest target "
    "when ambiguous; return unresolved with no reference. A source_alias cannot cross perspective "
    "or source role. Values and dates remain separate evidence for the later action's legal menu, "
    "not slot identity. Source spans, receipt tokens and exact stored/prior-publication references "
    "are revocable observations; schema validity and rationale never certify semantic truth. "
    "Use only supplied references; no forward references, guessed tokens or persisted alias rules."
)

PROPERTY_ACTION_INSTRUCTIONS = (
    "The observed-property-identity-v3 proposal separately classifies slot identity and assertion "
    "relation without rewriting source meaning. state_observation may use a legal TRANSITION "
    "for ordinary dated state or an exact legal NOOP, never a correction SUPERSEDE; "
    "mistaken_assertion may use a legal SUPERSEDE, never an ordinary TRANSITION; "
    "dependent_refinement may use only its exact legal REFINE anchor. unrelated/unresolved "
    "do not authorize a targeted operation. The actual supplied menu and structural/source "
    "guards remain authoritative; do not invent missing dates, targets, receipts "
    "or alias authority."
)


class PropertySlotSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    receipt_token: str
    witness: SemanticSourceWitness


class PropertySlotObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    target: ProjectedCandidate
    proposed_slot: AssertionKey
    sources: tuple[PropertySlotSource, ...]


class PropertySlotContext(BaseModel):
    """Separate identity coordinates from values/dates in the unchanged contexts."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_proposed_slot: AssertionKey
    source_proposed_refinement_anchor: AssertionKey | None
    observations: tuple[PropertySlotObservation, ...]


def property_slot_context(projection, context, candidates):
    """Project exact observed inputs, never canonicalize or infer an alias."""
    projection.validate_candidates(candidates, context)
    stored = {c.frame_id: c for c in candidates}
    observations = []
    for item in projection.targets:
        snapshot = (
            stored[item.target.frame_id].semantic if item.target.frame_id is not None else None
        )
        sources = (
            tuple(
                PropertySlotSource(receipt_token=r.token, witness=r.witness)
                for r in snapshot.receipts
                if r.token in item.receipt_tokens
            )
            if snapshot
            else ()
        )
        observations.append(
            PropertySlotObservation(
                target=item.target, proposed_slot=item.meaning.key, sources=sources
            )
        )
    return PropertySlotContext(
        source_proposed_slot=context.meaning.key,
        source_proposed_refinement_anchor=context.meaning.refines,
        observations=tuple(observations),
    )
