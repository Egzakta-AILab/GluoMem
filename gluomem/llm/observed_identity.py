"""Ephemeral observed relation authority; never a source-only meaning rewrite."""

from collections.abc import Callable
from copy import deepcopy
from typing import Generic, Literal, Protocol, cast, overload, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue, create_model, model_validator
from pydantic_core import PydanticUndefined
from typing_extensions import TypeVar

from gluomem.llm.outputs import (
    TARGET_REQUIRED_ACTIONS,
    ConsolidationProjection,
    ExtractedClaimSpec,
    ProjectedCandidate,
    StoredCandidate,
)
from gluomem.llm.semantic import (
    ClaimMeaning,
    SemanticClaimContext,
    SemanticClaimSpec,
    SemanticStoredCandidate,
    TargetMeaning,
    permits,
)

PRODUCER_CONTRACT = "observed-assertion-date-v1"
PREPARATION_CONTRACT = "observed-assertion-date-v2"
PROPERTY_IDENTITY_CONTRACT = "observed-property-identity-v3"
ASSERTION_SCOPE_CONTRACT = "observed-assertion-scope-v4"
PRODUCER_CONTRACTS = (
    PRODUCER_CONTRACT,
    PREPARATION_CONTRACT,
    PROPERTY_IDENTITY_CONTRACT,
    ASSERTION_SCOPE_CONTRACT,
)
SAFE_PREPARATION_CONTRACTS = (
    PREPARATION_CONTRACT,
    PROPERTY_IDENTITY_CONTRACT,
    ASSERTION_SCOPE_CONTRACT,
)


_ClaimT = TypeVar("_ClaimT", bound=ExtractedClaimSpec)


@overload
def observed_claim_type() -> type[SemanticClaimSpec]: ...


@overload
def observed_claim_type(claim_type: type[_ClaimT]) -> type[_ClaimT]: ...


def observed_claim_type(
    claim_type: type[ExtractedClaimSpec] = SemanticClaimSpec,
) -> type[ExtractedClaimSpec]:
    """Dspy-free required-nullable overlay preserving all inherited constraints."""
    date = deepcopy(claim_type.model_fields["event_date"])
    date.default = PydanticUndefined
    date.default_factory = None
    return create_model(
        "ObservedDatedClaimSpec", __base__=claim_type, event_date=(date.annotation, date)
    )


class IdentityProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    relation: Literal["same_assertion", "refines_assertion", "independent", "unresolved"]
    target_frame_id: int | None = Field(default=None, strict=True, gt=0)
    target_claim_index: int | None = Field(default=None, strict=True, ge=0)
    rationale: str = Field(default="", max_length=512)

    @model_validator(mode="after")
    def _reference(self):
        linked = self.relation in {"same_assertion", "refines_assertion"}
        if sum(x is not None for x in (self.target_frame_id, self.target_claim_index)) != int(
            linked
        ):
            raise ValueError("Identity relation requires exactly one observed reference, or none")
        return self


class PropertyIdentityProposal(IdentityProposal):
    """Fallible invocation-local alignment, never a replacement source meaning."""

    assertion_relation: Literal[
        "state_observation", "mistaken_assertion", "dependent_refinement", "unrelated", "unresolved"
    ]
    subject_alignment: Literal["same_label", "source_alias", "not_aligned"]
    property_alignment: Literal["same_label", "source_alias", "not_aligned"]

    @model_validator(mode="after")
    def _classification(self):
        expected = {
            "state_observation": "same_assertion",
            "mistaken_assertion": "same_assertion",
            "dependent_refinement": "refines_assertion",
            "unrelated": "independent",
            "unresolved": "unresolved",
        }
        if self.relation != expected[self.assertion_relation]:
            raise ValueError("Property identity relation and assertion classification disagree")
        linked = self.relation in {"same_assertion", "refines_assertion"}
        if any(
            (a != "not_aligned") != linked
            for a in (self.subject_alignment, self.property_alignment)
        ):
            raise ValueError("Property alignment requires an explicit linked reference, or none")
        return self


class IdentityTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    target: ProjectedCandidate
    meaning: ClaimMeaning
    receipt_tokens: tuple[str, ...]


class IdentityContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    claim_index: int = Field(strict=True, ge=0)
    targets: tuple[IdentityTarget, ...] = ()

    @model_validator(mode="after")
    def _ordered_refs(self):
        ConsolidationProjection(
            claim_index=self.claim_index, targets=tuple(t.target for t in self.targets)
        )
        return self

    def validate_candidates(self, candidates, context):
        stored = {c.frame_id: c for c in candidates}
        for item in self.targets:
            t = item.target
            if t.frame_id is None:
                continue  # Backward invocation-owned publications are bound by the commit guard.
            candidate = stored.get(t.frame_id)
            snapshot = getattr(candidate, "semantic", None)
            if (
                candidate is None
                or snapshot is None
                or snapshot.meaning != item.meaning
                or not item.receipt_tokens
                or not set(item.receipt_tokens) <= {r.token for r in snapshot.receipts}
                or any(
                    getattr(t, n) != v
                    for n, v in candidate.model_dump(exclude={"event_time", "semantic"}).items()
                )
                or any(
                    (r.witness.tenant_id, r.witness.principal_id, r.witness.namespace)
                    != (
                        context.witness.tenant_id,
                        context.witness.principal_id,
                        context.witness.namespace,
                    )
                    for r in snapshot.receipts
                )
            ):
                raise ValueError("Identity context differs from the observed candidate receipts")

    def validate_proposal(self, proposal, context):
        if isinstance(proposal, PropertyIdentityProposal):
            p = PropertyIdentityProposal.model_validate(proposal.model_dump())
        else:
            p = IdentityProposal.model_validate(proposal.model_dump())
        if p.relation in {"independent", "unresolved"}:
            return p
        selected = next(
            (
                t
                for t in self.targets
                if (t.target.frame_id, t.target.claim_index)
                == (p.target_frame_id, p.target_claim_index)
            ),
            None,
        )
        if selected is None:
            raise ValueError("Identity reference is outside the observed surviving context")
        new, old = context.meaning, selected.meaning
        if new.key.perspective != old.key.perspective or new.key.perspective != context.source_role:
            raise ValueError("Identity relation cannot cross source roles")
        if p.relation == "same_assertion":
            if new.independently_versioned != old.independently_versioned or (
                not new.independently_versioned and new.refines != old.refines
            ):
                raise ValueError("Identity same-assertion shape is incompatible")
        elif new.independently_versioned or not old.independently_versioned:
            raise ValueError(
                "Identity refinement requires dependent detail and an independent anchor"
            )
        if isinstance(p, PropertyIdentityProposal):
            slot = new.refines if p.relation == "refines_assertion" else new.key
            for field, alignment in (
                ("subject", p.subject_alignment),
                ("predicate", p.property_alignment),
            ):
                expected = (
                    "same_label"
                    if getattr(slot, field) == getattr(old.key, field)
                    else "source_alias"
                )
                if alignment != expected:
                    raise ValueError(
                        "Property alignment basis differs from the source-proposed labels"
                    )
        return p


@runtime_checkable
class ObservedIdentityAlignment(Protocol):
    @property
    def semantic_producer_capability(self) -> str: ...
    def align_claim_identity(
        self, new_claim, candidates, *, context, projection
    ) -> IdentityProposal: ...


def identity_proposal_model(
    projection, context, *, max_rationale_length=512, producer_contract=PRODUCER_CONTRACT
):
    """Schema and parser share the invocation's observed reference alternatives."""
    property_identity = producer_contract in (PROPERTY_IDENTITY_CONTRACT, ASSERTION_SCOPE_CONTRACT)
    base = PropertyIdentityProposal if property_identity else IdentityProposal
    choices: list[JsonValue] = [
        {
            "properties": {
                "relation": {"enum": ["independent", "unresolved"]},
                "target_frame_id": {"type": "null"},
                "target_claim_index": {"type": "null"},
            }
        }
    ]
    for item in projection.targets:
        t = item.target
        relations = []
        for relation in ("same_assertion", "refines_assertion"):
            try:
                projection.validate_proposal(
                    IdentityProposal(
                        relation=relation,
                        target_frame_id=t.frame_id,
                        target_claim_index=t.claim_index,
                    ),
                    context,
                )
            except ValueError:
                continue
            relations.append(relation)
        if not relations:
            continue
        if property_identity:
            for relation in relations:
                slot = (
                    context.meaning.refines
                    if relation == "refines_assertion"
                    else context.meaning.key
                )
                choices.append(
                    {
                        "properties": {
                            "relation": {"const": relation},
                            "target_frame_id": {"const": t.frame_id},
                            "target_claim_index": {"const": t.claim_index},
                            "assertion_relation": {
                                "enum": ["state_observation", "mistaken_assertion"]
                                if relation == "same_assertion"
                                else ["dependent_refinement"]
                            },
                            "subject_alignment": {
                                "const": "same_label"
                                if slot.subject == item.meaning.key.subject
                                else "source_alias"
                            },
                            "property_alignment": {
                                "const": "same_label"
                                if slot.predicate == item.meaning.key.predicate
                                else "source_alias"
                            },
                        },
                        "required": ["target_frame_id", "target_claim_index"],
                    }
                )
            continue
        choice: JsonValue = {
            "properties": {
                "relation": {"enum": list(relations)},
                "target_frame_id": {"const": t.frame_id},
                "target_claim_index": {"const": t.claim_index},
            },
            "required": ["target_frame_id", "target_claim_index"],
        }
        choices.append(choice)

    @model_validator(mode="after")
    def observed(self):
        projection.validate_proposal(self, context)
        return self

    if property_identity:
        independent_choices: list[JsonValue] = [
            {
                "properties": {
                    "relation": {"const": relation},
                    "target_frame_id": {"type": "null"},
                    "target_claim_index": {"type": "null"},
                    "assertion_relation": {"const": classification},
                    "subject_alignment": {"const": "not_aligned"},
                    "property_alignment": {"const": "not_aligned"},
                }
            }
            for relation, classification in (
                ("independent", "unrelated"),
                ("unresolved", "unresolved"),
            )
        ]
        choices[:1] = independent_choices
    return create_model(
        "ObservedPropertyIdentityProposal" if property_identity else "ObservedIdentityProposal",
        __base__=base,
        __config__=ConfigDict(json_schema_extra={"anyOf": choices}),
        # create_model installs decorated descriptors in the class namespace;
        # its Callable-only annotation omits this supported Pydantic form.
        __validators__={"observed": cast("Callable[..., object]", observed)},
        rationale=(str, Field(default="", max_length=min(512, max_rationale_length))),
    )


_ProducerT = TypeVar(
    "_ProducerT",
    bound=str,
    covariant=True,
    default=Literal[
        "observed-assertion-date-v1", "observed-assertion-date-v2", "observed-property-identity-v3"
    ],
)


class ObservedSemanticProjection(ConsolidationProjection, Generic[_ProducerT]):
    """Private frozen authority. Native tasks receive wire_view, not full receipts."""

    producer_contract: _ProducerT = cast(_ProducerT, PRODUCER_CONTRACT)
    meaning: ClaimMeaning
    context: SemanticClaimContext
    target_meanings: tuple[TargetMeaning, ...]
    structural: ConsolidationProjection
    identity_context: IdentityContext
    proposal: IdentityProposal | PropertyIdentityProposal
    identity_call: Literal["native", "empty-context"]

    @property
    def untargeted_actions(self) -> tuple[Literal["ADD", "CANDIDATE"], ...]:
        return ("ADD", "CANDIDATE") if self.meaning.independently_versioned else ("CANDIDATE",)

    def permits_relation(self, action, target, binding):
        if binding.meaning is None:
            return False
        if self.producer_contract in (PROPERTY_IDENTITY_CONTRACT, ASSERTION_SCOPE_CONTRACT):
            p = self.proposal
            if not isinstance(p, PropertyIdentityProposal) or (
                target.frame_id,
                target.claim_index,
            ) != (p.target_frame_id, p.target_claim_index):
                return False
            if p.assertion_relation == "state_observation":
                return (action == "NOOP" and permits(action, self.meaning, binding.meaning)) or (
                    action == "TRANSITION" and self.meaning.independently_versioned
                )
            return (p.assertion_relation == "mistaken_assertion" and action == "SUPERSEDE") or (
                p.assertion_relation == "dependent_refinement" and action == "REFINE"
            )
        if action == "NOOP":
            return permits(action, self.meaning, binding.meaning)
        p = self.proposal
        if (target.frame_id, target.claim_index) != (p.target_frame_id, p.target_claim_index):
            return False
        if p.relation == "same_assertion":
            return action == "SUPERSEDE" or (
                action == "TRANSITION" and self.meaning.independently_versioned
            )
        return p.relation == "refines_assertion" and action == "REFINE"

    @model_validator(mode="after")
    def _bound_menu(self):
        if (
            self.producer_contract in (PROPERTY_IDENTITY_CONTRACT, ASSERTION_SCOPE_CONTRACT)
        ) != isinstance(self.proposal, PropertyIdentityProposal):
            raise ValueError("Property identity proposal requires its exact producer contract")
        if self.meaning != self.context.meaning or self.claim_index != self.structural.claim_index:
            raise ValueError("Observed identity source binding changed")
        if self.identity_context.claim_index != self.claim_index:
            raise ValueError("Observed identity index changed")
        self.identity_context.validate_proposal(self.proposal, self.context)
        if (self.identity_call == "empty-context") != (not self.identity_context.targets):
            raise ValueError("Identity no-call path requires truly empty context")
        refs = [(t.frame_id, t.claim_index) for t in self.structural.targets]
        if refs != [(t.frame_id, t.claim_index) for t in self.targets] or refs != [
            (b.frame_id, b.claim_index) for b in self.target_meanings
        ]:
            raise ValueError("Observed identity menu must retain ordered structural references")
        for t, original, b in zip(
            self.targets, self.structural.targets, self.target_meanings, strict=True
        ):
            if t.model_dump(exclude={"allowed_actions"}) != original.model_dump(
                exclude={"allowed_actions"}
            ):
                raise ValueError("Observed identity structural target changed")
            if any(
                a not in original.allowed_actions or not self.permits_relation(a, t, b)
                for a in t.allowed_actions
            ):
                raise ValueError("Observed identity cannot expand structural relation authority")
        for item in self.identity_context.targets:
            key = (item.target.frame_id, item.target.claim_index)
            if key not in refs:
                raise ValueError("Identity context invented a target")
            index = refs.index(key)
            binding = self.target_meanings[index]
            if item.target != self.structural.targets[index] or item.meaning != binding.meaning:
                raise ValueError("Identity context differs from frozen target")
            if item.target.frame_id is not None and (
                binding.semantic is None
                or not item.receipt_tokens
                or not set(item.receipt_tokens) <= {r.token for r in binding.semantic.receipts}
            ):
                raise ValueError("Identity context requires surviving receipt tokens")
        return self

    def validate_candidates(self, candidates):
        ConsolidationProjection.validate_candidates(
            self, [StoredCandidate.model_validate(c.model_dump()) for c in candidates]
        )
        observed = {b.frame_id: b.semantic for b in self.target_meanings if b.frame_id is not None}
        if any(
            not isinstance(c, SemanticStoredCandidate) or c.semantic != observed[c.frame_id]
            for c in candidates
        ):
            raise ValueError("Observed identity candidates changed")

    def validate_decision(self, decision):
        checked = super().validate_decision(decision)
        if (
            checked.action not in TARGET_REQUIRED_ACTIONS
            and checked.action not in self.untargeted_actions
        ):
            raise ValueError("Dependent detail requires an explicit anchor target or review")
        return checked

    def wire_view(self):
        return {
            "producer_contract": self.producer_contract,
            "claim_index": self.claim_index,
            "targets": [t.model_dump(mode="json") for t in self.targets],
            "identity_context": self.identity_context.model_dump(mode="json"),
            "proposal": self.proposal.model_dump(mode="json"),
        }
