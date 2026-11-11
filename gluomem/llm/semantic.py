"""Opt-in semantic writer proposals. Consistency is NOT source entailment.

Keep these types separate: predecessor model schemas and cache keys are contracts.
Labels are source-derived, fallible model proposals; never derive them from a
selected target, memory kind, date, GOP, or candidate prose in host code.
"""

from __future__ import annotations

from typing import Generic, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import TypeVar

from gluomem.llm.outputs import (
    TARGET_REQUIRED_ACTIONS,
    ClaimConsolidationContext,
    ConsolidationProjection,
    ExtractedClaimSpec,
    ProjectedConsolidationDecision,
    StoredCandidate,
)

SEMANTIC_WRITER_CONTRACT = "semantic-state-writer-v4"


class AssertionKey(BaseModel):
    """Source-proposed assertion address, not a certified entity identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    subject: str = Field(min_length=1, max_length=160)
    predicate: str = Field(min_length=1, max_length=160)
    perspective: str = Field(min_length=1, max_length=160)


class ClaimMeaning(BaseModel):
    """Untrusted source-derived identity; independent state has no refinement anchor."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    key: AssertionKey
    value_key: str = Field(min_length=1, max_length=256)
    independently_versioned: bool
    refines: AssertionKey | None

    @model_validator(mode="after")
    def _shape(self):
        if self.independently_versioned == (self.refines is not None):
            raise ValueError("Independent state has no anchor; dependent detail requires one")
        if any(not x.strip() for x in (*self.key.model_dump().values(), self.value_key)):
            raise ValueError("Semantic labels must not be blank")
        return self


_MeaningT = TypeVar("_MeaningT", bound=ClaimMeaning, default=ClaimMeaning)


class SemanticClaimSpec(ExtractedClaimSpec, Generic[_MeaningT]):
    """One claim with an explicit, untrusted source-derived assertion meaning."""

    model_config = ConfigDict(extra="forbid")
    meaning: _MeaningT


class SemanticSourceWitness(BaseModel):
    """Exact input witness; literal binding is not semantic entailment."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    tenant_id: str
    principal_id: str
    namespace: str
    event_id: str
    revision: int
    source_ref: str
    message_id: str | None
    source_role: str
    observed_at: str | None
    verbatim_span: str


class SemanticReceipt(BaseModel):
    """Opaque revision identity plus the exact retained source witness."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    provenance_id: int
    source_event_row_id: int
    source_context: str
    token: str
    witness: SemanticSourceWitness


class SemanticSnapshot(BaseModel):
    """Consistent source-proposed meaning backed by surviving scoped receipts."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    meaning: ClaimMeaning
    receipts: tuple[SemanticReceipt, ...]


_SnapshotT = TypeVar("_SnapshotT", bound=SemanticSnapshot, default=SemanticSnapshot)


class SemanticStoredCandidate(StoredCandidate, Generic[_SnapshotT]):
    semantic: _SnapshotT | None = None


class SemanticClaimContext(ClaimConsolidationContext):
    meaning: ClaimMeaning
    witness: SemanticSourceWitness


class TargetMeaning(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    frame_id: int | None = None
    claim_index: int | None = None
    meaning: ClaimMeaning | None
    semantic: SemanticSnapshot | None = None

    @model_validator(mode="after")
    def _observed_meaning(self):
        if self.frame_id is not None and self.meaning != (
            self.semantic.meaning if self.semantic is not None else None
        ):
            raise ValueError("Semantic target meaning differs from its observed source binding")
        return self


def permits(action: str, new: ClaimMeaning, target: ClaimMeaning | None) -> bool:
    if target is None:
        return False
    if action == "REFINE":
        return (
            not new.independently_versioned
            and target.independently_versioned
            and new.refines == target.key
        )
    if action == "TRANSITION":
        return (
            new.independently_versioned and target.independently_versioned and new.key == target.key
        )
    same = (
        new.key == target.key
        and new.refines == target.refines
        and new.independently_versioned == target.independently_versioned
    )
    if action == "SUPERSEDE":
        return same
    return action == "NOOP" and same and new.value_key == target.value_key


class SemanticProjection(ConsolidationProjection):
    meaning: ClaimMeaning
    target_meanings: tuple[TargetMeaning, ...]

    @property
    def untargeted_actions(self) -> tuple[Literal["ADD", "CANDIDATE"], ...]:
        return ("ADD", "CANDIDATE") if self.meaning.independently_versioned else ("CANDIDATE",)

    @model_validator(mode="after")
    def _semantic_menu(self):
        keys = [(t.frame_id, t.claim_index) for t in self.targets]
        if keys != [(m.frame_id, m.claim_index) for m in self.target_meanings]:
            raise ValueError("Semantic menu must bind every observed target in order")
        for target, binding in zip(self.targets, self.target_meanings, strict=True):
            if any(not permits(a, self.meaning, binding.meaning) for a in target.allowed_actions):
                raise ValueError("Semantic menu cannot expand target authority")
        return self

    def validate_candidates(self, candidates):
        super().validate_candidates(
            [StoredCandidate.model_validate(c.model_dump()) for c in candidates]
        )
        observed = {m.frame_id: m for m in self.target_meanings if m.frame_id is not None}
        if any(
            not isinstance(c, SemanticStoredCandidate)
            or observed[c.frame_id].semantic != c.semantic
            for c in candidates
        ):
            raise ValueError("Semantic candidate bindings differ from their observed menu")

    def validate_decision(self, decision: ProjectedConsolidationDecision):
        checked = super().validate_decision(decision)
        if (
            checked.action not in TARGET_REQUIRED_ACTIONS
            and checked.action not in self.untargeted_actions
        ):
            raise ValueError("Dependent detail requires an explicit anchor target or review")
        return checked


def source_witness(event, spec):
    from gluomem.storage.scope import resolve_scope
    from gluomem.write.source_binding import bind_source_claim

    binding = bind_source_claim(
        event,
        message_id=spec.message_id,
        quote=spec.verbatim_span,
        capture_policy=event.capture_policy,
        writer_contract=SEMANTIC_WRITER_CONTRACT,
    )
    if spec.meaning.key.perspective != binding.role or (
        spec.meaning.refines is not None and spec.meaning.refines.perspective != binding.role
    ):
        raise ValueError("Semantic perspective must match the bound source role")
    return SemanticSourceWitness(
        **resolve_scope(event.scope).as_dict(),
        event_id=event.event_id,
        revision=event.revision,
        source_ref=event.source_ref,
        message_id=binding.message_id,
        source_role=binding.role,
        observed_at=binding.observed_at,
        verbatim_span=binding.verbatim_span,
    )
