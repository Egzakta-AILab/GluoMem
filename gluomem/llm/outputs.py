"""gluomem.llm.outputs — typed extraction outputs + the ``Extractors`` seam.

This module is DELIBERATELY dspy-free: it is the boundary ``gluomem/write/``
consumes. The write path never imports dspy; it programs against the
``Extractors`` Protocol, which is implemented by ``DspyExtractors``
(``gluomem/llm/modules.py``, dspy-backed), ``FakeExtractors`` (deterministic
offline), and ``NullExtractors`` (no extraction; raw-turns-only baseline).

Every facade method either returns validated Pydantic models or raises
``ExtractionError`` — never a silent empty result on failure.

Mapping onto the KG write surface (``gluomem/kg/graph.py``): ``KGEntity`` →
``kg.create_entity(entity_type=..., name=..., properties=...)``;
``KGRelation.source``/``target`` are entity NAMES (ids do not exist at
extraction time) that ``write/cognify.py`` resolves to ids (alias-normalized,
create-if-missing) before ``kg.create_relation``.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue, create_model, model_validator
from pydantic_core import PydanticUndefined

from gluomem.storage.scope import MemoryKind

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# NOTE ON DOCSTRINGS: pydantic serializes each model's class docstring (and
# every Field ``description``) into the JSON schema that DSPy inlines into the
# LIVE PROMPT. Docstrings below are therefore written FOR THE MODEL — concise
# task signal only. Developer commentary belongs in ``#`` comments or the
# module docstring, which never reach the wire.


class ExtractedFact(BaseModel):
    """One atomic fact that stands alone without pronoun references."""

    text: str
    confidence: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="How directly the text states this fact, 0-1.",
    )


class ExtractedEvent(BaseModel):
    """One episodic event with its absolute date when resolvable."""

    text: str
    event_date: datetime | None = Field(
        default=None,
        description=(
            "Absolute date, resolving relative cues against reference_date; "
            "null when the text gives no date."
        ),
    )
    confidence: float = Field(default=1.0, ge=0, le=1)


class KGEntity(BaseModel):
    """One entity mentioned in the text."""

    entity_type: str = Field(
        description="Lowercase category, e.g. person, city, organization, technology."
    )
    name: str = Field(description="Canonical name as mentioned in the text.")
    properties: dict[str, str] = Field(default_factory=dict)


class KGRelation(BaseModel):
    """One directed relation between two extracted entities."""

    source: str = Field(description="Source entity name; must appear in entities.")
    target: str = Field(description="Target entity name; must appear in entities.")
    relation_type: str = Field(description="UPPER_SNAKE_CASE predicate, e.g. LIVES_IN, WORKS_AT.")
    confidence: float = Field(default=1.0, ge=0, le=1)


# Aggregate for the write path; never referenced by a dspy OutputField, so its
# schema (and this comment) never reaches a prompt.
class CognifyResult(BaseModel):
    entities: list[KGEntity] = Field(default_factory=list)
    relations: list[KGRelation] = Field(default_factory=list)


class ContradictionVerdict(BaseModel):
    """Whether the new statement contradicts the stored statement."""

    is_contradiction: bool
    description: str = Field(
        default="",
        description="One-sentence explanation of the conflict; empty when none.",
    )
    confidence: float = Field(default=1.0, ge=0, le=1)


_DEFAULT_MEMORY_KIND: MemoryKind = "fact"


class ExtractedClaimSpec(BaseModel):
    """One atomic, self-contained memory claim with its kind, provenance, and KG triples."""

    text: str = Field(description="Atomic statement that stands alone without unresolved pronouns.")
    kind: MemoryKind = Field(
        default=_DEFAULT_MEMORY_KIND,
        description=(
            "Memory kind: fact, profile, preference, decision, procedure, episode, "
            "correction, or summary."
        ),
    )
    confidence: float = Field(default=1.0, ge=0, le=1)
    event_date: datetime | None = Field(
        default=None,
        description="Absolute ISO date resolved against reference_date when applicable, else null.",
    )
    verbatim_span: str = Field(
        default="",
        description="Supporting quote from the source message.",
    )
    message_id: str | None = Field(
        default=None,
        description="Optional source message identifier.",
    )
    entities: list[KGEntity] = Field(default_factory=list)
    relations: list[KGRelation] = Field(default_factory=list)


class ClaimConsolidationContext(BaseModel):
    """Authoritative new-claim metadata supplied by the source-bound writer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: MemoryKind
    source_role: str | None = Field(description="Bound source message role; null when unknown.")
    observed_at: str | None = Field(description="Trusted source observation time, not world time.")
    effective_at: str | None = Field(
        description="Explicit world-effective claim date; null means unknown, not observed_at."
    )


class StoredCandidate(BaseModel):
    """One existing active memory frame considered during consolidation."""

    frame_id: int
    content: str
    frame_type: Literal["I", "P", "B"] = "I"
    kind: MemoryKind = "fact"
    source: str = "user_stated"
    event_time: str | None = None
    temporal_operation: Literal["observation", "transition"] | None = None
    previous_frame_id: int | None = None
    observed_at: str | None = None
    effective_at: str | None = None


ConsolidationAction = Literal["ADD", "REFINE", "TRANSITION", "SUPERSEDE", "NOOP", "CANDIDATE"]
TARGET_REQUIRED_ACTIONS: tuple[ConsolidationAction, ...] = (
    "SUPERSEDE",
    "NOOP",
    "REFINE",
    "TRANSITION",
)


def _consolidation_target_schema(schema: dict[str, Any]) -> None:
    actions = schema["properties"]["action"]["enum"]
    schema["anyOf"] = [
        {
            "properties": {
                "action": {"enum": [a for a in actions if a not in TARGET_REQUIRED_ACTIONS]}
            }
        },
        {
            "properties": {
                "action": {"enum": list(TARGET_REQUIRED_ACTIONS)},
                "target_frame_id": {"type": "integer", "minimum": 1},
            },
            "required": ["target_frame_id"],
        },
    ]


class ConsolidationDecision(BaseModel):
    """How a newly extracted claim relates to existing stored memory candidates."""

    model_config = ConfigDict(json_schema_extra=_consolidation_target_schema)

    @model_validator(mode="after")
    def _require_target(self) -> ConsolidationDecision:
        if self.action in TARGET_REQUIRED_ACTIONS and self.target_frame_id is None:
            raise ValueError("Consolidation action requires a positive integer target_frame_id")
        return self

    action: ConsolidationAction = Field(
        description=(
            "ADD if new independent information, REFINE if compatible detail refines an observed "
            "I-frame as a P-frame, NOOP if already captured by target_frame_id, "
            "TRANSITION for an ordinary explicitly dated change retaining the prior I-frame, "
            "SUPERSEDE if it updates/replaces target_frame_id, or CANDIDATE if ambiguous."
        ),
    )
    target_frame_id: int | None = Field(
        default=None,
        strict=True,
        gt=0,
        description=(
            "Positive observed candidate frame id required for NOOP, SUPERSEDE, REFINE "
            "or TRANSITION; "
            "REFINE must select an eligible I-frame, never a P-frame or B-frame; "
            "TRANSITION must select an eligible I-frame with observed temporal metadata; "
            "optional for ADD or an untargeted CANDIDATE review proposal."
        ),
    )
    rationale: str = Field(
        default="",
        description="Brief explanation of why this consolidation action was chosen.",
    )


class ExtractionError(RuntimeError):
    """Raised when an extraction task fails after retries are exhausted."""


CONSOLIDATION_PROJECTION_PROTOCOL = "evolving-per-claim-v1"
WRITER_CONTRACT = "bounded-writer-v2"
ACTION_TARGET_WRITER_CONTRACT = "action-target-writer-v3"
SUPPORTED_WRITER_CONTRACTS = (
    WRITER_CONTRACT,
    ACTION_TARGET_WRITER_CONTRACT,
    "semantic-state-writer-v4",
)


class ProjectedConsolidationDecision(BaseModel):
    """One explicit selection; a claim index names a conditional earlier I publication."""

    model_config = ConfigDict(extra="forbid")

    action: ConsolidationAction
    target_frame_id: int | None = Field(default=None, strict=True, gt=0)
    target_claim_index: int | None = Field(default=None, strict=True, ge=0)
    rationale: str = Field(default="", max_length=512)

    @model_validator(mode="after")
    def _one_target(self) -> ProjectedConsolidationDecision:
        count = int(self.target_frame_id is not None) + int(self.target_claim_index is not None)
        if count != (1 if self.action in TARGET_REQUIRED_ACTIONS else 0):
            raise ValueError(
                "Projected action requires exactly one explicit target, or none for ADD/CANDIDATE"
            )
        return self


class ProjectedCandidate(BaseModel):
    """Observed stored candidate or conditional proposal, never projected stored truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_id: int | None = Field(default=None, strict=True, gt=0)
    claim_index: int | None = Field(default=None, strict=True, ge=0)
    conditional: bool = False
    content: str
    frame_type: Literal["I", "P", "B"] = "I"
    kind: MemoryKind = "fact"
    source: str = "user_stated"
    observed_at: str | None = None
    effective_at: str | None = None
    temporal_operation: Literal["observation", "transition"] | None = None
    previous_frame_id: int | None = None
    previous_claim_index: int | None = None
    allowed_actions: tuple[ConsolidationAction, ...] = ()

    @model_validator(mode="after")
    def _target_shape(self) -> ProjectedCandidate:
        if (self.frame_id is None) == (self.claim_index is None) or self.conditional != (
            self.claim_index is not None
        ):
            raise ValueError(
                "Candidate requires exactly one stored or conditional publication identity"
            )
        if self.conditional and self.frame_type != "I":
            raise ValueError("Only independent I publications have conditional handles")
        if set(self.allowed_actions) - set(TARGET_REQUIRED_ACTIONS) or (
            self.frame_type != "I" and {"REFINE", "TRANSITION"} & set(self.allowed_actions)
        ):
            raise ValueError("Candidate actions must preserve I/P/B target semantics")
        return self


class ConsolidationProjection(BaseModel):
    """Legal per-claim menu after prior proposed effects; order is not world chronology."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: Literal["evolving-per-claim-v1"] = CONSOLIDATION_PROJECTION_PROTOCOL
    claim_index: int = Field(strict=True, ge=0)
    targets: tuple[ProjectedCandidate, ...] = ()

    @model_validator(mode="after")
    def _backward_unique_targets(self) -> ConsolidationProjection:
        identities = set()
        for target in self.targets:
            target = ProjectedCandidate.model_validate(target.model_dump())
            key = (target.frame_id, target.claim_index)
            if key in identities:
                raise ValueError("Projection candidate identities must be unique")
            identities.add(key)
            if target.claim_index is not None and target.claim_index >= self.claim_index:
                raise ValueError("Publication reference must be backward")
        return self

    def validate_candidates(self, candidates: Sequence[StoredCandidate]) -> None:
        stored = {c.frame_id: c for c in candidates}
        projected = {t.frame_id: t for t in self.targets if t.frame_id is not None}
        if stored.keys() != projected.keys() or any(
            getattr(projected[fid], name) != value
            for fid, candidate in stored.items()
            for name, value in candidate.model_dump(exclude={"event_time"}).items()
        ):
            raise ExtractionError("Projection differs from the actual per-claim candidate bucket")

    def validate_decision(
        self, decision: ProjectedConsolidationDecision
    ) -> ProjectedConsolidationDecision:
        checked = ProjectedConsolidationDecision.model_validate(decision.model_dump())
        if checked.action not in TARGET_REQUIRED_ACTIONS:
            return decision
        if (
            checked.target_claim_index is not None
            and checked.target_claim_index >= self.claim_index
        ):
            raise ValueError("Publication reference must be backward")
        if not any(
            t.frame_id == checked.target_frame_id
            and t.claim_index == checked.target_claim_index
            and checked.action in t.allowed_actions
            for t in self.targets
        ):
            raise ValueError("Consolidation target/action is outside this claim's legal projection")
        return decision


def projected_decision_model(
    projection: ConsolidationProjection,
    *,
    max_rationale_length: int = 512,
    writer_contract: str | None = None,
) -> type[ProjectedConsolidationDecision]:
    """Reconstruct the versioned per-call schema/parser without DSPy or saved schemas."""
    if writer_contract is not None and writer_contract not in SUPPORTED_WRITER_CONTRACTS:
        raise ValueError("unknown writer contract")

    optional_targets = writer_contract in (
        ACTION_TARGET_WRITER_CONTRACT,
        "semantic-state-writer-v4",
    )

    def menu_schema(schema):
        branches: list[JsonValue] = [
            {
                "required": ["action"],
                "properties": {
                    "action": {
                        "enum": list(
                            getattr(projection, "untargeted_actions", ("ADD", "CANDIDATE"))
                        )
                    },
                    "target_frame_id": {"type": "null"},
                    "target_claim_index": {"type": "null"},
                },
            }
        ]
        for target in projection.targets:
            if target.allowed_actions:
                branches.append(
                    {
                        "required": [
                            "action",
                            "target_frame_id"
                            if target.frame_id is not None
                            else "target_claim_index",
                        ],
                        "properties": {
                            "action": {"enum": list(target.allowed_actions)},
                            "target_frame_id": (
                                {"const": target.frame_id}
                                if target.frame_id is not None
                                else {"type": "null"}
                            ),
                            "target_claim_index": (
                                {"const": target.claim_index}
                                if target.claim_index is not None
                                else {"type": "null"}
                            ),
                        },
                    }
                )
        schema["anyOf"] = branches

    def legal_menu(self):
        return projection.validate_decision(self)

    return create_model(
        "ProjectedConsolidationDecision",
        __base__=ProjectedConsolidationDecision,
        __config__=ConfigDict(extra="forbid", json_schema_extra=menu_schema),
        # Pydantic installs decorated descriptors directly into the new class;
        # create_model's Callable-only annotation omits that supported form.
        __validators__={
            "_legal_menu": cast("Callable[..., object]", model_validator(mode="after")(legal_menu))
        },
        # V3 preserves the base's null defaults for inapplicable/nonselected
        # targets. The menu branches require the actual selected target, and
        # the inherited validator still enforces exactly one for targeted actions.
        # Legacy/v2 deliberately retain their original unconditional presence.
        target_frame_id=(
            int | None,
            Field(
                default=None if optional_targets else PydanticUndefined,
                strict=True,
                gt=0,
            ),
        ),
        target_claim_index=(
            int | None,
            Field(
                default=None if optional_targets else PydanticUndefined,
                strict=True,
                ge=0,
            ),
        ),
        rationale=(
            str,
            Field(
                default="" if writer_contract else PydanticUndefined,
                strict=True if writer_contract else None,
                max_length=min(512, max_rationale_length),
                description=ConsolidationDecision.model_fields["rationale"].description,
            ),
        ),
    )


@runtime_checkable
class ProjectedConsolidation(Protocol):
    """Optional evolving per-claim capability; no whole-event planning call."""

    @property
    def consolidation_projection_protocol(self) -> str: ...

    def consolidate_claim_with_projection(
        self,
        new_claim: str,
        candidates: Sequence[StoredCandidate],
        *,
        context: ClaimConsolidationContext,
        projection: ConsolidationProjection,
    ) -> ProjectedConsolidationDecision: ...


@runtime_checkable
class ContextualConsolidation(Protocol):
    """Optional metadata-aware capability; the legacy Extractors contract is unchanged."""

    def consolidate_claim_with_context(
        self,
        new_claim: str,
        candidates: Sequence[StoredCandidate],
        *,
        context: ClaimConsolidationContext,
    ) -> ConsolidationDecision: ...


@runtime_checkable
class Extractors(Protocol):
    """The typed extraction facade the write path programs against."""

    def extract_facts(self, text: str) -> list[ExtractedFact]: ...

    def extract_events(
        self, source_text: str, reference_date: datetime
    ) -> list[ExtractedEvent]: ...

    def extract_profile(self, text: str, *, speaker: str) -> list[ExtractedFact]: ...

    def cognify(self, text: str) -> CognifyResult: ...

    def check_contradiction(
        self, new_statement: str, stored_statement: str
    ) -> ContradictionVerdict: ...

    def extract_event_claims(
        self, source_text: str, *, reference_date: datetime | None = None
    ) -> list[ExtractedClaimSpec]: ...

    def consolidate_claim(
        self, new_claim: str, candidates: Sequence[StoredCandidate]
    ) -> ConsolidationDecision: ...
