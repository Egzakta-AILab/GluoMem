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
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

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


class Verdict(BaseModel):
    """Whether the prediction answers the question in agreement with gold."""

    correct: bool
    score: float = Field(ge=0, le=1)
    rationale: str


# Answerer return value for the bench harness; never referenced by a dspy
# OutputField (AnswerQA outputs a plain str), so it is not prompt-visible.
class AnswerResult(BaseModel):
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ExtractionError(RuntimeError):
    """Raised when an extraction task fails after retries are exhausted."""


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
