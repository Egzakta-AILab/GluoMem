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

from datetime import datetime
from typing import TYPE_CHECKING, Literal

import dspy
from pydantic import Field, create_model

from gluomem.llm.outputs import (
    ContradictionVerdict,
    ExtractedEvent,
    ExtractedFact,
    KGEntity,
    KGRelation,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


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


# Task-name -> signature map (config ``dspy.tasks`` keys).
TASK_SIGNATURES: dict[str, type[dspy.Signature]] = {
    "fact_extraction": FactExtraction,
    "event_extraction": EventExtraction,
    "cognify": CognifyExtraction,
    "contradiction": ContradictionCheck,
}
