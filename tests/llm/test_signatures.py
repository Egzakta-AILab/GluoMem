"""Signature drift guards — field names, annotations, verbatim contract docstrings."""

from __future__ import annotations

from datetime import datetime

from gluomem.llm.outputs import (
    ContradictionVerdict,
    ExtractedEvent,
    ExtractedFact,
    KGEntity,
    KGRelation,
    Verdict,
)
from gluomem.llm.signatures import (
    TASK_SIGNATURES,
    CognifyExtraction,
    ContradictionCheck,
    EventExtraction,
    FactExtraction,
    JudgeQA,
)


def _fields(sig) -> dict[str, type]:
    return {name: field.annotation for name, field in sig.model_fields.items()}


def test_fact_extraction_fields() -> None:
    fields = _fields(FactExtraction)
    assert fields["text"] is str
    assert fields["facts"] == list[ExtractedFact]


def test_event_extraction_verbatim_contract() -> None:
    """Verbatim contract — docstring + inputs + typed output."""
    assert (
        EventExtraction.__doc__
        == "Extract episodic events; resolve relative time cues to ISO dates."
    )
    fields = _fields(EventExtraction)
    assert fields["source_text"] is str
    assert fields["reference_date"] is datetime
    assert fields["events"] == list[ExtractedEvent]


def test_cognify_extraction_verbatim_contract() -> None:
    assert CognifyExtraction.__doc__ == "Extract typed entities + relations from a frame."
    fields = _fields(CognifyExtraction)
    assert fields["text"] is str
    assert fields["entities"] == list[KGEntity]
    assert fields["relations"] == list[KGRelation]


def test_contradiction_check_fields() -> None:
    fields = _fields(ContradictionCheck)
    assert fields["new_statement"] is str
    assert fields["stored_statement"] is str
    assert fields["verdict"] is ContradictionVerdict


def test_judge_qa_fields() -> None:
    fields = _fields(JudgeQA)
    assert fields["question"] is str
    assert fields["gold"] is str
    assert fields["prediction"] is str
    assert fields["verdict"] is Verdict


def test_task_signature_map_covers_config_tasks() -> None:
    assert set(TASK_SIGNATURES) == {
        "fact_extraction",
        "event_extraction",
        "cognify",
        "contradiction",
    }


def test_make_cognify_signature_constrains_entity_type() -> None:
    """Prompt-side enum (kg.entity_type_prompt_enum): entity_type becomes a
    closed Literal on a KGEntity subclass; instructions stay verbatim."""
    from typing import Literal, get_args, get_origin

    from gluomem.llm.signatures import make_cognify_signature

    sig = make_cognify_signature(["person", "city"])
    assert sig.instructions == CognifyExtraction.instructions
    entities_ann = sig.model_fields["entities"].annotation
    model = get_args(entities_ann)[0]
    assert issubclass(model, KGEntity)
    et = model.model_fields["entity_type"].annotation
    assert get_origin(et) is Literal and get_args(et) == ("person", "city")
    # The JSON schema dspy renders carries the enum — the extractor cannot
    # invent a type at all.
    schema = model.model_json_schema()
    assert schema["properties"]["entity_type"]["enum"] == ["person", "city"]
    # Relations are untouched.
    assert sig.model_fields["relations"].annotation == list[KGRelation]


def test_build_module_uses_enum_signature_when_configured() -> None:
    from typing import get_args

    from gluomem.config.app_config import load_config
    from gluomem.llm.modules import build_module

    cfg = load_config().with_overrides({"kg.entity_type_prompt_enum": True})
    module = build_module("cognify", cfg)
    model = get_args(module.signature.model_fields["entities"].annotation)[0]
    assert model is not KGEntity and issubclass(model, KGEntity)
    off = build_module("cognify", load_config())
    assert get_args(off.signature.model_fields["entities"].annotation)[0] is KGEntity
