"""DSPy signature contracts, Pydantic output models, and dspy-free import boundary tests."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from typing import Literal, get_args, get_origin

import pytest
from pydantic import ValidationError

from gluomem.config.app_config import load_config
from gluomem.llm import (
    ContradictionVerdict,
    ExtractedEvent,
    ExtractedFact,
    Extractors,
    FakeExtractors,
    KGEntity,
    KGRelation,
    NullExtractors,
)
from gluomem.llm.modules import build_module
from gluomem.llm.signatures import (
    TASK_SIGNATURES,
    CognifyExtraction,
    ContradictionCheck,
    EventExtraction,
    FactExtraction,
    make_cognify_signature,
)


def _fields(sig) -> dict[str, type]:
    return {name: field.annotation for name, field in sig.model_fields.items()}


def test_output_models_validation_and_protocol_conformance() -> None:
    with pytest.raises(ValidationError):
        ExtractedFact(text="x", confidence=1.5)
    with pytest.raises(ValidationError):
        ExtractedEvent(text="x", confidence=-0.1)

    fact = ExtractedFact(text="Alice lives in Boston.")
    assert fact.confidence == 1.0
    assert ExtractedEvent(text="moved").event_date is None
    assert KGEntity(entity_type="person", name="alice").properties == {}
    v = ContradictionVerdict(is_contradiction=False)
    assert v.description == "" and v.confidence == 1.0

    assert isinstance(FakeExtractors(banner=False), Extractors)
    assert isinstance(NullExtractors(), Extractors)


def test_importing_llm_seam_never_imports_dspy() -> None:
    code = "import sys; import gluomem.llm; sys.exit(1 if 'dspy' in sys.modules else 0)"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_signature_fields_and_docstrings() -> None:
    assert _fields(FactExtraction) == {"text": str, "facts": list[ExtractedFact]}
    assert (
        EventExtraction.__doc__
        == "Extract episodic events; resolve relative time cues to ISO dates."
    )
    assert _fields(EventExtraction) == {
        "source_text": str,
        "reference_date": datetime,
        "events": list[ExtractedEvent],
    }
    assert CognifyExtraction.__doc__ == "Extract typed entities + relations from a frame."
    assert _fields(CognifyExtraction) == {
        "text": str,
        "entities": list[KGEntity],
        "relations": list[KGRelation],
    }
    assert _fields(ContradictionCheck) == {
        "new_statement": str,
        "stored_statement": str,
        "verdict": ContradictionVerdict,
    }
    assert set(TASK_SIGNATURES) == {
        "fact_extraction",
        "event_extraction",
        "cognify",
        "contradiction",
        "event_claims",
        "consolidate",
    }


def test_make_cognify_signature_and_build_module_enum() -> None:
    sig = make_cognify_signature(["person", "city"])
    assert sig.instructions == CognifyExtraction.instructions
    model = get_args(sig.model_fields["entities"].annotation)[0]
    assert issubclass(model, KGEntity)
    et = model.model_fields["entity_type"].annotation
    assert get_origin(et) is Literal and get_args(et) == ("person", "city")
    assert model.model_json_schema()["properties"]["entity_type"]["enum"] == [
        "person",
        "city",
    ]

    cfg = load_config().with_overrides({"kg.entity_type_prompt_enum": True})
    module = build_module("cognify", cfg)
    mod_model = get_args(module.signature.model_fields["entities"].annotation)[0]
    assert mod_model is not KGEntity and issubclass(mod_model, KGEntity)
    off = build_module("cognify", load_config())
    assert get_args(off.signature.model_fields["entities"].annotation)[0] is KGEntity
