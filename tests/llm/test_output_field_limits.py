import json

import pytest
from pydantic import TypeAdapter, ValidationError

from gluomem.llm import structured_adapter as sa
from gluomem.llm.output_policy import configured_signature
from gluomem.llm.signatures import ClaimConsolidation


def test_native_limit():
    limit = 37
    signature = configured_signature(ClaimConsolidation, {"decision.rationale": limit})
    schema = sa.output_schema(signature)
    assert schema["$defs"]["ConsolidationDecision"]["properties"]["rationale"]["maxLength"] == limit
    native = signature.output_fields["decision"].annotation
    assert (
        TypeAdapter(native).validate_python({"action": "ADD", "rationale": "é" * limit}).rationale
        == "é" * limit
    )
    with pytest.raises(ValidationError):
        TypeAdapter(native).validate_python({"action": "ADD", "rationale": "x" * (limit + 1)})
    assert configured_signature(ClaimConsolidation, {}) is ClaimConsolidation
    assert "maxLength" not in json.dumps(sa.output_schema(ClaimConsolidation))
    original = sa.output_schema(ClaimConsolidation)
    del schema["$defs"]["ConsolidationDecision"]["properties"]["rationale"]["maxLength"]
    assert schema == original


def test_unsupported_scalar_output_is_rejected_not_silently_ignored():
    import dspy

    signature = dspy.make_signature(
        {"text": (str, dspy.InputField()), "answer": (str, dspy.OutputField())}
    )
    with pytest.raises(ValueError, match="model"):
        configured_signature(signature, {"answer": 5})


def test_open_map_array_and_literal_paths_are_not_bypasses():
    from typing import Any, Literal

    import dspy
    from pydantic import BaseModel, Field

    class Native(BaseModel):
        text: str = Field(default="native", description="Retained description.", min_length=2)
        other: str = "untouched"
        mapping: dict[str, Any] = Field(default_factory=dict)
        values: list[str] = Field(default_factory=list)
        action: Literal["ADD", "NOOP"] = "ADD"

    signature = dspy.make_signature({"result": (Native, dspy.OutputField())})
    for path in ["result.mapping.free", "result.values.text", "result.action"]:
        with pytest.raises(ValueError):
            configured_signature(signature, {path: 7})
    cloned = configured_signature(signature, {"result.text": 7})
    native = cloned.output_fields["result"].annotation
    assert native is not None
    assert native.model_validate({}).text == "native"
    assert native.model_fields["text"].description == Native.model_fields["text"].description
    assert native.model_validate({"text": "abcdefg", "other": "x" * 10000}).other == "x" * 10000
    assert native.model_validate({"mapping": {"anything": [1, True]}}).mapping == {
        "anything": [1, True]
    }
    with pytest.raises(ValidationError):
        native.model_validate({"text": "x"})


def policy(limits):
    from gluomem.config.app_config import DspyConfig

    return DspyConfig(
        model="mock", adapter="json", schema_mode="required", output_field_limits=limits
    )


def test_config_reaches_module():
    from gluomem.config.app_config import AppConfig
    from gluomem.llm.bootstrap import load_or_build

    cfg = AppConfig(dspy=policy({"consolidate": {"decision.rationale": 37}}))
    module = load_or_build("consolidate", cfg)
    assert (
        sa.output_schema(module.signature)["$defs"]["ConsolidationDecision"]["properties"][
            "rationale"
        ]["maxLength"]
        == 37
    )
    assert cfg.dspy.output_field_limits == {"consolidate": {"decision.rationale": 37}}


@pytest.mark.parametrize(
    "limits",
    [
        {"unknown": {"decision.rationale": 1}},
        {"consolidate": {"unknown": 1}},
        {"consolidate": {"decision.action": 1}},
        {"consolidate": {"decision.target_frame_id": 1}},
        {"consolidate": {"new_claim": 1}},
        {"fact_extraction": {"facts.content": 37}},
        {"consolidate": {"decision.rationale.": 1}},
    ],
)
def test_invalid_semantic_policy_rejected_at_extractor_initialization(limits):
    from gluomem.config.app_config import AppConfig
    from gluomem.llm.modules import DspyExtractors

    cfg = AppConfig(dspy=policy(limits))
    with pytest.raises(ValueError):
        DspyExtractors(cfg)


@pytest.mark.parametrize("limit", [True, 0, "37", 1.5])
def test_invalid_primitive_limit_rejected_by_config(limit):
    with pytest.raises(ValueError):
        policy({"consolidate": {"decision.rationale": limit}})


def test_no_default_policy():
    from gluomem.config.app_config import DspyConfig

    assert DspyConfig(model="mock").output_field_limits == {}


def test_limits_require_declared_json_schema():
    from gluomem.config.app_config import DspyConfig

    with pytest.raises(ValueError):
        DspyConfig(model="mock", output_field_limits={"consolidate": {"decision.rationale": 1}})


@pytest.mark.parametrize("format", ["json", "program"])
def test_compiled_native_signature_cannot_bypass_limits(tmp_path, format):
    from gluomem.config.app_config import AppConfig
    from gluomem.llm.bootstrap import save_artifact, save_json_artifact
    from gluomem.llm.modules import DspyExtractors, build_module

    unbounded = AppConfig(dspy={"adapter": "json", "schema_mode": "required"})
    original = build_module("consolidate", unbounded)
    import dspy

    assert isinstance(original, dspy.Predict)
    assert original.signature is not None
    original.signature = original.signature.with_instructions("Retained compiled instructions.")
    original.demos = [
        {
            "new_claim": "demo",
            "candidates": [],
            "decision": {"action": "ADD", "rationale": "demo rationale"},
        }
    ]
    saver = save_json_artifact if format == "json" else save_artifact
    saver(original, tmp_path, "consolidate", model_ref="authored-only")
    cfg = AppConfig(dspy=policy({"consolidate": {"decision.rationale": 37}}))
    cfg.dspy.artifacts.dir = str(tmp_path)
    extractors = DspyExtractors(cfg)
    bounded = extractors._module("consolidate")
    assert bounded.signature.instructions == "Retained compiled instructions."
    assert bounded.demos == original.demos
    assert (
        sa.output_schema(bounded.signature)["$defs"]["ConsolidationDecision"]["properties"][
            "rationale"
        ]["maxLength"]
        == 37
    )
    assert "maxLength" not in json.dumps(sa.output_schema(original.signature))
