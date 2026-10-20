import json
from typing import Annotated, Literal, cast

import dspy
import pytest
from dspy.utils.exceptions import AdapterParseError
from pydantic import BaseModel, Field

from gluomem.llm.router import LMRouter, _make_adapter
from gluomem.llm.signatures import TASK_SIGNATURES


@pytest.mark.parametrize(
    "payload",
    [
        {"decision": {}},
        {"decision": {"target_frame_id": None, "rationale": "no action"}},
    ],
)
@pytest.mark.parametrize("canonicalize", [False, True])
def test_semantic_literal_never_created_from_defaults(payload, canonicalize):
    with pytest.raises(AdapterParseError):
        _make_adapter("json", schema_mode="required", canonicalize=canonicalize).parse(
            TASK_SIGNATURES["consolidate"], json.dumps(payload)
        )


def test_extractor_policy_changes_require_explicit_reconstruction(tmp_path):
    from gluomem.config.app_config import AppConfig
    from gluomem.llm.modules import DspyExtractors
    from gluomem.llm.structured_adapter import output_schema

    cfg = AppConfig(
        dspy={
            "adapter": "json",
            "schema_mode": "required",
            "artifacts": {"dir": str(tmp_path)},
            "output_field_limits": {"consolidate": {"decision.rationale": 37}},
        }
    )
    extractor = DspyExtractors(cfg)
    cfg.dspy.output_field_limits["consolidate"]["decision.rationale"] = 83
    for instance, expected in [(extractor, 37), (DspyExtractors(cfg), 83)]:
        signature = instance._module("consolidate").signature
        field = output_schema(signature)["$defs"]["ConsolidationDecision"]["properties"][
            "rationale"
        ]
        assert field["maxLength"] == expected


def test_run_wrapper_categorical_preserves_cause():
    from contextlib import nullcontext

    from gluomem.config.app_config import AppConfig
    from gluomem.llm.modules import DspyExtractors
    from gluomem.llm.outputs import ExtractionError

    original = ValueError("private-marker-must-not-escape")

    class Router:
        def bind(self, task):
            return nullcontext()

    fx = DspyExtractors(AppConfig(), router=cast(LMRouter, Router()))

    def fail(**kwargs):
        raise original

    fx._modules["event_claims"] = cast(dspy.Module, fail)
    with pytest.raises(ExtractionError) as caught:
        fx.extract_event_claims("safe authored input")
    assert str(caught.value) == "structured extraction failed"
    assert caught.value.__cause__ is original


def test_validation_retry_and_refine_are_mutually_exclusive():
    from gluomem.config.app_config import DspyConfig, DspyTaskConfig

    with pytest.raises(ValueError):
        DspyConfig(
            model="mock",
            adapter="json",
            validation_retries=1,
            tasks={
                "sample": DspyTaskConfig(
                    model="mock", module="predict", refine={"enabled": True, "N": 2}
                )
            },
        )


@pytest.mark.parametrize(
    "policy", [{"canonicalize": True}, {"schema_mode": "required"}, {"validation_retries": 1}]
)
def test_structured_options_reject_chat_during_config_validation(policy):
    from gluomem.config.app_config import DspyConfig

    with pytest.raises(ValueError, match="json or baml"):
        DspyConfig(model="mock", **policy)


@pytest.mark.parametrize(
    "payload",
    [
        '{"claims":[],"CLAIMS":[]}',
        '{"claims":[],"claims":[]}',
        '{"claims":[],"other":"do not discard"}',
        '[{"claims":[]}]',
        '{"claims":[{"text":"x","kind":"request"}]}',
        '{"claims":[{"text":"x","kind":"fact","event_date":"2023-02-29T00:00:00Z"}]}',
        '{"claims":[{"text":"x","kind":"fact","relations":[{"source":"A","target":"B"}]}]}',
    ],
)
def test_adversarial_rejection(payload):
    for name in ["json", "baml"]:
        with pytest.raises(AdapterParseError):
            _make_adapter(name, canonicalize=True).parse(TASK_SIGNATURES["event_claims"], payload)


def test_generic_nested_types_and_no_free_text_normalization():
    class Item(BaseModel):
        status: Literal["on", "off"]
        score: Annotated[int, Field(ge=0)]
        quote: str
        meta: dict[str, str]

    class Sig(dspy.Signature):
        items: list[Item | None] = dspy.OutputField()

    result = _make_adapter("baml", canonicalize=True).parse(
        Sig,
        json.dumps(
            {
                "ITEMS": [
                    None,
                    {
                        "STATUS": " ON ",
                        "SCORE": 2,
                        "QUOTE": " Keep CASE ",
                        "META": {"ArbitraryKey": "  UNCHANGED "},
                    },
                ]
            }
        ),
    )
    assert result["items"][1].quote == " Keep CASE "
    assert result["items"][1].meta == {"ArbitraryKey": "  UNCHANGED "}
    with pytest.raises(AdapterParseError):
        _make_adapter("json", canonicalize=True).parse(
            Sig,
            json.dumps({"items": [{"status": "on", "score": -1, "quote": "x", "meta": {}}]}),
        )


def test_ambiguous_model_keys_and_literal_rejected():
    class Collision(BaseModel):
        key: int
        KEY: int

    class Sig(dspy.Signature):
        data: Collision = dspy.OutputField()

    with pytest.raises(AdapterParseError):
        _make_adapter("json", canonicalize=True).parse(Sig, '{"data":{"key":1,"KEY":2}}')

    class Lit(dspy.Signature):
        data: Literal["a", "A"] = dspy.OutputField()

    with pytest.raises(AdapterParseError):
        _make_adapter("json", canonicalize=True).parse(Lit, '{"data":" a "}')
