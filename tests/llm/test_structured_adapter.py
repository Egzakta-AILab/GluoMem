import json
from typing import Any, cast

import pytest
from dspy.utils.exceptions import AdapterParseError

from gluomem.llm.router import _make_adapter
from gluomem.llm.signatures import TASK_SIGNATURES


@pytest.mark.parametrize("asynchronous", [False, True])
def test_validation_feedback_one_same_lm_attempt_at_zero(asynchronous):
    import asyncio

    from dspy.utils.dummies import DummyLM

    class Scripted(DummyLM):
        def __init__(self):
            super().__init__([])
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return (
                ['{"claims":[{"text":"x","kind":"request"}]}']
                if len(self.calls) == 1
                else ['{"claims":[]}']
            )

        async def acall(self, prompt=None, *, messages=None, **kwargs):
            return self(prompt=prompt, messages=messages, **kwargs)

    adapter = _make_adapter("json", canonicalize=True, validation_retries=1)
    lm = Scripted()
    args = (
        lm,
        {"temperature": 0.3, "max_tokens": 123},
        TASK_SIGNATURES["event_claims"],
        [],
        {"source_text": "private-marker", "reference_date": ""},
    )
    result = asyncio.run(adapter.acall(*args)) if asynchronous else adapter(*args)
    assert result == [{"claims": []}]
    assert len(lm.calls) == 2
    assert lm.calls[1]["temperature"] == 0
    assert lm.calls[1]["max_tokens"] == 123
    assert lm.calls[0]["messages"][-1] == lm.calls[1]["messages"][-1]
    feedback = lm.calls[1]["messages"][0]["content"]
    assert "schema_validation" in feedback
    assert "request" not in feedback and "private-marker" not in feedback


@pytest.mark.parametrize("name", ["json", "baml"])
def test_required_schema_is_explicit_lossless_and_capability_gated(name):
    from types import SimpleNamespace

    adapter = _make_adapter(name, schema_mode="required")
    kwargs = {}
    lm = SimpleNamespace(supported_params={"response_format"}, supports_response_schema=True)
    cast(Any, adapter)._prepare_response_format(lm, kwargs, TASK_SIGNATURES["event_claims"])
    wire = kwargs["response_format"]
    assert wire["type"] == "json_schema"
    schema = wire["json_schema"]["schema"]
    definitions = schema["$defs"]
    assert definitions["ExtractedClaimSpec"]["properties"]["kind"]["enum"] == [
        "fact",
        "profile",
        "preference",
        "decision",
        "procedure",
        "episode",
        "correction",
        "summary",
    ]
    assert definitions["KGEntity"]["properties"]["properties"]["additionalProperties"] == {
        "type": "string"
    }
    lm.supports_response_schema = False
    with pytest.raises(ValueError, match="capability"):
        cast(Any, adapter)._prepare_response_format(lm, {}, TASK_SIGNATURES["event_claims"])


def test_config_exposes_opt_in_adapter_policy():
    from gluomem.config.app_config import DspyConfig
    from gluomem.llm.router import LMRouter

    config = DspyConfig(
        adapter="baml", canonicalize=True, schema_mode="required", validation_retries=1
    )
    assert (
        config.canonicalize and config.schema_mode == "required" and config.validation_retries == 1
    )
    with pytest.raises(ValueError):
        DspyConfig.model_validate({"validation_retries": 2})
    from gluomem.config.app_config import AppConfig

    router = LMRouter(AppConfig(dspy=config))
    assert cast(Any, router._adapter).canonicalize


@pytest.mark.parametrize("name", ["json", "baml"])
def test_complete_single_output_envelope_only(name):
    adapter = _make_adapter(name, canonicalize=True)
    value = {"action": " add ", "target_frame_id": None, "rationale": "Exact rationale"}
    assert (
        adapter.parse(TASK_SIGNATURES["consolidate"], json.dumps(value))["decision"].action == "ADD"
    )
    for invalid in (
        {},
        {"action": "ADD"},
        {**value, "extra": "do not discard"},
        {"rationale": "no action", "target_frame_id": None},
    ):
        with pytest.raises(AdapterParseError):
            adapter.parse(TASK_SIGNATURES["consolidate"], json.dumps(invalid))


def test_explicit_case_normalization_preserves_text():
    adapter = _make_adapter("json", canonicalize=True)
    result = adapter.parse(
        TASK_SIGNATURES["event_claims"],
        json.dumps(
            {
                "CLAIMS": [
                    {
                        "text": "  Source QUOTE unchanged  ",
                        "kind": " PREFERENCE ",
                        "relations": [{"source": "A", "target": "B", "relation_TYPE": "WORKS_AT"}],
                    }
                ]
            }
        ),
    )
    assert result["claims"][0].text == "  Source QUOTE unchanged  "
    assert result["claims"][0].kind == "preference"
    assert result["claims"][0].relations[0].relation_type == "WORKS_AT"
