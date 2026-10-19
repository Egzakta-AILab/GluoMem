import asyncio
import json
from typing import Any, cast

import pytest
from dspy.utils.exceptions import AdapterParseError

from gluomem.llm.router import _make_adapter
from gluomem.llm.signatures import TASK_SIGNATURES

from ._helpers import Scripted


@pytest.mark.parametrize("asynchronous", [False, True])
def test_validation_feedback_one_same_lm_attempt_at_zero(asynchronous):
    adapter = _make_adapter("json", canonicalize=True, validation_retries=1)
    lm = Scripted(['{"claims":[{"text":"x","kind":"request"}]}', '{"claims":[]}'])
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


def test_config_exposes_opt_in_adapter_policy():
    from gluomem.config.app_config import DspyConfig
    from gluomem.llm.router import LMRouter

    config = DspyConfig(
        model="mock",
        adapter="baml",
        canonicalize=True,
        schema_mode="required",
        validation_retries=1,
    )
    assert (
        config.canonicalize and config.schema_mode == "required" and config.validation_retries == 1
    )
    with pytest.raises(ValueError):
        DspyConfig.model_validate({"model": "mock", "validation_retries": 2})
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


@pytest.mark.parametrize("wrapper", ["```json\n{}\n```", "<decision>\n{}\n</decision>"])
def test_exact_structural_delimiters_only(wrapper):
    adapter = _make_adapter("json", canonicalize=True)
    body = json.dumps({"action": "ADD", "target_frame_id": None, "rationale": "Exact quote"})
    assert (
        adapter.parse(TASK_SIGNATURES["consolidate"], wrapper.format(body))["decision"].rationale
        == "Exact quote"
    )
    for extra in [
        "private outside " + wrapper.format(body),
        wrapper.format(body) + " trailing",
        "<unknown>" + body + "</unknown>",
    ]:
        with pytest.raises(AdapterParseError):
            adapter.parse(TASK_SIGNATURES["consolidate"], extra)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("safe"),
        RuntimeError("length/context/privacy/resource"),
        asyncio.CancelledError(),
    ],
)
def test_nonvalidation_failure_never_retried(asynchronous, error):
    lm = Scripted([error, '{"claims":[]}'])
    adapter = _make_adapter("json", canonicalize=True, validation_retries=1)
    args = (
        lm,
        {},
        TASK_SIGNATURES["event_claims"],
        [],
        {"source_text": "safe", "reference_date": ""},
    )
    with pytest.raises(type(error)) as caught:
        asyncio.run(adapter.acall(*args)) if asynchronous else adapter(*args)
    assert caught.value is error and len(lm.calls) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_second_validation_failure_preserves_original_cause_no_third_call(asynchronous):
    lm = Scripted(['{"claims":"not-list"}', '{"claims":"not-list"}', '{"claims":[]}'])
    adapter = _make_adapter("json", canonicalize=True, validation_retries=1)
    args = (
        lm,
        {},
        TASK_SIGNATURES["event_claims"],
        [],
        {"source_text": "safe", "reference_date": ""},
    )
    with pytest.raises(AdapterParseError) as caught:
        asyncio.run(adapter.acall(*args)) if asynchronous else adapter(*args)
    assert len(lm.calls) == 2
    assert isinstance(caught.value.__cause__, AdapterParseError)
    assert caught.value.__cause__.__cause__ is not None


def test_default_parser_and_no_retry_are_legacy_compatible():
    import dspy

    assert type(_make_adapter("json")) is dspy.JSONAdapter
    assert type(_make_adapter("chat")) is dspy.ChatAdapter
    lm = Scripted(['{"claims":"not-list"}', '{"claims":[]}'])
    with pytest.raises(AdapterParseError):
        _make_adapter("json", canonicalize=True)(
            lm,
            {},
            TASK_SIGNATURES["event_claims"],
            [],
            {"source_text": "safe", "reference_date": ""},
        )
    assert len(lm.calls) == 1
