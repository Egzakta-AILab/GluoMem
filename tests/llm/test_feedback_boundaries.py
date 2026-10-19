import asyncio
from typing import Any, cast

import dspy
import pytest
from dspy.utils.dummies import DummyLM
from dspy.utils.exceptions import AdapterParseError

from gluomem.llm.router import _make_adapter
from gluomem.llm.signatures import TASK_SIGNATURES


class Scripted(DummyLM):
    def __init__(self, responses):
        super().__init__([])
        self.responses = iter(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        value = next(self.responses)
        if isinstance(value, BaseException):
            raise value
        return [value]

    async def acall(self, prompt=None, *, messages=None, **kwargs):
        return self(prompt=prompt, messages=messages, **kwargs)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("safe"),
        OSError("safe"),
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


def test_declared_public_provider_capabilities_without_dispatch():
    from dspy.lm15 import (
        AccessPolicy,
        ModelSupport,
        OpenAIChatCompat,
        ProviderDefinition,
        register_provider,
        unregister_provider,
    )

    name = "gluomem-offline-schema-test"
    try:
        register_provider(
            ProviderDefinition.chat(
                AccessPolicy(provider=name, env_keys=(), base_url="https://example.invalid/v1"),
                compat=OpenAIChatCompat(),
            ),
            supports=ModelSupport(response_schema=True, function_calling=False),
        )
        lm = dspy.LM(
            name + "/opaque-model-token",
            engine="lm15",
            num_retries=0,
            api_key="offline-placeholder",
        )
        assert lm.supports_response_schema and "response_format" in lm.supported_params
        options = {}
        cast(Any, _make_adapter("json", schema_mode="required"))._prepare_response_format(
            lm, options, TASK_SIGNATURES["event_claims"]
        )
        assert options["response_format"]["type"] == "json_schema"
    finally:
        unregister_provider(name)
