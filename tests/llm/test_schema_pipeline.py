import asyncio
import json

import dspy
import pytest

from gluomem.config.app_config import AppConfig, DspyConfig
from gluomem.llm.modules import DspyExtractors
from gluomem.llm.router import LMRouter, _make_adapter
from gluomem.llm.signatures import TASK_SIGNATURES

from ._helpers import Scripted


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("name", ["json", "baml"])
def test_required_schema_reaches_authored_lm_call(name, asynchronous):
    class Declared(Scripted):
        @property
        def supported_params(self):
            return {"response_format"}

        @property
        def supports_response_schema(self) -> bool:
            return True

    lm = Declared(['{"claims":[]}'])
    adapter = _make_adapter(name, canonicalize=True, schema_mode="required", schema_strict=True)
    args = (
        lm,
        {"temperature": 0, "max_tokens": 42},
        TASK_SIGNATURES["event_claims"],
        [],
        {"source_text": "authored", "reference_date": ""},
    )
    result = asyncio.run(adapter.acall(*args)) if asynchronous else adapter(*args)
    assert result == [{"claims": []}]
    assert len(lm.calls) == 1
    wire = lm.calls[0]["response_format"]
    assert wire["type"] == "json_schema" and wire["json_schema"]["strict"] is True
    definitions = wire["json_schema"]["schema"]["$defs"]
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

    class Unsupported(Declared):
        @property
        def supports_response_schema(self) -> bool:
            return False

    unsupported = Unsupported(['{"claims":[]}'])
    rejected_args = (unsupported, *args[1:])
    with pytest.raises(ValueError, match="capability"):
        asyncio.run(adapter.acall(*rejected_args)) if asynchronous else adapter(*rejected_args)
    assert unsupported.calls == []


def test_strict_without_required_schema_refused():
    with pytest.raises(ValueError):
        DspyConfig(model="mock", schema_strict=True)


def test_real_extractor_compiled_program_boundary_with_authored_transport():
    cfg = AppConfig(
        dspy=DspyConfig(
            model="mock", adapter="json", canonicalize=True, validation_retries=1, track_usage=False
        )
    )
    claim = {
        "text": "Exact source text",
        "kind": "FACT",
        "verbatim_span": "Exact source text",
        "message_id": "opaque-1",
        "event_date": "2024-02-29T00:00:00+00:00",
    }
    lm = Scripted(['{"claims":"invalid"}', json.dumps({"claims": [claim]})])

    class Router(LMRouter):
        def bind(self, task):
            return dspy.context(lm=lm, adapter=self._adapter, track_usage=False)

    extractor = DspyExtractors(cfg, router=Router(cfg))
    result = extractor.extract_event_claims("[opaque-1] Exact source text")
    assert len(lm.calls) == 2 and len(result) == 1
    assert result[0].verbatim_span == "Exact source text"
    assert result[0].message_id == "opaque-1"
    assert result[0].event_date is not None
    assert result[0].event_date.isoformat() == "2024-02-29T00:00:00+00:00"
    assert result[0].kind == "fact"
    assert "schema_validation" not in extractor._modules["event_claims"].signature.instructions
