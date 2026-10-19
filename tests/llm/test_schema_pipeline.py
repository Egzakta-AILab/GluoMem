import asyncio

import pytest

from gluomem.llm.router import _make_adapter
from gluomem.llm.signatures import TASK_SIGNATURES

from .test_feedback_boundaries import Scripted


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("name", ["json", "baml"])
def test_required_schema_reaches_authored_lm_call(name, asynchronous):
    class Declared(Scripted):
        @property
        def supported_params(self):
            return {"response_format"}

        @property
        def supports_response_schema(self):
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
    assert wire["json_schema"]["schema"]["$defs"]["ExtractedClaimSpec"]["properties"]["kind"][
        "enum"
    ]
