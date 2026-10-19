import json

import dspy

from gluomem.config.app_config import AppConfig, DspyConfig
from gluomem.llm.modules import DspyExtractors
from gluomem.llm.router import LMRouter

from .test_feedback_boundaries import Scripted


def test_real_extractor_compiled_program_boundary_with_authored_transport():
    cfg = AppConfig(
        dspy=DspyConfig(adapter="json", canonicalize=True, validation_retries=1, track_usage=False)
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
