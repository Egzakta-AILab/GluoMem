from types import SimpleNamespace
from typing import Any, cast

import pytest

from gluomem.config.app_config import AppConfig, DspyConfig
from gluomem.llm.router import LMRouter
from gluomem.llm.signatures import TASK_SIGNATURES


def test_explicit_strict_wire_policy_does_not_close_semantic_mappings():
    cfg = AppConfig(dspy=DspyConfig(adapter="json", schema_mode="required", schema_strict=True))
    adapter = LMRouter(cfg)._adapter
    options = {}
    cast(Any, adapter)._prepare_response_format(
        SimpleNamespace(supported_params={"response_format"}, supports_response_schema=True),
        options,
        TASK_SIGNATURES["event_claims"],
    )
    assert options["response_format"]["json_schema"]["strict"] is True
    props = options["response_format"]["json_schema"]["schema"]["$defs"]["KGEntity"]["properties"]
    assert props["properties"]["additionalProperties"] == {"type": "string"}


def test_strict_without_required_schema_refused():
    with pytest.raises(ValueError):
        DspyConfig(schema_strict=True)
