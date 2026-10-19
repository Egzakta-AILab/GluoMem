"""Authored loopback SDK contracts, not real-model or decoder-quality evidence."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from dspy.lm15 import (
    AccessPolicy,
    ModelSupport,
    OpenAIChatCompat,
    ProviderDefinition,
    register_provider,
    unregister_provider,
)

from gluomem.config.app_config import AppConfig, DspyConfig, DspyRetryConfig, ModelEntry
from gluomem.llm.modules import DspyExtractors
from gluomem.llm.outputs import ExtractionError, StoredCandidate


@pytest.fixture(params=[False, True], ids=["canonicalize-off", "canonicalize-on"])
def native_core(tmp_path, monkeypatch, request):
    requests = []
    state: dict[str, Any] = {"rationale": "é" * 8192, "remaining_http_failures": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def reply(self, status, response):
            encoded = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            if state["remaining_http_failures"]:
                state["remaining_http_failures"] -= 1
                self.reply(503, {"error": {"message": "PRIVATE_TRANSPORT_MARKER"}})
                return
            response = {
                "id": "authored-fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "decision": {
                                        "action": "ADD",
                                        "target_frame_id": None,
                                        "rationale": state["rationale"],
                                    }
                                }
                            ),
                        },
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
            }
            self.reply(200, response)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    provider = "gluomem-authored-native-core"
    canonicalize = request.param
    try:
        register_provider(
            ProviderDefinition.chat(
                AccessPolicy(
                    provider=provider,
                    env_keys=(),
                    base_url=f"http://127.0.0.1:{server.server_port}/v1",
                ),
                compat=OpenAIChatCompat(),
            ),
            supports=ModelSupport(response_schema=True, function_calling=False),
        )
        monkeypatch.setenv("GLUOMEM_TEST_NATIVE_KEY", "authored-placeholder")
        cfg = AppConfig(
            dspy=DspyConfig(
                model="mock",
                adapter="json",
                schema_mode="required",
                schema_strict=True,
                canonicalize=canonicalize,
                output_field_limits={"consolidate": {"decision.rationale": 8192}},
                artifacts={"dir": str(tmp_path / "artifacts")},
            )
        )
        cfg.models["authored-native"] = ModelEntry(
            provider=provider,
            provider_model=f"{provider}/fixture-model",
            api_key_env="GLUOMEM_TEST_NATIVE_KEY",
            extra={"engine": "lm15", "num_retries": 0, "timeout": 5},
        )
        cfg.dspy.tasks["consolidate"].model = "authored-native"

        extractors = DspyExtractors(cfg)
        yield {"extractors": extractors, "cfg": cfg, "requests": requests, "state": state}
    finally:
        unregister_provider(provider)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def consolidate(native_core):
    return native_core["extractors"].consolidate_claim(
        "Rhea owns a telescope.",
        [StoredCandidate(frame_id=1, content="Rhea owns a camera.")],
    )


def test_core_native_wire_schema_and_parser_boundary(native_core):
    decision = consolidate(native_core)
    assert decision.rationale == native_core["state"]["rationale"]
    assert decision.action == "ADD" and decision.target_frame_id is None
    path, body = native_core["requests"][-1]
    assert path == "/v1/chat/completions"
    wire = body["response_format"]["json_schema"]
    assert wire["strict"] is True
    assert (
        wire["schema"]["$defs"]["ConsolidationDecision"]["properties"]["rationale"]["maxLength"]
        == 8192
    )
    assert "maxLength" in json.dumps(body["messages"])
    assert len(native_core["requests"]) == 1
    # A stop response is not decoder enforcement: reject over-limit output locally.
    native_core["state"]["rationale"] = "é" * 8193
    with pytest.raises(ExtractionError, match="^structured extraction failed$"):
        consolidate(native_core)
    assert len(native_core["requests"]) == 2


def test_core_retries_native_http_failure_and_logs_safe_json(native_core, caplog):
    from gluomem.logging import JsonLogFormatter

    cfg = native_core["cfg"].model_copy(deep=True)
    cfg.dspy.retry = DspyRetryConfig(max_attempts=3, initial_delay_seconds=0, max_delay_seconds=0)
    native_core["state"]["remaining_http_failures"] = 1
    native_core["extractors"] = DspyExtractors(cfg)
    with caplog.at_level("INFO", logger="gluomem.llm.retry"):
        decision = consolidate(native_core)
    assert decision.action == "ADD"
    assert len(native_core["requests"]) == 2
    assert native_core["requests"][0] == native_core["requests"][1]
    records = [r for r in caplog.records if r.name == "gluomem.llm.retry"]
    assert [r.outcome for r in records] == ["retrying", "succeeded"]
    assert [r.http_status for r in records] == [503, None]
    encoded = [JsonLogFormatter().format(record) for record in records]
    assert all(json.loads(line)["event"] == "llm_attempt" for line in encoded)
    assert all(
        "PRIVATE_TRANSPORT_MARKER" not in line and "telescope" not in line for line in encoded
    )
