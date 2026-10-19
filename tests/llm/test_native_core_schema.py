"""Authored loopback SDK contracts, not real-model or decoder-quality evidence."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from dspy.lm15 import (
    AccessPolicy,
    ModelSupport,
    OpenAIChatCompat,
    ProviderDefinition,
    register_provider,
    unregister_provider,
)

from gluomem.config.app_config import AppConfig, DspyConfig, ModelEntry
from gluomem.llm.bootstrap import save_artifact, save_json_artifact
from gluomem.llm.modules import DspyExtractors, build_module
from gluomem.llm.outputs import ExtractionError, StoredCandidate


@pytest.fixture(params=[(False, "zero"), (True, "zero"), (True, "json"), (True, "program")])
def native_core(tmp_path, monkeypatch, request):
    requests = []
    state = {"rationale": "é" * 8192}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
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
            encoded = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    provider = "gluomem-authored-native-core"
    canonicalize, artifact = request.param
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
        if artifact != "zero":
            legacy = cfg.model_copy(deep=True)
            legacy.dspy.output_field_limits = {}
            module = build_module("consolidate", legacy)
            saver = save_json_artifact if artifact == "json" else save_artifact
            saver(module, cfg.dspy.artifacts.dir, "consolidate", model_ref="authored-native")
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


def test_core_factory_native_schema_reaches_sdk_wire(native_core):
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


def test_core_stop_overlimit_is_parser_rejection_not_decoder_proof(native_core):
    native_core["state"]["rationale"] = "é" * 8193
    with pytest.raises(ExtractionError, match="^structured extraction failed$"):
        consolidate(native_core)
    assert len(native_core["requests"]) == 1
