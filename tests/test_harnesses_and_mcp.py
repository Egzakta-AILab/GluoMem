"""Tests for Universal Agentic Harness Layer: MCP server, hooks, and MemoryToolPack."""

from __future__ import annotations

import io
import json
from pathlib import Path

from fastapi.testclient import TestClient

from gluomem.cli import main
from gluomem.client.sdk import EmbeddedMemoryClient
from gluomem.config.app_config import load_config
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.harnesses import MemoryToolPack, run_hook_cli
from gluomem.mcp import McpServer, run_stdio_server
from gluomem.service.app import create_app
from gluomem.storage.db import MindDB
from gluomem.storage.scope import MemoryScope


def _test_cfg():
    return load_config().with_overrides(
        {
            "components.allow_fake": True,
            "embedding.model": "mock",
            "models.mock.dim": 16,
            "storage.vector_dim": 16,
            "write.extraction_backend": "fake",
        }
    )


def test_memory_toolpack_schemas_and_lifecycle(tmp_path: Path) -> None:
    cfg = _test_cfg()
    scope = MemoryScope("egzakta", "alice", "coding")
    with EmbeddedMemoryClient.open(
        tmp_path / "tools.mind",
        cfg=cfg,
        default_scope=scope,
        embedder=MockEmbedder(dim=16),
    ) as client:
        pack = MemoryToolPack(client, default_scope=scope)
        assert len(pack.openai_tools()) == 6
        assert len(pack.anthropic_tools()) == 6
        assert len(pack.mcp_tools()) == 6
        assert len(pack.as_callables()) == 6

        remembered = pack.dispatch(
            "memory_remember",
            json.dumps(
                {
                    "content": "Always run pytest with -q before opening a PR.",
                    "kind": "procedure",
                    "source_ref": "cursor:rule",
                }
            ),
        )
        frame_id = remembered["frames"][0]["id"]
        assert remembered["frames"][0]["kind"] == "procedure"

        recalled = pack.dispatch("memory_recall", {"query": "pytest before PR", "k": 5})
        assert not any(h["frame"]["id"] == frame_id for h in recalled["hits"])
        assert remembered["frames"][0]["status"] == "candidate"
        assert remembered["frames"][0]["verified_by_user"] is False
        client.review_memory(frame_id, "accept", scope=scope)
        recalled = pack.dispatch("memory_recall", {"query": "pytest before PR", "k": 5})
        assert any(h["frame"]["id"] == frame_id for h in recalled["hits"])

        proposed = pack.dispatch(
            "memory_correct",
            {
                "frame_id": frame_id,
                "new_content": "Always run uv run pytest -q before opening a PR.",
            },
        )
        assert proposed["requires_review"] is True
        assert proposed["frames"][0]["status"] == "candidate"
        # Authenticated host confirmation, not an autonomous tool invocation.
        corrected = client.correct_memory(
            frame_id, "Always run uv run pytest -q before opening a PR.", scope=scope
        ).model_dump(mode="json")
        new_id = corrected["replacement_frame"]["id"]
        assert new_id != frame_id

        inspected = pack.dispatch("memory_inspect_provenance", {"frame_id": new_id})
        assert [f["id"] for f in inspected["lineage"]] == [frame_id, new_id]

        exported = pack.dispatch("memory_export", {})
        assert any(m["id"] == new_id for m in exported["active_memories"])

        forgotten = pack.dispatch("memory_forget", {"frame_id": new_id})
        assert forgotten["frame_id"] == new_id


def test_mcp_stdio_and_http_jsonrpc(tmp_path: Path) -> None:
    cfg = _test_cfg()
    scope = MemoryScope("egzakta", "bob", "agent")
    db = MindDB(tmp_path / "mcp.mind", dim=16)
    store = FrameStore(db, embedder=MockEmbedder(dim=16), scope=scope)
    client = EmbeddedMemoryClient(db, store, cfg=cfg, default_scope=scope)
    server = McpServer(client, default_scope=scope)

    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "memory_remember",
                "arguments": {
                    "content": "Deployment target is eu-central-1.",
                    "kind": "fact",
                },
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "memory_recall",
                "arguments": {"query": "Deployment target eu-central-1", "k": 3},
            },
        },
    ]
    instream = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n")
    outstream = io.StringIO()
    assert run_stdio_server(server, instream=instream, outstream=outstream) == 0

    replies = [json.loads(line) for line in outstream.getvalue().splitlines() if line.strip()]
    assert [r["id"] for r in replies] == [1, 2, 3, 4]
    assert len(replies[1]["result"]["tools"]) == 6
    assert replies[2]["result"]["isError"] is False
    assert replies[2]["result"]["structuredContent"]["frames"][0]["status"] == "candidate"
    assert replies[3]["result"]["structuredContent"]["hits"] == []

    # Also verify HTTP /mcp mount on FastAPI service
    app = create_app(db, store, cfg=cfg, service_key="test-secret-key")
    with TestClient(app) as http:
        headers = {
            "Authorization": "Bearer test-secret-key",
            "X-GluoMem-Tenant": "egzakta",
            "X-GluoMem-Principal": "bob",
            "X-GluoMem-Namespace": "agent",
        }
        resp = http.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 99, "method": "tools/list"},
        )
        assert resp.status_code == 200
        assert len(resp.json()["result"]["tools"]) == 6
    db.close()


def test_hook_recall_and_capture_strips_thinking_and_is_idempotent(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    cfg = _test_cfg()
    db_path = tmp_path / "hooks.mind"
    scope = MemoryScope("egzakta", "dev", "cursor")
    with EmbeddedMemoryClient.open(
        db_path, cfg=cfg, default_scope=scope, embedder=MockEmbedder(dim=16)
    ) as client:
        capture_payload = {
            "session_id": "sess-42",
            "turn_id": "turn-1",
            "messages": [
                {
                    "message_id": "u1",
                    "role": "user",
                    "content": (
                        "<thinking>internal scratchpad</thinking>Our staging API port is 8443."
                    ),
                },
                {
                    "message_id": "a1",
                    "role": "assistant",
                    "content": "Noted that staging API port is 8443.",
                },
            ],
        }
        out1 = json.loads(
            run_hook_cli("capture", json.dumps(capture_payload), client=client, default_scope=scope)
        )
        assert out1["ingested"] is True
        assert out1["deduped_event"] is False
        assert len(out1["frame_ids"]) == 1

        # Replaying the exact same hook payload is idempotent
        out2 = json.loads(
            run_hook_cli("capture", json.dumps(capture_payload), client=client, default_scope=scope)
        )
        assert out2["deduped_event"] is True

        recall_payload = {"prompt": "What is our staging API port?"}
        recall_out = json.loads(
            run_hook_cli("recall", json.dumps(recall_payload), client=client, default_scope=scope)
        )
        assert recall_out["continue"] is True
        assert "<memory_context>" in recall_out["additionalContext"]
        assert "8443" in recall_out["additionalContext"]
        assert "internal scratchpad" not in recall_out["additionalContext"]

    # Also test CLI entrypoint `gluomem hook recall`
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"prompt": "staging API port 8443"})))
    rc = main(
        [
            "hook",
            "recall",
            "--db",
            str(db_path),
            "--tenant-id",
            "egzakta",
            "--principal-id",
            "dev",
            "--namespace",
            "cursor",
            "--override",
            "components.allow_fake=true",
            "--override",
            "embedding.model=mock",
            "--override",
            "models.mock.dim=16",
            "--override",
            "storage.vector_dim=16",
        ]
    )
    assert rc == 0
    cli_stdout = json.loads(capsys.readouterr().out)
    assert "8443" in cli_stdout["additionalContext"]
