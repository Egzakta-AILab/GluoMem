"""Synthetic database regressions for host/agent security boundaries."""

import inspect

import pytest
from fastapi.testclient import TestClient

from gluomem.client.sdk import EmbeddedMemoryClient
from gluomem.embedding.providers import MockEmbedder
from gluomem.harnesses.hooks import handle_capture_hook, resolve_hook_scope
from gluomem.harnesses.tools import MemoryToolPack
from gluomem.mcp import McpServer
from gluomem.service.app import create_app
from gluomem.storage.scope import MemoryScope
from tests.test_harnesses_and_mcp import _test_cfg


@pytest.fixture
def boundary_client(tmp_path):
    with EmbeddedMemoryClient.open(
        tmp_path / "gate-a.mind",
        cfg=_test_cfg(),
        embedder=MockEmbedder(dim=16),
        default_scope=BOB,
    ) as client:
        yield client


ALICE = MemoryScope("test", "alice", "private")
BOB = MemoryScope("test", "bob", "private")


@pytest.mark.parametrize("transport", ["pack", "mcp", "http"])
def test_agent_scope_cannot_override_host(boundary_client, transport):
    client = boundary_client
    victim = client.ingest_event(
        event_id="bob", source_ref="host", content="Bob secret launch code.", scope=BOB
    ).frames[0]
    args = {"scope": BOB.as_dict()}
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "memory_export", "arguments": args},
    }
    if transport == "pack":
        result = MemoryToolPack(client, default_scope=ALICE).dispatch("memory_export", args)
    elif transport == "mcp":
        reply = McpServer(client, default_scope=BOB).handle_message(request, scope_override=ALICE)
        assert reply is not None
        result = reply["result"]["structuredContent"]
    else:
        app = create_app(client.db, client.store, cfg=_test_cfg(), service_key="test")
        with TestClient(app) as http:
            result = http.post(
                "/mcp",
                json=request,
                headers={
                    "Authorization": "Bearer test",
                    "X-GluoMem-Tenant": ALICE.tenant_id,
                    "X-GluoMem-Principal": ALICE.principal_id,
                    "X-GluoMem-Namespace": ALICE.namespace,
                },
            ).json()["result"]["structuredContent"]
    assert result["scope"] == ALICE.as_dict()
    assert victim.id not in [f["id"] for f in result["active_memories"]]
    assert client.get_lineage(victim.id, scope=BOB)[0].content == "Bob secret launch code."


def test_agent_schema_and_unpinned_dispatch_do_not_accept_identity(boundary_client):
    pack = MemoryToolPack(boundary_client)
    for spec in pack.mcp_tools():
        assert "scope" not in spec["inputSchema"]["properties"]
        if spec["name"] == "memory_remember":
            assert not {"pinned", "require_review"} & spec["inputSchema"]["properties"].keys()
    result = pack.dispatch("memory_export", {"scope": ALICE.as_dict()})
    assert result["scope"] == BOB.as_dict()


def test_agent_remember_cannot_claim_user_verification(boundary_client):
    pack = MemoryToolPack(boundary_client, default_scope=BOB)
    result = pack.dispatch(
        "memory_remember",
        {
            "content": "Agent proposes using port 8443.",
            "require_review": False,
            "pinned": True,
            "source_type": "explicit_memory",
            "verified_by_user": True,
        },
    )
    frame = boundary_client.get_lineage(result["frames"][0]["id"], scope=BOB)[0]
    assert frame.status == "candidate"
    assert frame.trust != "trusted"
    for tool in MemoryToolPack(boundary_client).as_callables():
        assert "scope" not in inspect.signature(tool).parameters
    assert frame.verified_by_user is False
    assert frame.pinned is False
    assert result["frames"][0]["provenance"][0]["source_type"] == "tool_output"


def test_agent_correction_is_a_proposal_not_a_user_receipt(boundary_client):
    original = boundary_client.ingest_event(
        event_id="user",
        source_ref="host:user",
        source_type="explicit_memory",
        content="Use port 443.",
        kind="fact",
        scope=BOB,
    ).frames[0]
    result = MemoryToolPack(boundary_client, default_scope=BOB).memory_correct(
        original.id, "Use port 8443."
    )
    frames = boundary_client.store.all_frames(scope=BOB)
    proposed = next(f for f in frames if f.content == "Use port 8443.")
    assert proposed.status == "candidate"
    assert proposed.verified_by_user is False
    assert proposed.trust != "trusted"
    assert boundary_client.store.get_frame(original.id, scope=BOB).content == "Use port 443."
    assert result["requires_review"] is True


@pytest.mark.parametrize("transport", ["embedded", "http"])
def test_export_lineage_uses_requested_scope_not_store_default(boundary_client, transport):
    client = boundary_client
    original = client.ingest_event(
        event_id="alice", source_ref="host", content="Alice uses port 443.", scope=ALICE
    ).frames[0]
    replacement = client.correct_memory(original.id, "Alice uses port 8443.", scope=ALICE)
    target = replacement.replacement_frame.id
    if transport == "embedded":
        result = client.export_memories(scope=ALICE)
    else:
        app = create_app(client.db, client.store, cfg=_test_cfg(), service_key="test")
        with TestClient(app) as http:
            response = http.post(
                "/v1/memories:export",
                headers={
                    "Authorization": "Bearer test",
                    "X-GluoMem-Tenant": ALICE.tenant_id,
                    "X-GluoMem-Principal": ALICE.principal_id,
                    "X-GluoMem-Namespace": ALICE.namespace,
                },
            )
            assert response.status_code == 200, response.text
            result = response.json()
    exported = next(f for f in result["active_memories"] if f["id"] == target)
    assert exported["lineage"][0]["source_frame_id"] == original.id


def test_hook_payload_cannot_override_host_identity_or_mint_user_receipt(boundary_client):
    payload = {
        "scope": BOB.as_dict(),
        "source_type": "explicit_memory",
        "content": "Captured proposal from untrusted hook.",
        "event_id": "capture",
    }
    assert resolve_hook_scope(payload, default_scope=ALICE) == ALICE
    result = handle_capture_hook(boundary_client, payload, default_scope=ALICE)
    frame = boundary_client.store.get_frame(result["frame_ids"][0], scope=ALICE)
    assert frame.verified_by_user is False
    assert frame.trust != "trusted"
    for tool in MemoryToolPack(boundary_client).as_callables():
        assert "scope" not in inspect.signature(tool).parameters
