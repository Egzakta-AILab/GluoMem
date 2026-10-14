"""Tests for authenticated FastAPI service contract, auth enforcement, and scope isolation."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from gluomem.config.app_config import AppConfig
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.service.app import create_app
from gluomem.service.auth import (
    HEADER_NAMESPACE,
    HEADER_PRINCIPAL,
    HEADER_TENANT,
)
from gluomem.storage.db import MindDB
from gluomem.storage.scope import MemoryScope


def test_auth_and_header_scope_fail_closed(tmp_path: Path) -> None:
    cfg = AppConfig()
    db = MindDB(tmp_path / "api_auth.mind", vector_dim=32)
    embedder = MockEmbedder(dim=32)
    store = FrameStore(db, embedder=embedder)
    app = create_app(db, store, cfg=cfg, service_key="top-secret-key")
    client = TestClient(app)

    # 1. Liveness/readiness probes succeed without auth
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").json()["status"] == "ready"

    # 2. Missing Authorization header -> 401
    r_no_auth = client.post(
        "/v1/memories:recall",
        json={"query": "hello"},
        headers={HEADER_TENANT: "egzakta", HEADER_PRINCIPAL: "mikhail"},
    )
    assert r_no_auth.status_code == 401

    # 3. Wrong bearer token -> 403
    r_bad_auth = client.post(
        "/v1/memories:recall",
        json={"query": "hello"},
        headers={
            "Authorization": "Bearer wrong-key",
            HEADER_TENANT: "egzakta",
            HEADER_PRINCIPAL: "mikhail",
        },
    )
    assert r_bad_auth.status_code == 403

    # 4. Missing X-GluoMem-Tenant or X-GluoMem-Principal -> 422
    r_missing_scope = client.post(
        "/v1/memories:recall",
        json={"query": "hello"},
        headers={"Authorization": "Bearer top-secret-key"},
    )
    assert r_missing_scope.status_code == 422

    # 5. Request body attempting to smuggle tenant_id or principal_id -> 422
    r_body_scope = client.post(
        "/v1/events",
        json={
            "event_id": "evt-1",
            "source_ref": "conv-1",
            "content": "Attempted scope override",
            "principal_id": "victim",
        },
        headers={
            "Authorization": "Bearer top-secret-key",
            HEADER_TENANT: "egzakta",
            HEADER_PRINCIPAL: "mikhail",
            HEADER_NAMESPACE: "personal",
        },
    )
    assert r_body_scope.status_code == 422

    db.close()


def test_http_endpoints_isolate_scopes_end_to_end(tmp_path: Path) -> None:
    cfg = AppConfig()
    db = MindDB(tmp_path / "api_iso.mind", vector_dim=32)
    embedder = MockEmbedder(dim=32)
    store = FrameStore(db, embedder=embedder)
    app = create_app(db, store, cfg=cfg, service_key="svc-secret")
    client = TestClient(app)

    headers_a = {
        "Authorization": "Bearer svc-secret",
        HEADER_TENANT: "egzakta",
        HEADER_PRINCIPAL: "alice",
        HEADER_NAMESPACE: "personal",
    }
    headers_b = {
        "Authorization": "Bearer svc-secret",
        HEADER_TENANT: "egzakta",
        HEADER_PRINCIPAL: "bob",
        HEADER_NAMESPACE: "personal",
    }

    # Alice writes a memory
    resp_in = client.post(
        "/v1/events",
        json={
            "event_id": "evt-a1",
            "source_ref": "chat-a1",
            "content": "My SSH bastion port is 22022.",
        },
        headers=headers_a,
    )
    assert resp_in.status_code == 200
    alice_frame_id = resp_in.json()["frames"][0]["id"]

    # Bob cannot recall, view, correct, or delete Alice's memory
    bob_recall = client.post(
        "/v1/memories:recall",
        json={"query": "SSH bastion port 22022"},
        headers=headers_b,
    )
    assert bob_recall.status_code == 200
    assert bob_recall.json()["hits"] == []

    assert (
        client.delete(
            f"/v1/memories/{alice_frame_id}", headers=headers_b
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/v1/memories/{alice_frame_id}:correct",
            json={"new_content": "Hacked port 22."},
            headers=headers_b,
        ).status_code
        == 404
    )

    # Alice's memory is still intact for Alice
    alice_recall = client.post(
        "/v1/memories:recall",
        json={"query": "SSH bastion port 22022"},
        headers=headers_a,
    )
    assert len(alice_recall.json()["hits"]) == 1
    assert (
        alice_recall.json()["hits"][0]["frame"]["scope"]
        == MemoryScope("egzakta", "alice", "personal").as_dict()
    )

    db.close()
