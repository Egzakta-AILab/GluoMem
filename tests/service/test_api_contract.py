"""Authentication, validation, isolation, and request transaction contracts."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gluomem.config.app_config import AppConfig
from gluomem.frames.store import FrameStore
from gluomem.service import app as app_module
from gluomem.service.app import create_app
from gluomem.storage.db import MindDB

HEADERS = {
    "Authorization": "Bearer secret",
    "X-GluoMem-Tenant": "egzakta",
    "X-GluoMem-Principal": "alice",
}


@pytest.fixture
def client(mind_db: MindDB, frame_store: FrameStore, cfg: AppConfig):
    with TestClient(create_app(mind_db, frame_store, cfg=cfg, service_key="secret")) as client:
        yield client


@pytest.mark.parametrize(
    ("headers", "body", "status"),
    [
        ({"X-GluoMem-Tenant": "egzakta", "X-GluoMem-Principal": "alice"}, {}, 401),
        ({**HEADERS, "Authorization": "Bearer wrong"}, {}, 401),
        ({"Authorization": "Bearer secret"}, {}, 422),
        (HEADERS, {"principal_id": "victim"}, 422),
        (HEADERS, {}, 200),
    ],
)
def test_auth_and_header_scope_fail_closed(
    client: TestClient, headers: dict[str, str], body: dict[str, Any], status: int
) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ok"}
    response = client.post("/v1/memories:recall", json={"query": "hello", **body}, headers=headers)
    assert response.status_code == status
    if status == 401:
        assert response.headers["www-authenticate"] == "Bearer"
    assert client.get("/v1/status").status_code == 401


@pytest.mark.parametrize(
    ("path", "body", "headers"),
    [
        ("/v1/memories:recall", {"query": "hello", "profile": "unknown"}, HEADERS),
        ("/v1/memories:recall", {"query": "q" * 4097}, HEADERS),
        (
            "/v1/events",
            {"event_id": "bad", "source_ref": "doc", "event_time": "not-a-date"},
            HEADERS,
        ),
        ("/v1/events", {"event_id": "big", "source_ref": "doc", "content": "c" * 100001}, HEADERS),
        (
            "/v1/events",
            {
                "event_id": "msg",
                "source_ref": "doc",
                "messages": [{"message_id": "m", "content": "c" * 100001}],
            },
            HEADERS,
        ),
        ("/v1/memories:recall", {"query": "hi"}, {**HEADERS, "X-GluoMem-Principal": "x" * 129}),
        (
            "/v1/memories:recall",
            {"query": "hi"},
            {**HEADERS, "X-GluoMem-Principal": "bad principal"},
        ),
    ],
)
def test_invalid_input_is_422(
    client: TestClient, path: str, body: dict[str, Any], headers: dict[str, str]
) -> None:
    assert client.post(path, json=body, headers=headers).status_code == 422


def test_scope_isolation_and_pagination(client: TestClient) -> None:
    event = {"event_id": "a1", "source_ref": "doc", "content": "SSH bastion port is 22022."}
    frame_id = client.post("/v1/events", json=event, headers=HEADERS).json()["frames"][0]["id"]
    event.update(event_id="a2", content="Our proxy port is 21000.")
    client.post("/v1/events", json=event, headers=HEADERS)
    bob = {**HEADERS, "X-GluoMem-Principal": "bob"}
    assert (
        client.post("/v1/memories:recall", json={"query": "SSH bastion"}, headers=bob).json()[
            "hits"
        ]
        == []
    )
    missing = client.delete(f"/v1/memories/{frame_id}", headers=bob)
    absent = client.delete("/v1/memories/9999", headers=HEADERS)
    assert missing.status_code == absent.status_code == 404
    assert missing.json() == absent.json() == {"detail": "Memory resource not found"}
    page = client.get(
        "/v1/memories", params={"limit": 1, "source_ref": "doc", "kind": "fact"}, headers=HEADERS
    ).json()
    assert page["by_source_ref"] == {"doc": [frame_id]}
    assert page["next_cursor"] == frame_id
    following = client.get("/v1/memories", params={"cursor": frame_id}, headers=HEADERS).json()
    assert len(following["active_frames"]) == 1
    assert following["next_cursor"] is None


@pytest.mark.parametrize("async_mode", [True, False])
def test_payload_conflict(client: TestClient, async_mode: bool) -> None:
    body = {"event_id": "event", "source_ref": "doc", "content": "one", "async_mode": async_mode}
    first = client.post("/v1/events", json=body, headers=HEADERS)
    assert first.status_code == (202 if async_mode else 200)
    duplicate = client.post("/v1/events", json=body, headers=HEADERS)
    key = "id" if async_mode else "source_event_row_id"
    assert duplicate.json()[key] == first.json()[key]
    body["content"] = "two"
    assert client.post("/v1/events", json=body, headers=HEADERS).status_code == 409


def test_failed_request_cannot_rollback_concurrent_write(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, written = Event(), Event()
    ingest = app_module.ingest_source_event

    def interleave(event: Any, *, store: FrameStore, **kwargs: Any):
        if event.event_id == "fails":
            with store.db.transaction():
                entered.set()
                assert written.wait(5)
                raise ValueError("request rejected")
        assert entered.wait(5)
        try:
            return ingest(event, store=store, **kwargs)
        finally:
            written.set()

    monkeypatch.setattr(app_module, "ingest_source_event", interleave)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                client.post,
                "/v1/events",
                headers=HEADERS,
                json={"event_id": name, "source_ref": name, "content": "Persist this decision."},
            )
            for name in ("fails", "succeeds")
        ]
        assert [future.result().status_code for future in futures] == [422, 200]
    frames = client.get("/v1/memories", headers=HEADERS).json()["active_frames"]
    assert len(frames) == 1
    assert frames[0]["provenance"][0]["source_ref"] == "succeeds"
