"""Both SDK transports exercise the same complete HTTP lifecycle."""

import asyncio
import inspect
import subprocess
import sys
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from gluomem.client.sdk import (
    AsyncGluoMemClient,
    EmbeddedMemoryClient,
    GluoMemApiError,
    GluoMemClient,
)
from gluomem.config.app_config import AppConfig
from gluomem.frames.store import FrameStore
from gluomem.service.app import create_app
from gluomem.service.worker import MemoryWorker
from gluomem.storage.db import MindDB


class _SyncASGITransport(httpx.BaseTransport):
    def __init__(self, app: Any) -> None:
        self.transport = httpx.ASGITransport(app=app)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        async def send():
            response = await self.transport.handle_async_request(request)
            content = await response.aread()
            await response.aclose()
            return httpx.Response(response.status_code, headers=response.headers, content=content)

        return asyncio.run(send())


class _Calls:
    def __init__(self, client: Any) -> None:
        self.client = client

    def __getattr__(self, name: str):
        def call(*args: Any, **kwargs: Any):
            result = getattr(self.client, name)(*args, **kwargs)
            if inspect.isawaitable(result):

                async def await_result():
                    return await result

                return asyncio.run(await_result())
            return result

        return call


@pytest.fixture(params=["sync", "async", "embedded"])
def sdk(request: pytest.FixtureRequest, mind_db: MindDB, frame_store: FrameStore, cfg: AppConfig):
    if request.param == "embedded":
        with EmbeddedMemoryClient(
            mind_db, frame_store, cfg=cfg, default_scope=frame_store.scope
        ) as embedded:
            yield _Calls(embedded)
        return
    app = create_app(mind_db, frame_store, cfg=cfg, service_key="secret")
    with TestClient(app):
        if request.param == "sync":
            client = GluoMemClient(
                service_key="secret",
                default_scope=frame_store.scope,
                transport=_SyncASGITransport(app),
            )
        else:
            client = AsyncGluoMemClient(
                service_key="secret",
                default_scope=frame_store.scope,
                transport=httpx.ASGITransport(app=app),
            )
        try:
            yield _Calls(client)
        finally:
            if isinstance(client, GluoMemClient):
                client.close()
            else:
                asyncio.run(client.aclose())


def test_sdk_full_lifecycle(
    sdk: Any, mind_db: MindDB, frame_store: FrameStore, cfg: AppConfig
) -> None:
    assert sdk.healthz() == sdk.readyz() == {"status": "ok"}
    assert sdk.status()["queue"]["pending"] == 0
    ingested = sdk.ingest_event(
        event_id="decision",
        source_ref="chat",
        content="We decided to use PostgreSQL 18 for GluoMem.",
    )
    original = ingested.frames[0]
    assert original.kind == "decision"
    candidate = sdk.ingest_event(
        event_id="candidate",
        source_ref="doc",
        source_type="document",
        content="Our HNSW m is 16.",
        require_review=True,
    ).frames[0]
    listed = sdk.list_memories(limit=1)
    assert listed.next_cursor == original.id
    assert sdk.list_memories(cursor=listed.next_cursor).review_queue[0].id == candidate.id
    assert sdk.review_memory(candidate.id, "accept").frame.status == "active"
    assert any(hit.frame.id == original.id for hit in sdk.recall("PostgreSQL GluoMem").hits)
    correction = sdk.correct_memory(original.id, "We decided to use PostgreSQL 18 with pgvector.")
    replacement = correction.replacement_frame
    assert replacement.supersedes == original.id
    assert [frame.id for frame in sdk.get_lineage(replacement.id)] == [original.id, replacement.id]
    job = sdk.enqueue_event(event_id="pending", source_ref="chat-2", content="I prefer dark mode.")
    worker = MemoryWorker(mind_db, frame_store, cfg=cfg)
    done = worker.run_once()
    assert done is not None and done.status == "succeeded"
    assert sdk.get_job(job.id).status == "succeeded"
    assert len(sdk.export_memories()["active_memories"]) == 3
    ep = sdk.ingest_event(
        event_id="ep-1",
        source_ref="chat-3",
        content="Yesterday a transient outage occurred.",
        created_at="2026-01-01T00:00:00",
    ).frames[0]
    assert ep.kind == "episode"
    swept = sdk.sweep_expired(now_iso="2027-01-01T00:00:00")
    assert swept.swept_frame_ids == [ep.id]
    sweep_job = sdk.enqueue_sweep_expired(
        now_iso="2027-01-02T00:00:00", dedup_key="sweep-2027-01-02"
    )
    assert sweep_job.job_type == "sweep_expired"
    done_sweep = worker.run_once()
    assert done_sweep is not None and done_sweep.id == sweep_job.id
    assert sdk.retract_source("doc").forgotten_frame_ids == [candidate.id]
    assert sdk.forget_memory(replacement.id).frame_id == replacement.id
    assert len(sdk.list_memories().active_frames) == 1
    with pytest.raises(GluoMemApiError) as error:
        sdk.get_lineage(99999)
    assert error.value.status_code == 404


def test_client_import_does_not_load_storage_or_server() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "importtime",
            "-c",
            "import sys, gluomem.client; "
            "assert not {'numpy', 'sqlite_vec', 'fastapi'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
