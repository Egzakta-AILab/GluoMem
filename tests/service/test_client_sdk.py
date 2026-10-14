"""End-to-end tests for GluoMemClient and AsyncGluoMemClient across all 10 service endpoints."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from gluomem.client.sdk import AsyncGluoMemClient, GluoMemClient
from gluomem.config.app_config import AppConfig
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.service.app import create_app
from gluomem.service.worker import MemoryWorker
from gluomem.storage.db import MindDB
from gluomem.storage.scope import MemoryScope


def test_sync_and_async_client_sdk_full_lifecycle(tmp_path: Path) -> None:
    cfg = AppConfig()
    db = MindDB(tmp_path / "sdk.mind", vector_dim=32)
    embedder = MockEmbedder(dim=32)
    scope = MemoryScope("egzakta", "mikhail", "personal")
    store = FrameStore(db, embedder=embedder, scope=scope)
    app = create_app(db, store, cfg=cfg, service_key="sdk-secret")
    worker = MemoryWorker(db, store, cfg=cfg)

    async def _exercise_async_sdk() -> None:
        transport = httpx.ASGITransport(app=app)
        async with AsyncGluoMemClient(
            "http://testserver",
            service_key="sdk-secret",
            default_scope=scope,
            transport=transport,
        ) as client:
            assert (await client.healthz())["status"] == "ok"
            assert (await client.readyz())["status"] == "ready"

            # 1. Synchronous ingest
            ing = await client.ingest_event(
                event_id="evt-sdk-1",
                source_ref="chat-sdk-1",
                content="We decided to use PostgreSQL 18 for GluoMem.",
            )
            assert len(ing.frames) == 1
            f1 = ing.frames[0]
            assert f1.kind == "decision"

            # 2. Candidate review queue ingest + accept
            cand_ing = await client.ingest_event(
                event_id="evt-sdk-cand",
                source_ref="doc-sdk-1",
                source_type="document",
                content="We decided to use HNSW m=16 ef_construction=64.",
                require_review=True,
            )
            cand_frame = cand_ing.frames[0]
            assert cand_frame.status == "candidate"

            listed = await client.list_memories()
            assert len(listed.review_queue) == 1
            rev = await client.review_memory(cand_frame.id, "accept")
            assert rev.frame is not None and rev.frame.status == "active"

            # 3. Recall
            rec = await client.recall("PostgreSQL 18 GluoMem")
            assert any(h.frame.id == f1.id for h in rec.hits)

            # 4. Correct memory
            corr = await client.correct_memory(
                f1.id,
                "We decided to use PostgreSQL 18 with pgvector for GluoMem.",
            )
            f2 = corr.replacement_frame
            assert f2.supersedes == f1.id

            # 5. Async job enqueue + get_job
            job = await client.enqueue_event(
                event_id="evt-sdk-async",
                source_ref="chat-sdk-2",
                content="I prefer dark mode in the KVARK editor.",
            )
            assert job.status == "queued"
            worker.run_once()
            done_job = await client.get_job(job.id)
            assert done_job.status == "succeeded"

            # 6. Export
            exp = await client.export_memories()
            assert exp["scope"] == scope.as_dict()
            assert len(exp["active_memories"]) == 3

            # 7. Retract source & forget
            ret = await client.retract_source("doc-sdk-1")
            assert ret.forgotten_frame_ids == [cand_frame.id]

            f_res = await client.forget_memory(f2.id)
            assert f_res.frame_id == f2.id

    asyncio.run(_exercise_async_sdk())

    # Also verify synchronous GluoMemClient healthz/readyz using FastAPI TestClient transport
    from fastapi.testclient import TestClient

    tc = TestClient(app)
    sync_client = GluoMemClient(
        "http://testserver",
        service_key="sdk-secret",
        default_scope=scope,
        transport=tc._transport,  # type: ignore[arg-type]
    )
    assert sync_client.healthz()["status"] == "ok"
    listed_after = sync_client.list_memories()
    assert len(listed_after.active_frames) == 1
    assert "dark mode" in listed_after.active_frames[0].content
    sync_client.close()
    db.close()
