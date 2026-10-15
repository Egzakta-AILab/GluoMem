"""Tests for Iteration 6 (KVAD-1373: Waves 1-4)."""

from __future__ import annotations

import asyncio
import io
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from gluomem.client.sdk import AsyncGluoMemClient, EmbeddedMemoryClient, GluoMemClient
from gluomem.config.app_config import AppConfig, StorageConfig, load_config
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.llm.bootstrap import ArtifactError, _sha256, load_or_build, save_json_artifact
from gluomem.llm.cache import CachedExtractors
from gluomem.llm.fake import FakeExtractors
from gluomem.llm.modules import build_module, make_extractors
from gluomem.logging import configure_logging
from gluomem.service.app import create_app
from gluomem.service.worker import MemoryWorker
from gluomem.storage.db import MindDB
from gluomem.storage.postgres_backend import (
    PostgresBackend,
    PostgresConnectionPool,
    _configure_connection,
)
from gluomem.storage.scope import MemoryScope
from gluomem.write.events import (
    SourceEvent,
    commit_source_event,
    prepare_source_event,
)
from integrations.hermes import HermesMemoryAdapter, load_hermes_mcp_config
from integrations.kvark import KvarkMemoryAdapter, kvark_scope
from integrations.openclaw import load_openclaw_hooks_config, load_openclaw_mcp_config

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _fake_cfg(tmp_path: Path | None = None) -> AppConfig:
    overrides: dict[str, Any] = {
        "components.allow_fake": True,
        "embedding.model": "mock",
        "models.mock.dim": 16,
        "storage.vector_dim": 16,
        "write.extraction_backend": "fake",
        "search.reranker.enabled": False,
    }
    if tmp_path is not None:
        overrides["storage.path"] = str(tmp_path / "test.mind")
    return load_config(overrides=overrides)


# ---------------------------------------------------------------------------
# Wave 1 — Postgres Connection Pool & Startup-Only Schema Verification
# ---------------------------------------------------------------------------


def test_storage_config_pool_bounds() -> None:
    cfg = StorageConfig()
    assert cfg.pool_min_size == 2
    assert cfg.pool_max_size == 16

    with pytest.raises(ValueError, match="pool_max_size"):
        StorageConfig(pool_min_size=8, pool_max_size=4)


def test_configure_connection_and_borrowed_postgres_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[str] = []
    registered: list[Any] = []

    class FakeConn:
        def __init__(self) -> None:
            self.closed = False

        def execute(self, sql: str, params: Any = None) -> Any:
            executed.append(sql)
            res = MagicMock()
            if "to_regclass" in sql:
                res.fetchone.return_value = {"to_regclass": "gluomem.alembic_version"}
            elif "FROM gluomem.alembic_version" in sql:
                from gluomem.migrations.runner import get_heads

                res.fetchall.return_value = [{"version_num": h} for h in get_heads()]
            elif "pg_attribute" in sql:
                res.fetchone.return_value = {"atttypmod": 16}
            return res

        def close(self) -> None:
            self.closed = True

    import pgvector.psycopg

    monkeypatch.setattr(pgvector.psycopg, "register_vector", lambda c: registered.append(c))

    conn = FakeConn()
    _configure_connection(conn, "gluomem")
    assert executed == [
        "SET search_path TO gluomem, public",
        "SET hnsw.iterative_scan = strict_order",
        "SET hnsw.max_scan_tuples = 20000",
        "SET hnsw.scan_mem_multiplier = 4",
    ]
    assert registered == [conn]

    executed.clear()
    borrowed = PostgresBackend(
        "postgresql://localhost/test",
        dim=16,
        postgres_schema="gluomem",
        conn=conn,
        verify_schema=False,
        owns_connection=False,
    )
    assert executed == []
    borrowed.close()
    assert conn.closed is False


def test_postgres_pool_verifies_schema_once_and_service_skips_catalog_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_queries: list[str] = []

    class FakePooledConn:
        def __init__(self) -> None:
            self.closed = False

        def execute(self, sql: str, params: Any = ()) -> Any:
            if "to_regclass" in sql or "alembic_version" in sql or "pg_attribute" in sql:
                catalog_queries.append(sql)
            res = MagicMock()
            if "to_regclass" in sql:
                res.fetchone.return_value = {"to_regclass": "gluomem.alembic_version"}
            elif "alembic_version" in sql:
                from gluomem.migrations.runner import get_heads

                res.fetchall.return_value = [{"version_num": h} for h in get_heads()]
            elif "pg_attribute" in sql:
                res.fetchone.return_value = {"atttypmod": 16}
            elif "SELECT 1" in sql:
                res.fetchone.return_value = {"?column?": 1}
            else:
                res.fetchone.return_value = None
                res.fetchall.return_value = []
            return res

        @contextmanager
        def transaction(self):
            yield

        def commit(self) -> None:
            pass

        def rollback(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    shared_conn = FakePooledConn()

    class FakeUnderlyingPool:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.opened = False
            self.closed = False

        def open(self, *, wait: bool = True) -> None:
            self.opened = True

        @contextmanager
        def connection(self):
            yield shared_conn

        def close(self) -> None:
            self.closed = True

    import psycopg_pool

    monkeypatch.setattr(psycopg_pool, "ConnectionPool", FakeUnderlyingPool)

    pool = PostgresConnectionPool(
        "postgresql://localhost/test",
        dim=16,
        postgres_schema="gluomem",
        min_size=2,
        max_size=8,
    )
    assert len(catalog_queries) == 4  # to_regclass + alembic_version + 2 vec tables at startup
    catalog_queries.clear()

    with pool.borrow_backend() as backend:
        assert backend.scalar("SELECT 1") == 1
    assert catalog_queries == []
    assert shared_conn.closed is False

    # Verify create_app runtime() with a Postgres MindDB uses the pool
    # and executes 0 catalog queries per request.
    cfg = _fake_cfg()
    pg_db = MindDB(
        dim=16,
        backend_type="postgres",
        dsn="postgresql://localhost/test",
        postgres_schema="gluomem",
        conn=shared_conn,
        verify_schema=False,
        owns_connection=False,
    )
    from gluomem.embedding.providers import make_embedder

    fake_store = MagicMock(spec=FrameStore)
    fake_store.embedder = make_embedder(cfg)
    fake_store.chunking = cfg.chunking
    fake_store.scope = MemoryScope()

    app = create_app(pg_db, fake_store, cfg=cfg, service_key="secret", pg_pool=pool)
    with TestClient(app) as client:
        for _ in range(3):
            resp = client.get("/readyz")
            assert resp.status_code == 200
    assert catalog_queries == []
    pool.close()


# ---------------------------------------------------------------------------
# Wave 2 — make_extractors, Two-Phase ingest_source_event & Worker Locking
# ---------------------------------------------------------------------------


def test_make_extractors_and_embedded_open_wiring(tmp_path: Path) -> None:
    cfg = _fake_cfg(tmp_path)
    ext = make_extractors(cfg)
    assert isinstance(ext, FakeExtractors)

    cached_cfg = cfg.with_overrides(
        {
            "write.extraction_cache.mode": "readthrough",
            "write.extraction_cache.path": str(tmp_path / "cache.sqlite"),
        }
    )
    cached_ext = make_extractors(cached_cfg)
    assert isinstance(cached_ext, CachedExtractors)

    with EmbeddedMemoryClient.open(tmp_path / "embedded.mind", cfg=cfg) as client:
        assert isinstance(client.extractors, FakeExtractors)
        res = client.ingest_event(
            event_id="ev-chat-1",
            source_type="chat_turn",
            source_ref="session:1",
            content="We migrated our primary cache to Valkey.",
        )
        assert len(res.frames) >= 1


def test_two_phase_ingest_and_worker_locks_only_in_phase_two(tmp_path: Path) -> None:
    cfg = _fake_cfg(tmp_path)
    from gluomem.embedding.providers import make_embedder

    db = MindDB(tmp_path / "worker.mind", dim=16)
    store = FrameStore(db, embedder=make_embedder(cfg), chunking=cfg.chunking)
    kg = KnowledgeGraph(db)

    tx_states_during_extract: list[bool] = []
    progress_calls: list[int] = []

    class SpyExtractors(FakeExtractors):
        def extract_event_claims(self, source_text: str, *, reference_date: Any = None):
            tx_states_during_extract.append(db.in_transaction)
            return super().extract_event_claims(source_text, reference_date=reference_date)

        def consolidate_claim(self, new_claim: str, candidates: Any):
            tx_states_during_extract.append(db.in_transaction)
            return super().consolidate_claim(new_claim, candidates)

    spy = SpyExtractors()
    ev1 = SourceEvent(
        event_id="p1",
        source_type="chat_turn",
        source_ref="chat:1",
        content="Deployment target is Kubernetes cluster prod-eu.",
    )
    prep = prepare_source_event(
        ev1,
        store=store,
        kg=kg,
        cfg=cfg,
        extractors=spy,
        on_progress=lambda: progress_calls.append(1),
    )
    assert db.in_transaction is False
    assert all(in_tx is False for in_tx in tx_states_during_extract)
    assert len(progress_calls) >= 1

    committed = commit_source_event(prep, store=store, kg=kg)
    assert len(committed.frames) == 1

    # Enqueue a second event through MemoryWorker and verify Phase 1 runs
    # before _assert_not_fenced.
    tx_states_during_extract.clear()
    fenced_called: list[bool] = []
    extract_saw_fenced: list[bool] = []

    class WorkerSpyExtractors(FakeExtractors):
        def extract_event_claims(self, source_text: str, *, reference_date: Any = None):
            extract_saw_fenced.append(bool(fenced_called))
            tx_states_during_extract.append(db.in_transaction)
            return super().extract_event_claims(source_text, reference_date=reference_date)

    worker = MemoryWorker(db, store, cfg=cfg, kg=kg, extractors=WorkerSpyExtractors())
    orig_assert = worker._assert_not_fenced

    def tracked_assert(job: Any) -> None:
        fenced_called.append(True)
        assert db.in_transaction is True
        orig_assert(job)

    worker._assert_not_fenced = tracked_assert  # type: ignore[method-assign]
    worker.queue.enqueue(
        store.scope,
        "ingest_event",
        "ev-2:1",
        {
            "event_id": "ev-2",
            "revision": 1,
            "source_type": "chat_turn",
            "source_ref": "chat:2",
            "content": "Alerting routes through PagerDuty.",
        },
    )
    done = worker.run_once()
    assert done is not None and done.status == "succeeded"
    assert extract_saw_fenced == [False]
    assert tx_states_during_extract == [False]
    assert fenced_called == [True]
    db.close()


# ---------------------------------------------------------------------------
# Wave 3 — Pinned DSPy Artifacts & Hash Verification
# ---------------------------------------------------------------------------


def test_pinned_dspy_json_artifacts_load_and_detect_tampering(tmp_path: Path) -> None:
    cfg = load_config()
    manifest_path = _REPO_ROOT / "artifacts" / "dspy" / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for task in ("event_claims", "consolidate"):
        artifact_file = _REPO_ROOT / "artifacts" / "dspy" / f"{task}.json"
        assert artifact_file.is_file()
        assert manifest[task] == _sha256(artifact_file)
        assert manifest["artifacts"][task]["sha256"] == _sha256(artifact_file)
        loaded = load_or_build(task, cfg)
        assert loaded is not None

    # Verify tamper detection on JSON artifacts
    custom_dir = tmp_path / "dspy_artifacts"
    mod = build_module("event_claims", cfg)
    json_path = save_json_artifact(mod, custom_dir, "event_claims", model_ref="deepseek-flash")
    custom_cfg = cfg.with_overrides(
        {"dspy.artifacts.dir": str(custom_dir), "dspy.artifacts.require": True}
    )
    assert load_or_build("event_claims", custom_cfg) is not None

    json_path.write_text(json_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="hash mismatch"):
        load_or_build("event_claims", custom_cfg)


# ---------------------------------------------------------------------------
# Wave 4 — Structured JSON Logging, Dockerfile & Host Integration Bundles
# ---------------------------------------------------------------------------


def test_structured_json_logging_service_and_worker(tmp_path: Path) -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)

    cfg = _fake_cfg(tmp_path)
    from gluomem.embedding.providers import make_embedder

    db = MindDB(tmp_path / "logs.mind", dim=16)
    store = FrameStore(db, embedder=make_embedder(cfg), chunking=cfg.chunking)
    app = create_app(db, store, cfg=cfg, service_key="secret", extractors=FakeExtractors())

    headers = {
        "Authorization": "Bearer secret",
        "X-GluoMem-Tenant": "acme",
        "X-GluoMem-Principal": "u-1",
        "X-GluoMem-Namespace": "ws-1",
    }
    with TestClient(app) as client:
        resp = client.post(
            "/v1/events",
            headers=headers,
            json={
                "event_id": "log-ev-1",
                "source_type": "chat_turn",
                "source_ref": "chat:1",
                "content": "Grafana dashboards live in monitoring repo.",
            },
        )
        assert resp.status_code == 202

    worker = MemoryWorker(db, store, cfg=cfg, extractors=FakeExtractors())
    done = worker.run_once()
    assert done is not None and done.status == "succeeded"

    records = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    http_logs = [r for r in records if r.get("event") == "http_request"]
    assert len(http_logs) >= 1
    assert http_logs[-1]["method"] == "POST"
    assert http_logs[-1]["path"] == "/v1/events"
    assert http_logs[-1]["status"] == 202
    assert isinstance(http_logs[-1]["latency_ms"], float)
    assert http_logs[-1]["scope"] == {
        "tenant_id": "acme",
        "principal_id": "u-1",
        "namespace": "ws-1",
    }

    job_logs = [r for r in records if r.get("event") == "job_succeeded"]
    assert len(job_logs) == 1
    assert job_logs[0]["job_id"] == done.id
    assert isinstance(job_logs[0]["duration_ms"], float)
    db.close()


def test_dockerfile_and_host_integration_bundles(tmp_path: Path) -> None:
    dockerfile = (_REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.12-slim" in dockerfile
    assert "USER gluomem" in dockerfile
    assert 'ENTRYPOINT ["gluomem"]' in dockerfile
    assert (_REPO_ROOT / ".dockerignore").is_file()

    # OpenClaw & Hermes static configs
    oc_hooks = load_openclaw_hooks_config()
    assert "gluomem hook recall" in oc_hooks["hooks"]["beforeSubmitPrompt"][0]["command"]
    assert "gluomem hook capture --async" in oc_hooks["hooks"]["stop"][0]["command"]
    assert "gluomem" in load_openclaw_mcp_config()["mcpServers"]
    assert "gluomem" in load_hermes_mcp_config()["mcpServers"]

    # KvarkMemoryAdapter + HermesMemoryAdapter end-to-end
    cfg = _fake_cfg(tmp_path)
    from gluomem.embedding.providers import make_embedder

    db = MindDB(tmp_path / "integrations.mind", dim=16)
    store = FrameStore(db, embedder=make_embedder(cfg), chunking=cfg.chunking)
    app = create_app(db, store, cfg=cfg, service_key="secret", extractors=FakeExtractors())

    async def _exercise_kvark() -> None:
        transport = httpx.ASGITransport(app=app)
        async with AsyncGluoMemClient(
            "http://test", service_key="secret", transport=transport
        ) as aclient:
            adapter = KvarkMemoryAdapter(aclient, recall_timeout_seconds=2.0)
            assert adapter.scope_for("org-9", "user-4", "ws-2") == kvark_scope(
                "org-9", "user-4", "ws-2"
            )
            await adapter.remember_explicit(
                "All production deploys require two approvals.",
                organization_id="org-9",
                user_id="user-4",
                workspace_id="ws-2",
            )
            ctx = await adapter.recall(
                "production deploys approvals",
                organization_id="org-9",
                user_id="user-4",
                workspace_id="ws-2",
            )
            assert "<memory_context>" in ctx
            assert "two approvals" in ctx

            job = await adapter.capture_turn(
                organization_id="org-9",
                user_id="user-4",
                workspace_id="ws-2",
                session_id="sess-1",
                turn_id="t-1",
                user_message="<thinking>plan</thinking>We use Vault for secrets.",
                assistant_message="Noted.",
            )
            assert job is not None
            assert job.status == "queued"

        # Fail-open recall when service is unreachable
        def _boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        async with AsyncGluoMemClient(
            "http://unreachable",
            service_key="secret",
            transport=httpx.MockTransport(_boom),
        ) as broken_client:
            broken_adapter = KvarkMemoryAdapter(broken_client, recall_timeout_seconds=0.5)
            assert (
                await broken_adapter.recall(
                    "anything",
                    organization_id="org-9",
                    user_id="user-4",
                )
                == ""
            )

    asyncio.run(_exercise_kvark())

    with (
        TestClient(app) as tc,
        GluoMemClient(
            "http://testserver", service_key="secret", transport=tc._transport
        ) as sync_client,
    ):
        hermes = HermesMemoryAdapter(
            sync_client, tenant_id="org-9", principal_id="user-4", namespace="hermes"
        )
        assert len(hermes.openai_tools()) == 6
        assert len(hermes.mcp_tools()) == 6
        rem = hermes.dispatch(
            "memory_remember",
            {"content": "Hermes agent uses strict JSON schemas.", "kind": "fact"},
        )
        assert len(rem["frames"]) == 1

    db.close()
