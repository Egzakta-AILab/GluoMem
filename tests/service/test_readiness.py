"""API/DB health must not imply queued ingestion is running."""

import pytest
from fastapi.testclient import TestClient

from gluomem.service.app import create_app
from gluomem.service.readiness import WORKER_HEARTBEAT_KEY
from gluomem.service.worker import MemoryWorker
from gluomem.storage.db import MindDB


def test_idle_worker_liveness_is_independent_of_api_and_database(mind_db, frame_store, cfg):
    with TestClient(create_app(mind_db, frame_store, cfg=cfg, service_key="test")) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200
        response = client.get("/readyz/worker")
        assert response.status_code == 503
        assert response.json()["checks"] == {"database": True, "worker": False}
        worker = MemoryWorker(mind_db, frame_store, cfg=cfg)
        assert worker.run_once() is None
        response = client.get("/readyz/worker")
        assert response.status_code == 200
        assert response.json()["checks"] == {"database": True, "worker": True}


def test_db_readiness_is_503_when_probe_connection_fails(mind_db, frame_store, cfg, monkeypatch):
    with TestClient(create_app(mind_db, frame_store, cfg=cfg, service_key="test")) as client:

        def broken(*args, **kwargs):
            raise RuntimeError("sensitive driver connection details")

        monkeypatch.setattr(MindDB, "scalar", broken)
        for path in ("/readyz", "/readyz/worker"):
            response = client.get(path)
            assert response.status_code == 503
            assert "sensitive" not in response.text
        assert client.get("/healthz").status_code == 200


def test_probe_cleanup_failure_cannot_leave_worker_ready(mind_db, frame_store, cfg, monkeypatch):
    MemoryWorker(mind_db, frame_store, cfg=cfg).run_once()
    original_close = MindDB.close

    def failing_close(handle):
        original_close(handle)
        raise RuntimeError("synthetic connection cleanup failure")

    with TestClient(create_app(mind_db, frame_store, cfg=cfg, service_key="test")) as client:
        monkeypatch.setattr(MindDB, "close", failing_close)
        for path in ("/readyz/worker", "/readyz/ingestion"):
            response = client.get(path)
            assert response.status_code == 503
            assert response.json()["checks"]["database"] is False
        monkeypatch.setattr(MindDB, "close", original_close)


def test_ingestion_readiness_does_not_treat_fake_components_as_ready(mind_db, frame_store, cfg):
    cfg = cfg.with_overrides(
        {
            "embedding.model": "mock",
            "write.extraction_backend": "fake",
            "components.allow_fake": True,
        }
    )
    MemoryWorker(mind_db, frame_store, cfg=cfg).run_once()
    with TestClient(create_app(mind_db, frame_store, cfg=cfg, service_key="test")) as client:
        assert client.get("/readyz/worker").status_code == 200
        response = client.get("/readyz/ingestion")
        assert response.status_code == 503
        assert response.json()["checks"] == {
            "database": True,
            "worker": True,
            "components_config": False,
            "backlog": True,
        }


def test_ingestion_backlog_reports_old_pending_jobs(mind_db, frame_store, cfg):
    from gluomem.service.jobs import JobQueue

    MemoryWorker(mind_db, frame_store, cfg=cfg).run_once()
    JobQueue(mind_db.backend).enqueue(dedup_key="old", now_iso="2000-01-01")
    with TestClient(create_app(mind_db, frame_store, cfg=cfg, service_key="test")) as client:
        assert client.get("/readyz/worker").status_code == 200
        response = client.get("/readyz/ingestion")
        assert response.status_code == 503
        assert response.json()["checks"]["backlog"] is False


@pytest.mark.parametrize("stamp", ["2000-01-01", "2100-01-01", "corrupt"])
def test_stale_future_or_corrupt_worker_is_not_ready(mind_db, frame_store, cfg, stamp):
    mind_db.meta_set(WORKER_HEARTBEAT_KEY, stamp)
    with TestClient(create_app(mind_db, frame_store, cfg=cfg, service_key="test")) as client:
        assert client.get("/readyz").status_code == 200
        assert client.get("/readyz/worker").status_code == 503


def test_paused_worker_does_not_announce_liveness(mind_db, frame_store, cfg):
    worker = MemoryWorker(mind_db, frame_store, cfg=cfg, should_pause=lambda: True)
    worker.run_once()
    assert mind_db.meta_get(WORKER_HEARTBEAT_KEY) is None
