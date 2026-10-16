"""Offline restore targets remain blocked without an authoritative deletion journal."""

import pytest

from gluomem.frames.store import FrameStore


@pytest.mark.parametrize("state", ["blocked", "unknown"])
def test_blocked_restore_rejects_api_reads_and_worker_polls(mind_db, frame_store, cfg, state):
    from fastapi.testclient import TestClient

    from gluomem.service.app import create_app
    from gluomem.service.worker import MemoryWorker

    mind_db.meta_set("operations.restore_state", state)
    with TestClient(create_app(mind_db, frame_store, cfg=cfg, service_key="test")) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503
        response = client.get(
            "/v1/status",
            headers={
                "Authorization": "Bearer test",
                "X-GluoMem-Tenant": "local",
                "X-GluoMem-Principal": "default",
            },
        )
        assert response.status_code == 503
    with pytest.raises(RuntimeError, match="restore"):
        MemoryWorker(mind_db, frame_store, cfg=cfg).run_once()


def test_missing_deletion_journal_blocks_restored_store(mind_db, frame_store, tmp_path):
    from gluomem.service.restore import reconcile_restored_store

    with pytest.raises(FileNotFoundError):
        reconcile_restored_store(frame_store, tmp_path / "missing-journal.json")
    assert mind_db.meta_get("operations.restore_state") == "blocked"


@pytest.mark.parametrize(
    "journal",
    [
        {"version": 1, "complete": False, "actions": []},
        {"version": 1, "complete": True, "actions": [{"operation": "forget_memory"}]},
        {"version": 99, "complete": True, "actions": []},
    ],
)
def test_invalid_or_uncovered_journal_keeps_restore_blocked(
    mind_db, frame_store, tmp_path, journal
):
    import json

    from gluomem.service.restore import reconcile_restored_store

    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(journal))
    with pytest.raises(ValueError):
        reconcile_restored_store(frame_store, path)
    assert mind_db.meta_get("operations.restore_state") == "blocked"


def test_replay_failure_rolls_back_but_leaves_target_blocked(
    mind_db, frame_store, tmp_path, monkeypatch
):
    import json

    from gluomem.frames import lifecycle
    from gluomem.service.restore import reconcile_restored_store

    journal = tmp_path / "replay-failure.json"
    journal.write_text(
        json.dumps(
            {
                "version": 1,
                "complete": True,
                "actions": [
                    {
                        "operation": "retract_source",
                        "scope": list(frame_store.scope.as_tuple()),
                        "source_ref": ref,
                    }
                    for ref in ("test:first", "test:second")
                ],
            }
        )
    )
    original = lifecycle.retract_source

    def fail_second(store, source_ref, **kwargs):
        if source_ref == "test:second":
            raise RuntimeError("synthetic replay failure")
        return original(store, source_ref, **kwargs)

    monkeypatch.setattr(lifecycle, "retract_source", fail_second)
    with pytest.raises(RuntimeError, match="synthetic replay failure"):
        reconcile_restored_store(frame_store, journal)
    assert mind_db.meta_get("operations.restore_state") == "blocked"
    assert mind_db.scalar("SELECT COUNT(*) FROM source_retractions") == 0


def test_restore_replays_authorized_scope_erasure(mind_db, frame_store, cfg, tmp_path):
    import json

    from gluomem.service.restore import reconcile_restored_store
    from gluomem.write.events import SourceEvent, ingest_source_event

    result = ingest_source_event(
        SourceEvent(
            event_id="scope-original", source_ref="scope:source", content="Synthetic scope data"
        ),
        store=frame_store,
        cfg=cfg,
    )
    journal = tmp_path / "scope-deletions.json"
    journal.write_text(
        json.dumps(
            {
                "version": 1,
                "complete": True,
                "actions": [
                    {"operation": "forget_scope", "scope": list(frame_store.scope.as_tuple())}
                ],
            }
        )
    )
    reconcile_restored_store(frame_store, journal)
    assert frame_store.get_frame(result.frames[0].id).content != "Synthetic scope data"


def test_restore_replays_later_source_deletion(
    mind_db, frame_store, cfg, tmp_path, assert_no_recall
):
    import json

    from gluomem.frames.lifecycle import retract_source
    from gluomem.service.restore import reconcile_restored_store
    from gluomem.write.events import SourceEvent, ingest_source_event

    event = SourceEvent(
        event_id="before-backup", source_ref="test:deleted", content="Synthetic secret"
    )
    result = ingest_source_event(event, store=frame_store, cfg=cfg)
    frame_id = result.frames[0].id
    # The backend-specific fixture takes and restores an actual backup.
    backup_restore = getattr(mind_db, "_restore_drill", None)
    if backup_restore is None:
        import shutil

        from gluomem.storage.db import MindDB

        backup = tmp_path / "before-delete.mind"
        restored = tmp_path / "restored.mind"
        mind_db.backend.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        shutil.copy2(mind_db.path, backup)
        retract_source(frame_store, "test:deleted")
        shutil.copy2(backup, restored)
        target = MindDB(restored, dim=mind_db.dim)
    else:
        target = backup_restore(lambda: retract_source(frame_store, "test:deleted"))
    try:
        store = FrameStore(target, frame_store.embedder)
        assert store.get_frame(frame_id).content == "Synthetic secret"
        journal = tmp_path / "deletions.json"
        journal.write_text(
            json.dumps(
                {
                    "version": 1,
                    "complete": True,
                    "actions": [
                        {
                            "operation": "retract_source",
                            "scope": list(store.scope.as_tuple()),
                            "source_ref": "test:deleted",
                        }
                    ],
                }
            )
        )
        receipt = reconcile_restored_store(store, journal)
        assert receipt == {"actions_applied": 1, "state": "reconciled"}
        assert target.meta_get("operations.restore_state") == "reconciled"
        assert store.get_frame(frame_id).content != "Synthetic secret"
        with pytest.raises(ValueError, match="retracted"):
            ingest_source_event(event, store=store, cfg=cfg)
        assert "Synthetic secret" not in str(
            [dict(r) for r in target.fetchall("SELECT * FROM source_events")]
        )
        assert target.scalar("SELECT COUNT(*) FROM source_retractions") == 1
        from gluomem.kg.graph import KnowledgeGraph

        assert_no_recall("Synthetic secret", db=target, store=store, kg=KnowledgeGraph(target))
    finally:
        target.close()
