"""SQLite v1-to-v2 source fences preserve existing projections and old retractions."""

from pathlib import Path

import pytest

from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.lifecycle import retract_source
from gluomem.frames.source_fences import SourceRetractedError
from gluomem.frames.store import FrameStore
from gluomem.storage.db import MindDB
from gluomem.write.events import SourceEvent, ingest_source_event


def test_sqlite_v1_upgrade_preserves_data_and_backfills_old_retractions(tmp_path: Path, cfg):
    path = tmp_path / "old.mind"
    db = MindDB(path, dim=384)
    store = FrameStore(db, embedder=MockEmbedder(dim=384), chunking=cfg.chunking)
    event = SourceEvent(event_id="legacy", source_ref="doc:legacy", content="Legacy secret")
    ingest_source_event(event, store=store, cfg=cfg)
    retract_source(store, event.source_ref)
    kept = store.create_i_frame("kept", "Keep indexed content")
    # Reconstruct a real v1 file: same base schema, no durable source fence table.
    db.execute("DROP TABLE source_retractions")
    db.execute("PRAGMA user_version=1")
    db.commit()
    db.close()
    for _ in range(2):
        db = MindDB(path, dim=384)
        store = FrameStore(db, embedder=MockEmbedder(dim=384), chunking=cfg.chunking)
        try:
            assert db.scalar("PRAGMA user_version") == 2
            assert store.get_frame(kept.id).content == "Keep indexed content"
            assert db.fts.search("indexed", 10) == [kept.id]
            assert kept.id in db.frame_vec.ids()
            assert db.scalar("SELECT COUNT(*) FROM source_retractions") == 1
            with pytest.raises(SourceRetractedError):
                ingest_source_event(
                    event.model_copy(update={"event_id": "new-legacy"}), store=store, cfg=cfg
                )
        finally:
            db.close()


def test_source_fence_survives_database_restart(tmp_path: Path, cfg):
    path = tmp_path / "restart.mind"
    for attempt in range(3):
        db = MindDB(path, dim=384)
        store = FrameStore(db, embedder=MockEmbedder(dim=384), chunking=cfg.chunking)
        try:
            if attempt == 0:
                retract_source(store, "doc:before-receipt")
            with pytest.raises(SourceRetractedError):
                ingest_source_event(
                    SourceEvent(
                        event_id=f"new-{attempt}",
                        source_ref="doc:before-receipt",
                        content="No resurrection",
                        force_reassert=True,
                    ),
                    store=store,
                    cfg=cfg,
                )
            assert db.scalar("SELECT COUNT(*) FROM memory_frames") == 0
        finally:
            db.close()
