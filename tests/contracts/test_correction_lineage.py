"""Tests for correct_memory supersession, version_lineage, list_memories, and export_memories."""

from __future__ import annotations

from pathlib import Path

from gluomem.config.app_config import AppConfig
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.lifecycle import (
    correct_memory,
    export_memories,
    list_memories,
    version_lineage,
)
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.search.hybrid import HybridSearch
from gluomem.storage.db import MindDB
from gluomem.storage.scope import MemoryScope
from gluomem.write.events import SourceEvent, ingest_source_event


def test_correct_memory_supersedes_old_frame_and_preserves_lineage(
    tmp_path: Path,
) -> None:
    cfg = AppConfig()
    db = MindDB(tmp_path / "correct.mind", vector_dim=32)
    embedder = MockEmbedder(dim=32)
    scope = MemoryScope("egzakta", "mikhail", "personal")
    store = FrameStore(db, embedder=embedder, scope=scope)
    kg = KnowledgeGraph(db, scope=scope)
    search = HybridSearch(db, store, cfg, embedder=embedder, kg=kg, scope=scope)

    evt = SourceEvent(
        event_id="evt-corr-1",
        revision=1,
        source_type="chat_turn",
        source_ref="conv-release",
        scope=scope,
        content="We decided that release freeze is on Thursday at 18:00.",
    )
    f1 = ingest_source_event(evt, store=store).frames[0]
    assert f1.kind == "decision"
    assert f1.status == "active"

    # First correction: Thursday -> Friday
    corr1 = correct_memory(
        store,
        f1.id,
        "We decided that release freeze is on Friday at 16:00.",
        scope=scope,
    )
    f2 = corr1.replacement_frame
    assert corr1.superseded_frame.status == "superseded"
    assert corr1.superseded_frame.superseded_by == f2.id
    assert f2.status == "active"
    assert f2.supersedes == f1.id
    assert f2.verified_by_user is True

    # Second correction: Friday -> Monday
    corr2 = correct_memory(
        store,
        f2.id,
        "We decided that release freeze is on Monday at 09:00.",
        scope=scope,
    )
    f3 = corr2.replacement_frame
    assert f3.status == "active"
    assert f3.supersedes == f2.id

    # Superseded texts ("Thursday", "Friday") return 0 FTS hits; only "Monday" is active
    assert db.fts.search('"Thursday"', 10, scope=scope) == []
    assert db.fts.search('"Friday"', 10, scope=scope) == []
    assert db.fts.search('"Monday"', 10, scope=scope) == [f3.id]

    hits = search.search("release freeze", scope=scope).hits
    assert [h.frame.id for h in hits] == [f3.id]

    # Version lineage returns the full chronological chain [f1, f2, f3] from any node
    chain = version_lineage(store, f2.id, scope=scope)
    assert [f.id for f in chain] == [f1.id, f2.id, f3.id]
    assert [f.status for f in chain] == ["superseded", "superseded", "active"]

    # FrameStore lineage audit records the 'correct' operation
    audit_f3 = store.lineage(f3.id)
    assert len(audit_f3) == 1
    assert audit_f3[0].source_frame_id == f2.id
    assert audit_f3[0].operation == "correct"

    # list_memories and export_memories include only the active frame f3
    grouped = list_memories(store, scope=scope)
    assert [f.id for f in grouped.active_frames] == [f3.id]
    assert [f.id for f in grouped.by_kind["decision"]] == [f3.id]

    exported = export_memories(store, scope=scope)
    assert exported["scope"] == scope.as_dict()
    assert [m["id"] for m in exported["active_memories"]] == [f3.id]
    assert exported["active_memories"][0]["lineage"][0]["operation"] == "correct"

    db.close()
