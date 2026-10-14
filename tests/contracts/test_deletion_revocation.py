"""Tests for hard forget_memory, retract_source, KG retirement, and replay tombstones."""

from __future__ import annotations

from pathlib import Path

from gluomem.config.app_config import AppConfig
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.lifecycle import (
    export_memories,
    forget_memory,
    get_scope_generation,
    list_memories,
    retract_source,
)
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.search.hybrid import HybridSearch
from gluomem.storage.db import MindDB
from gluomem.storage.scope import MemoryScope
from gluomem.write.events import SourceEvent, ingest_source_event


def test_forget_memory_removes_from_all_surfaces_and_blocks_replay(
    tmp_path: Path,
) -> None:
    cfg = AppConfig()
    db = MindDB(tmp_path / "forget.mind", vector_dim=32)
    embedder = MockEmbedder(dim=32)
    scope = MemoryScope("egzakta", "mikhail", "personal")
    store = FrameStore(db, embedder=embedder, scope=scope)
    kg = KnowledgeGraph(db, scope=scope)
    search = HybridSearch(db, store, cfg, embedder=embedder, kg=kg, scope=scope)

    evt = SourceEvent(
        event_id="evt-del-1",
        revision=1,
        source_type="chat_turn",
        source_ref="conv-secret",
        scope=scope,
        content="Project Kraken uses vault passcode 9941-ALPHA.",
    )
    ingested = ingest_source_event(evt, store=store)
    frame = ingested.frames[0]

    # Attach KG entities + relation to the frame
    e_proj = kg.create_entity("project", "Kraken", scope=scope)
    e_code = kg.create_entity("secret", "9941-ALPHA", scope=scope)
    kg.link_entity_frame(e_proj.id, frame.id, scope=scope)
    kg.link_entity_frame(e_code.id, frame.id, scope=scope)
    rel = kg.create_relation(e_proj.id, e_code.id, "uses_passcode", scope=scope)

    assert len(search.search("Kraken 9941-ALPHA", scope=scope).hits) == 1
    assert kg.contextual("Kraken", cfg.kg, depth=2, scope=scope) == {frame.id: 0}
    assert get_scope_generation(db.backend, scope) == 0

    # Execute hard forget
    f_res = forget_memory(store, frame.id, scope=scope)
    assert f_res.frame_id == frame.id
    assert f_res.reason == "forgotten"
    assert set(f_res.retired_entity_ids) == {e_proj.id, e_code.id}
    assert f_res.retired_relation_ids == [rel.id]
    assert f_res.scope_generation == 1

    # 0 hits across FTS, vector KNN, HybridSearch, KG contextual, reconstruct_state, list/export
    qvec = embedder.embed("Project Kraken uses vault passcode 9941-ALPHA.")
    assert db.fts.search('"Kraken"', 10, scope=scope) == []
    assert db.frame_vec.search(qvec, 10, scope=scope) == []
    assert db.chunk_vec.search(qvec, 10, scope=scope) == []
    assert search.search("Kraken 9941-ALPHA", scope=scope).hits == []
    assert kg.contextual("Kraken", cfg.kg, depth=2, scope=scope) == {}
    latest_i, p_frames = store.reconstruct_state(frame.gop_id, scope=scope)
    assert latest_i is None and p_frames == []
    assert list_memories(store, scope=scope).active_frames == []
    assert export_memories(store, scope=scope)["active_memories"] == []

    # Stored row is scrubbed and tombstone stores only SHA-256 content_hash (never plaintext)
    scrubbed = store.get_frame(frame.id, scope=scope)
    assert scrubbed.status == "forgotten"
    assert scrubbed.content == "[forgotten]"
    tomb_rows = db.backend.fetchall(
        "SELECT content_hash, reason FROM memory_tombstones "
        "WHERE tenant_id = ? AND principal_id = ? AND namespace = ?",
        scope.as_tuple(),
    )
    assert len(tomb_rows) == 1
    assert tomb_rows[0]["content_hash"] == f_res.content_hash
    assert "9941-ALPHA" not in str(dict(tomb_rows[0]))

    # Replaying a new SourceEvent with the forgotten text is blocked by memory_tombstones
    replay_evt = SourceEvent(
        event_id="evt-del-replay",
        revision=1,
        source_type="chat_turn",
        source_ref="conv-another",
        scope=scope,
        content="Project Kraken uses vault passcode 9941-ALPHA.",
    )
    replay_res = ingest_source_event(replay_evt, store=store)
    assert replay_res.frames == []
    assert replay_res.suppressed_hashes == [f_res.content_hash]
    assert search.search("Kraken 9941-ALPHA", scope=scope).hits == []

    db.close()


def test_retract_source_forgets_sole_source_and_preserves_corroborated(
    tmp_path: Path,
) -> None:
    cfg = AppConfig()
    db = MindDB(tmp_path / "retract.mind", vector_dim=32)
    embedder = MockEmbedder(dim=32)
    scope = MemoryScope("egzakta", "mikhail", "workspace")
    store = FrameStore(db, embedder=embedder, scope=scope)
    kg = KnowledgeGraph(db, scope=scope)
    search = HybridSearch(db, store, cfg, embedder=embedder, kg=kg, scope=scope)

    # Frame A: supported ONLY by doc-revoked
    evt_only_doc = SourceEvent(
        event_id="evt-doc-1",
        revision=1,
        source_type="document",
        source_ref="doc-revoked",
        scope=scope,
        content="Internal acquisition target is codenamed Aurora.",
    )
    frame_a = ingest_source_event(evt_only_doc, store=store).frames[0]

    # Frame B: supported by doc-revoked AND independently corroborated by chat-kept
    evt_shared_doc = SourceEvent(
        event_id="evt-doc-2",
        revision=1,
        source_type="document",
        source_ref="doc-revoked",
        scope=scope,
        content="We decided to standardize on Python 3.12 across backend services.",
    )
    frame_b = ingest_source_event(evt_shared_doc, store=store).frames[0]

    # Corroborate Frame B from chat-kept (attach provenance to frame_b)
    db.backend.execute(
        "INSERT INTO memory_provenance "
        "(tenant_id, principal_id, namespace, frame_id, source_event_id, "
        "source_type, source_ref, message_id, verbatim_span, created_at) "
        "VALUES (?, ?, ?, ?, NULL, 'chat_turn', 'chat-kept', 'm-9', ?, '2026-03-02T10:00:00')",
        (
            scope.tenant_id,
            scope.principal_id,
            scope.namespace,
            frame_b.id,
            frame_b.content,
        ),
    )
    db.backend.commit()

    # Retract doc-revoked
    ret = retract_source(store, "doc-revoked", scope=scope)
    assert ret.forgotten_frame_ids == [frame_a.id]
    assert ret.retained_frame_ids == [frame_b.id]

    # Frame A is forgotten and unsearchable; Frame B remains active and searchable
    assert store.get_frame(frame_a.id, scope=scope).status == "forgotten"
    assert store.get_frame(frame_b.id, scope=scope).status == "active"
    assert db.fts.search('"Aurora"', 10, scope=scope) == []
    assert db.fts.search('"Python"', 10, scope=scope) == [frame_b.id]
    assert [
        h.frame.id for h in search.search("Python 3.12 backend", scope=scope).hits
    ] == [frame_b.id]

    # Re-ingesting evt_only_doc after retraction is blocked by the 'retracted' tombstone
    replay_doc = ingest_source_event(evt_only_doc, store=store)
    assert replay_doc.frames == []
    assert len(replay_doc.suppressed_hashes) == 1

    db.close()
