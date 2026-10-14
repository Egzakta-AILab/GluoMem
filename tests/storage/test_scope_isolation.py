"""Tests for MemoryScope multi-tenant isolation across SQLite, Vec, FTS, Frames, KG, and Search."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from gluomem.config.app_config import AppConfig, ChunkingConfig, load_config
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.sessions import end_session, get_or_create_session
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.search.hybrid import HybridSearch
from gluomem.storage.db import MindDB
from gluomem.storage.scope import DEFAULT_SCOPE, MemoryScope, resolve_scope
from gluomem.write.dedup import SemanticDeduper
from gluomem.write.gate import InjectionScanner, admit


def _cfg() -> AppConfig:
    return load_config().with_overrides(
        {
            "components.allow_fake": True,
            "embedding.model": "mock",
            "models.mock.dim": 16,
            "storage.vector_dim": 16,
        }
    )


def test_memory_scope_validation_and_defaults() -> None:
    assert DEFAULT_SCOPE == MemoryScope("local", "default", "personal")
    assert resolve_scope(None) == DEFAULT_SCOPE
    custom = MemoryScope("tenant_1", "user_42", "workspace")
    assert resolve_scope(custom) == custom
    assert custom.as_tuple() == ("tenant_1", "user_42", "workspace")
    assert custom.as_dict() == {
        "tenant_id": "tenant_1",
        "principal_id": "user_42",
        "namespace": "personal" if False else "workspace",
    }

    with pytest.raises(ValueError, match="tenant_id"):
        MemoryScope("", "user", "personal")
    with pytest.raises(ValueError, match="principal_id"):
        MemoryScope("tenant", "   ", "personal")
    with pytest.raises(ValueError, match="namespace"):
        MemoryScope("tenant", "user", "")


def test_two_scopes_in_same_db_zero_cross_scope_leakage_vector_fts_hybrid(
    tmp_path: Path,
) -> None:
    cfg = _cfg()
    db = MindDB(tmp_path / "shared.mind", dim=16)
    emb = MockEmbedder(dim=16)
    # Force chunking so both chunk_vec and frame_vec are exercised
    chunk_cfg = ChunkingConfig(chunk_size=40, chunk_overlap=5, whole_frame_threshold=20)

    scope_a = MemoryScope("tenant_alpha", "alice", "personal")
    scope_b = MemoryScope("tenant_beta", "bob", "workspace")

    store_a = FrameStore(db, embedder=emb, chunking=chunk_cfg, scope=scope_a)
    store_b = FrameStore(db, embedder=emb, chunking=chunk_cfg, scope=scope_b)

    shared_text = (
        "Project Hyperion launch code is ZETA-9941 and deployment runs on cluster-eu-west."
    )
    frame_a = store_a.create_i_frame("gop-shared", shared_text)

    # Flood scope_b with 25 identical-text frames so post-LIMIT filtering would fail at k=1
    b_ids: list[int] = []
    for i in range(25):
        fb = store_b.create_i_frame(f"gop-b-{i}", shared_text)
        b_ids.append(fb.id)

    # 1. Pre-filtered frame_vec KNN at k=1 returns scope_a's frame, never scope_b's
    qvec = emb.embed(shared_text)
    vec_hits_a = db.frame_vec.search(qvec, k=1, scope=scope_a)
    assert [fid for fid, _ in vec_hits_a] == [frame_a.id]

    vec_hits_b = db.frame_vec.search(qvec, k=5, scope=scope_b)
    assert len(vec_hits_b) == 5
    assert all(fid in b_ids for fid, _ in vec_hits_b)

    # 2. Pre-filtered chunk_vec KNN at k=1 returns only chunks belonging to scope_a
    chunk_hits_a = db.chunk_vec.search(qvec, k=1, scope=scope_a)
    assert len(chunk_hits_a) == 1
    resolved_a = store_a.resolve_chunk_frames([chunk_hits_a[0][0]], scope=scope_a)
    assert list(resolved_a.values()) == [frame_a.id]
    assert store_a.resolve_chunk_frames([chunk_hits_a[0][0]], scope=scope_b) == {}

    # 3. FTS5 search and LIKE search at k=1 respect scope before LIMIT
    fts_hits_a = db.fts.search("Hyperion", limit=1, scope=scope_a)
    assert fts_hits_a == [frame_a.id]
    like_hits_a = db.fts.search_like("ZETA-9941", limit=1, scope=scope_a)
    assert like_hits_a == [frame_a.id]

    # 4. HybridSearch returns only scope_a hits when queried under scope_a
    hs = HybridSearch(db, store_a, cfg, embedder=emb, scope=scope_a)
    res_a = hs.search("Hyperion ZETA-9941", limit=5)
    assert [h.frame.id for h in res_a.hits] == [frame_a.id]

    res_b = hs.search("Hyperion ZETA-9941", limit=5, scope=scope_b)
    assert len(res_b.hits) == 5
    assert all(h.frame.id in b_ids for h in res_b.hits)
    assert all(h.frame.scope == scope_b for h in res_b.hits)
    db.close()


def test_same_gop_id_across_two_scopes_isolated(tmp_path: Path) -> None:
    cfg = _cfg()
    db = MindDB(tmp_path / "gop.mind", dim=16)
    emb = MockEmbedder(dim=16)

    scope_a = MemoryScope("acme", "user_1", "personal")
    scope_b = MemoryScope("acme", "user_2", "personal")

    # Same gop_id in two different scopes
    get_or_create_session(db, "conv-1:facts", project_id="proj-a", scope=scope_a, commit=True)
    get_or_create_session(db, "conv-1:facts", project_id="proj-b", scope=scope_b, commit=True)
    end_session(db, "conv-1:facts", summary="summary-a", scope=scope_a)
    row_a = db.backend.fetchone(
        "SELECT project_id, status, summary FROM sessions "
        "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? AND gop_id = ?",
        (scope_a.tenant_id, scope_a.principal_id, scope_a.namespace, "conv-1:facts"),
    )
    row_b = db.backend.fetchone(
        "SELECT project_id, status, summary FROM sessions "
        "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? AND gop_id = ?",
        (scope_b.tenant_id, scope_b.principal_id, scope_b.namespace, "conv-1:facts"),
    )
    assert row_a is not None and row_a["project_id"] == "proj-a" and row_a["status"] == "ended"
    assert row_b is not None and row_b["project_id"] == "proj-b" and row_b["status"] == "active"

    store = FrameStore(db, embedder=emb, scope=scope_a)
    fa_i = store.create_i_frame("conv-1:facts", "Alice prefers Rust.", scope=scope_a)
    fa_p = store.create_p_frame("conv-1:facts", "Alice also uses Python.", fa_i.id, scope=scope_a)

    fb_i = store.create_i_frame("conv-1:facts", "Bob prefers Go.", scope=scope_b)
    assert fb_i.t == 0  # Independent sequence numbering in scope_b!
    assert fa_p.t == 1

    # Cross-scope get_frame raises KeyError
    with pytest.raises(KeyError):
        store.get_frame(fa_i.id, scope=scope_b)
    with pytest.raises(KeyError):
        store.get_frame(fb_i.id, scope=scope_a)

    # Cross-scope P-frame creation is rejected
    with pytest.raises(KeyError):
        store.create_p_frame("conv-1:facts", "Cross-scope delta", fa_i.id, scope=scope_b)

    # reconstruct_state is isolated per scope
    base_a, deltas_a = store.reconstruct_state("conv-1:facts", scope=scope_a)
    assert base_a is not None and base_a.id == fa_i.id
    assert [d.id for d in deltas_a] == [fa_p.id]

    base_b, deltas_b = store.reconstruct_state("conv-1:facts", scope=scope_b)
    assert base_b is not None and base_b.id == fb_i.id
    assert deltas_b == []

    # SemanticDeduper and admit do not deduplicate across scopes
    scanner = InjectionScanner(cfg.safety)
    deduper = SemanticDeduper(emb, threshold=0.99)
    r1 = admit(
        "Shared observation text",
        gop="shared-gop",
        scanner=scanner,
        store=store,
        kg=None,
        cfg=cfg,
        deduper=deduper,
        scope=scope_a,
    )
    r2 = admit(
        "Shared observation text",
        gop="shared-gop",
        scanner=scanner,
        store=store,
        kg=None,
        cfg=cfg,
        deduper=deduper,
        scope=scope_b,
    )
    assert r1.admitted and not r1.deduped and not r1.semantic_deduped
    assert r2.admitted and not r2.deduped and not r2.semantic_deduped
    assert r1.frame is not None and r2.frame is not None
    assert r1.frame.id != r2.frame.id
    db.close()


def test_kg_entities_relations_and_bfs_isolated_across_scopes(tmp_path: Path) -> None:
    cfg = _cfg()
    db = MindDB(tmp_path / "kg.mind", dim=16)
    emb = MockEmbedder(dim=16)

    scope_a = MemoryScope("org_a", "u1", "personal")
    scope_b = MemoryScope("org_b", "u1", "personal")

    store = FrameStore(db, embedder=emb, scope=scope_a)
    kg = KnowledgeGraph(db, scope=scope_a)

    fa = store.create_i_frame("gop", "Alice works at Acme", scope=scope_a)
    fb = store.create_i_frame("gop", "Alice works at Globex", scope=scope_b)

    ea1 = kg.create_entity("person", "Alice", scope=scope_a)
    ea2 = kg.create_entity("company", "Acme", scope=scope_a)
    ra = kg.create_relation(ea1.id, ea2.id, "works_at", scope=scope_a)
    kg.link_entity_frame(ea2.id, fa.id, scope=scope_a)

    eb1 = kg.create_entity("person", "Alice", scope=scope_b)
    eb2 = kg.create_entity("company", "Globex", scope=scope_b)
    rb = kg.create_relation(eb1.id, eb2.id, "works_at", scope=scope_b)
    kg.link_entity_frame(eb2.id, fb.id, scope=scope_b)

    # Cross-scope relation creation or entity-frame link fails
    with pytest.raises(KeyError):
        kg.create_relation(ea1.id, eb2.id, "works_at", scope=scope_a)
    with pytest.raises(KeyError):
        kg.link_entity_frame(ea1.id, fb.id, scope=scope_a)

    # Entity lookup and contextual BFS stay strictly in-scope
    assert [e.id for e in kg.find_valid_entities("Alice", cfg.kg, scope=scope_a)] == [ea1.id]
    assert [e.id for e in kg.find_valid_entities("Alice", cfg.kg, scope=scope_b)] == [eb1.id]

    ctx_a = kg.contextual("Where does Alice work?", cfg.kg, depth=2, scope=scope_a)
    assert fa.id in ctx_a
    assert fb.id not in ctx_a

    ctx_b = kg.contextual("Where does Alice work?", cfg.kg, depth=2, scope=scope_b)
    assert fb.id in ctx_b
    assert fa.id not in ctx_b

    snap_a = kg.as_of("2030-01-01T00:00:00Z", scope=scope_a)
    assert [r.id for r in snap_a.relations] == [ra.id]
    snap_b = kg.as_of("2030-01-01T00:00:00Z", scope=scope_b)
    assert [r.id for r in snap_b.relations] == [rb.id]
    db.close()


def test_lifecycle_status_and_expires_at_filtered_from_recall(tmp_path: Path) -> None:
    cfg = _cfg()
    db = MindDB(tmp_path / "lifecycle.mind", dim=16)
    emb = MockEmbedder(dim=16)
    fixed_now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)

    store = FrameStore(db, embedder=emb)
    f_active = store.create_i_frame(
        "gop",
        "Deployment target is kubernetes-prod",
        expires_at="2026-06-01T00:00:00",
    )
    f_expired = store.create_i_frame(
        "gop",
        "Deployment target is kubernetes-staging",
        expires_at="2026-04-01T00:00:00",
    )
    f_superseded = store.create_i_frame(
        "gop",
        "Deployment target is kubernetes-legacy",
    )
    store.set_status(f_superseded.id, "superseded", superseded_by=f_active.id)

    hs = HybridSearch(db, store, cfg, embedder=emb, clock=lambda: fixed_now)
    res = hs.search("Deployment target kubernetes", limit=10)
    hit_ids = [h.frame.id for h in res.hits]
    assert hit_ids == [f_active.id]
    assert f_expired.id not in hit_ids
    assert f_superseded.id not in hit_ids
    db.close()
