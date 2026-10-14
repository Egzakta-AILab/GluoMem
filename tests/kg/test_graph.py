"""KnowledgeGraph bitemporal store, BFS, seeding, normalizer, resolution, and error metric tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from gluomem.config.app_config import AppConfig, KgConfig, load_config
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import Entity, KnowledgeGraph, Relation
from gluomem.kg.normalizer import normalize, predicate_family
from gluomem.search.hybrid import HybridSearch
from gluomem.storage.db import MindDB


def _holds(kg: KnowledgeGraph, rid: int, iso: str) -> bool:
    return rid in {r.id for r in kg.as_of(iso).relations}


def test_entity_relation_roundtrip_bridge_and_retire(
    mind_db: MindDB, kg: KnowledgeGraph, frame_store: FrameStore
) -> None:
    e1 = kg.create_entity("tool", "git")
    e2 = kg.create_entity("tool", "svn")
    assert isinstance(e1, Entity) and e1.valid_to is None
    rel = kg.create_relation(e1.id, e2.id, "related", confidence=0.8)
    assert isinstance(rel, Relation) and rel.confidence == pytest.approx(0.8)
    with pytest.raises(sqlite3.IntegrityError):
        kg.create_relation(e1.id, e2.id, "related")

    frame = frame_store.create_i_frame("gop-1", "a memory frame")
    kg.link_entity_frame(e1.id, frame.id)
    assert kg.frame_ids_for_entity(e1.id) == [frame.id]
    assert kg.entity_ids_for_frame(frame.id) == [e1.id]

    kg.retire_entity(e1.id, valid_to="2025-03-01T00:00:00")
    assert kg.get_entity(e1.id).valid_to == "2025-03-01T00:00:00"
    kg.retire_relation(rel.id)
    assert kg.get_relation(rel.id).valid_to is not None


def test_as_of_and_reassertion_intervals(tmp_path: Path) -> None:
    db = MindDB(tmp_path / "test.mind", dim=384)
    kg = KnowledgeGraph(db)
    a = kg.create_entity("person", "alice", valid_from="2025-01-01T00:00:00").id
    b = kg.create_entity("city", "boston", valid_from="2025-01-01T00:00:00").id
    rel = kg.create_relation(a, b, "LIVES_IN", valid_from="2025-01-01T00:00:00")
    assert _holds(kg, rel.id, "2025-06-01T00:00:00")

    kg.retire_relation(rel.id, valid_to="2025-03-01T00:00:00")
    assert not _holds(kg, rel.id, "2025-03-01T00:00:00")
    assert not _holds(kg, rel.id, "2025-06-01T00:00:00")

    back = kg.reopen_relation(rel.id, valid_from="2025-06-01T00:00:00")
    assert back.id == rel.id and back.valid_to is None
    assert kg.relation_intervals(rel.id) == [
        ("2025-01-01T00:00:00", "2025-03-01T00:00:00"),
        ("2025-06-01T00:00:00", None),
    ]
    expected = {
        "2024-12-31": False,
        "2025-02-01": True,
        "2025-03-01": False,
        "2025-04-15": False,
        "2025-06-01": True,
        "2025-07-01": True,
    }
    assert {d: _holds(kg, rel.id, d) for d in expected} == expected
    db.close()

    db2 = MindDB(tmp_path / "test.mind", dim=384)
    kg2 = KnowledgeGraph(db2)
    assert {d: _holds(kg2, rel.id, d) for d in expected} == expected
    db2.close()


def test_overlapping_reassertion_and_offset_utc(kg: KnowledgeGraph) -> None:
    a = kg.create_entity("person", "alice").id
    b = kg.create_entity("city", "boston").id
    rel = kg.create_relation(a, b, "LIVES_IN", valid_from="2025-01-01T00:00:00")
    assert kg.reopen_relation(rel.id, valid_from="2025-06-01T00:00:00") == rel

    kg.retire_relation(rel.id, valid_to="2025-03-01T00:00:00")
    kg.reopen_relation(rel.id, valid_from="2025-02-15T00:00:00")
    assert kg.relation_intervals(rel.id) == [("2025-01-01T00:00:00", None)]

    c = kg.create_entity("city", "seattle").id
    rel_tz = kg.create_relation(a, c, "VISITED", valid_from="2025-01-01T02:00:00+02:00")
    assert rel_tz.valid_from == "2025-01-01T00:00:00"
    kg.retire_relation(rel_tz.id, valid_to="2025-03-01T00:00:00Z")
    assert _holds(kg, rel_tz.id, "2025-01-01T01:00:00+01:00")
    assert not _holds(kg, rel_tz.id, "2025-03-01T01:00:00+01:00")


def test_bfs_symmetric_depth_and_retired(kg: KnowledgeGraph) -> None:
    a, b, c, d, e = [kg.create_entity("t", n) for n in ("A", "B", "C", "D", "E")]
    kg.create_relation(a.id, b.id, "r")
    rel_ca = kg.create_relation(c.id, a.id, "r")
    kg.create_relation(b.id, d.id, "r")
    kg.create_relation(d.id, e.id, "r")

    dist = kg.bfs_distances([a.id], depth=2)
    assert dist == {a.id: 0, b.id: 1, c.id: 1, d.id: 2}
    assert e.id not in dist

    kg.retire_relation(rel_ca.id)
    assert c.id not in kg.bfs_distances([a.id], depth=2)


def test_seeding_and_normalizer(kg: KnowledgeGraph, cfg: AppConfig) -> None:
    assert normalize("Postgres", cfg.kg) == "postgresql"
    assert normalize("PG", cfg.kg) == "postgresql"
    assert normalize("K8S", cfg.kg) == "kubernetes"
    assert normalize("GO", cfg.kg) == "go"
    assert normalize("Mountain", cfg.kg) == "mountain"

    fam_cfg = KgConfig(predicate_families=[["LIVES_IN", "MOVED_TO", "RESIDES_IN"]])
    assert predicate_family("lives_in", fam_cfg) == ("LIVES_IN", "MOVED_TO", "RESIDES_IN")
    assert predicate_family("WORKS_AT", fam_cfg) == ("WORKS_AT",)

    ai = kg.create_entity("concept", "ai")
    pg = kg.create_entity("tool", "postgresql")
    assert ai.id not in kg.seed_entities("I climbed a mountain", cfg.kg)
    assert ai.id in kg.seed_entities("I use AI daily", cfg.kg)
    assert pg.id in kg.seed_entities("we deploy pg in production", cfg.kg)


def test_entity_and_relation_resolution_helpers(kg: KnowledgeGraph) -> None:
    cfg = KgConfig()
    k8s = kg.create_entity("technology", "kubernetes")
    assert [x.id for x in kg.find_valid_entities("k8s", cfg)] == [k8s.id]

    alice = kg.create_entity("person", "Alice Johnson")
    assert [x.id for x in kg.find_valid_entities("alice johnson", cfg)] == [alice.id]
    assert kg.find_valid_entities("Alice", cfg) == []

    p_merc = kg.create_entity("person", "mercury")
    kg.create_entity("planet", "mercury")
    assert [x.id for x in kg.find_valid_entities("mercury", cfg, entity_type="person")] == [
        p_merc.id
    ]
    kg.retire_entity(p_merc.id)
    assert kg.find_valid_entities("mercury", cfg, entity_type="person") == []

    boston = kg.create_entity("city", "boston")
    lives = kg.create_relation(alice.id, boston.id, "LIVES_IN")
    assert [r.id for r in kg.find_valid_relations(alice.id, "LIVES_IN")] == [lives.id]
    kg.retire_relation(lives.id)
    assert kg.find_valid_relations(alice.id, "LIVES_IN") == []
    found = kg.find_relation_by_triple(alice.id, boston.id, "LIVES_IN")
    assert found is not None and found.id == lives.id


def test_kg_lane_error_metric_in_hybrid_search(
    mind_db: MindDB, frame_store: FrameStore, mock_embedder: MockEmbedder
) -> None:
    class _RaisingKG(KnowledgeGraph):
        def contextual(self, *args: object, **kwargs: object) -> dict[int, int]:
            raise RuntimeError("simulated KG failure")

    cfg = load_config().with_overrides(
        {"search.rerank.enabled": False, "search.abstain.enabled": False}
    )
    frame_store.create_i_frame("gop-1", "alpha beta")

    bad_search = HybridSearch(
        mind_db, frame_store, cfg, embedder=mock_embedder, kg=_RaisingKG(mind_db)
    )
    res_bad = bad_search.search("alpha beta", limit=10)
    assert res_bad.kg_lane_errors >= 1 and res_bad.hits

    ok_search = HybridSearch(
        mind_db, frame_store, cfg, embedder=mock_embedder, kg=KnowledgeGraph(mind_db)
    )
    res_ok = ok_search.search("alpha beta", limit=10)
    assert res_ok.kg_lane_errors == 0 and res_ok.hits
