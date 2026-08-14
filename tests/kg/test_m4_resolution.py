"""KG resolution lookup tests — find_valid_entities + find_valid_relations.

These power the write-path cognify step: whole-name alias-aware entity
resolution (create-if-missing), idempotent duplicate-triple recovery, and
conflicting-edge retirement.
"""

from __future__ import annotations

from pathlib import Path

from gluomem.config.app_config import KgConfig
from gluomem.kg.graph import KnowledgeGraph
from gluomem.storage.db import MindDB


def _kg(tmp_path: Path, *, dim: int = 384) -> tuple[MindDB, KnowledgeGraph]:
    db = MindDB(tmp_path / "test.mind", dim=dim)
    return db, KnowledgeGraph(db)


def test_find_valid_entities_exact_and_alias(tmp_path: Path) -> None:
    _, kg = _kg(tmp_path)
    cfg = KgConfig()
    e = kg.create_entity("technology", "kubernetes")
    # Exact (case-insensitive via normalizer).
    found = kg.find_valid_entities("Kubernetes", cfg)
    assert [x.id for x in found] == [e.id]
    # Alias group: k8s normalizes to the same canonical name.
    found_alias = kg.find_valid_entities("k8s", cfg)
    assert [x.id for x in found_alias] == [e.id]


def test_find_valid_entities_multiword_no_tokenization(tmp_path: Path) -> None:
    """Whole-name equality — multi-word names match verbatim, never by token."""
    _, kg = _kg(tmp_path)
    cfg = KgConfig()
    e = kg.create_entity("person", "Alice Johnson")
    assert [x.id for x in kg.find_valid_entities("alice johnson", cfg)] == [e.id]
    # A single shared token is NOT a match.
    assert kg.find_valid_entities("Alice", cfg) == []


def test_find_valid_entities_type_filter_and_retired(tmp_path: Path) -> None:
    _, kg = _kg(tmp_path)
    cfg = KgConfig()
    person = kg.create_entity("person", "mercury")
    kg.create_entity("planet", "mercury")
    only_person = kg.find_valid_entities("mercury", cfg, entity_type="person")
    assert [x.id for x in only_person] == [person.id]
    assert len(kg.find_valid_entities("mercury", cfg)) == 2
    # Retired entities are excluded.
    kg.retire_entity(person.id)
    assert kg.find_valid_entities("mercury", cfg, entity_type="person") == []


def test_find_valid_relations_filter_and_retired(tmp_path: Path) -> None:
    _, kg = _kg(tmp_path)
    a = kg.create_entity("person", "alice")
    boston = kg.create_entity("city", "boston")
    seattle = kg.create_entity("city", "seattle")
    lives = kg.create_relation(a.id, boston.id, "LIVES_IN")
    kg.create_relation(a.id, seattle.id, "VISITED")

    typed = kg.find_valid_relations(a.id, "LIVES_IN")
    assert [r.id for r in typed] == [lives.id]
    assert len(kg.find_valid_relations(a.id)) == 2
    # Retired edges drop out (retire-not-delete keeps the row).
    kg.retire_relation(lives.id)
    assert kg.find_valid_relations(a.id, "LIVES_IN") == []
    assert kg.get_relation(lives.id).valid_to is not None


def test_find_relation_by_triple_ignores_validity(tmp_path: Path) -> None:
    """The triple lookup sees retired rows (the triple is the UNIQUE key)."""
    _, kg = _kg(tmp_path)
    a = kg.create_entity("person", "alice")
    boston = kg.create_entity("city", "boston")
    lives = kg.create_relation(a.id, boston.id, "LIVES_IN")
    kg.retire_relation(lives.id)
    found = kg.find_relation_by_triple(a.id, boston.id, "LIVES_IN")
    assert found is not None and found.id == lives.id
    assert found.valid_to is not None
    assert kg.find_relation_by_triple(a.id, boston.id, "WORKS_AT") is None


def test_reopen_relation_clears_valid_to(tmp_path: Path) -> None:
    """Re-asserting a retired fact makes the edge valid again (row kept)."""
    _, kg = _kg(tmp_path)
    a = kg.create_entity("person", "alice")
    boston = kg.create_entity("city", "boston")
    lives = kg.create_relation(a.id, boston.id, "LIVES_IN")
    kg.retire_relation(lives.id)
    reopened = kg.reopen_relation(lives.id)
    assert reopened.id == lives.id
    assert reopened.valid_to is None
    assert [r.id for r in kg.find_valid_relations(a.id, "LIVES_IN")] == [lives.id]
