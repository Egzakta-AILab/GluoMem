"""KnowledgeGraph bitemporal store tests.

Drives the public ``KnowledgeGraph`` surface over a ``tmp_path`` ``*.mind``
file via ``MindDB``. The schema (knowledge_entities / knowledge_relations /
kg_entity_frames) is created by ``schema.sql``; these tests assert the
bitemporal read/write semantics: create, link, retire-not-delete, as_of(T)
reconstruction.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from gluomem.frames.store import FrameStore
from gluomem.kg.graph import Entity, KnowledgeGraph, Relation
from gluomem.storage.db import MindDB


def _kg(tmp_path: Path, *, dim: int = 384) -> tuple[MindDB, KnowledgeGraph]:
    """Open a MindDB at a temp path and return (db, KnowledgeGraph) over it."""
    db = MindDB(tmp_path / "test.mind", dim=dim)
    return db, KnowledgeGraph(db)


# --- entity/relation round-trip ---------------------------------------------


def test_entity_and_relation_roundtrip(tmp_path: Path) -> None:
    """Create two entities + a relation; read back type/name/relation_type
    intact, honoring the triple-UNIQUE constraint."""
    db, kg = _kg(tmp_path)

    e1 = kg.create_entity("tool", "git")
    e2 = kg.create_entity("tool", "svn")
    assert isinstance(e1, Entity)
    assert e1.entity_type == "tool"
    assert e1.name == "git"
    assert e1.valid_to is None
    assert e1.recorded_at is not None

    rel = kg.create_relation(e1.id, e2.id, "related", confidence=0.8)
    assert isinstance(rel, Relation)
    assert rel.source_id == e1.id
    assert rel.target_id == e2.id
    assert rel.relation_type == "related"
    assert rel.confidence == pytest.approx(0.8)
    assert rel.valid_to is None

    # Read back via accessors — typed rows with type/name/relation_type intact.
    back_e = kg.get_entity(e1.id)
    assert back_e.entity_type == "tool"
    assert back_e.name == "git"
    back_r = kg.get_relation(rel.id)
    assert back_r.relation_type == "related"
    assert back_r.source_id == e1.id
    assert back_r.target_id == e2.id

    # The triple-UNIQUE constraint: a second identical triple raises.
    with pytest.raises(sqlite3.IntegrityError):
        kg.create_relation(e1.id, e2.id, "related")
    db.close()


# --- entity-frame bridge ----------------------------------------------------


def test_entity_frame_bridge_both_directions(tmp_path: Path) -> None:
    """Link entity<->frame; query both directions."""
    db, kg = _kg(tmp_path)
    store = FrameStore(db)

    e = kg.create_entity("concept", "memory")
    frame = store.create_i_frame("gop-1", "a memory frame")

    kg.link_entity_frame(e.id, frame.id)

    # By entity -> frame_ids.
    frame_ids = kg.frame_ids_for_entity(e.id)
    assert frame_ids == [frame.id]

    # By frame -> entity_ids.
    entity_ids = kg.entity_ids_for_frame(frame.id)
    assert entity_ids == [e.id]
    db.close()


# --- retire-not-delete ------------------------------------------------------


def test_retire_entity_sets_valid_to_and_keeps_row(tmp_path: Path) -> None:
    """retire_entity sets valid_to, row still present, exactly one row for
    the entity."""
    db, kg = _kg(tmp_path)
    e = kg.create_entity("tool", "git")

    kg.retire_entity(e.id)

    # Exactly one row for the entity remains (no DELETE).
    n = db.conn.execute(
        "SELECT COUNT(*) FROM knowledge_entities WHERE id=?", (e.id,)
    ).fetchone()[0]
    assert n == 1

    back = kg.get_entity(e.id)
    assert back.valid_to is not None, "retire must set valid_to (non-null)"
    db.close()


def test_retire_relation_sets_valid_to_and_keeps_row(tmp_path: Path) -> None:
    """Retiring a relation sets valid_to while the row remains (retire-not-delete)."""
    db, kg = _kg(tmp_path)
    a = kg.create_entity("tool", "git")
    b = kg.create_entity("tool", "svn")
    rel = kg.create_relation(a.id, b.id, "related")

    kg.retire_relation(rel.id)

    n = db.conn.execute(
        "SELECT COUNT(*) FROM knowledge_relations WHERE id=?", (rel.id,)
    ).fetchone()[0]
    assert n == 1
    back = kg.get_relation(rel.id)
    assert back.valid_to is not None
    db.close()


def test_retire_entity_with_supplied_iso(tmp_path: Path) -> None:
    """retire_entity(eid, valid_to=iso) stamps the supplied value verbatim."""
    db, kg = _kg(tmp_path)
    e = kg.create_entity("tool", "git")
    kg.retire_entity(e.id, valid_to="2025-03-01T00:00:00")
    back = kg.get_entity(e.id)
    assert back.valid_to == "2025-03-01T00:00:00"
    db.close()


# --- as_of: present ---------------------------------------------------------


def test_as_of_present(tmp_path: Path) -> None:
    """A relation with valid_from 2025-01-01 (not retired) is present in
    as_of('2025-06-01')."""
    db, kg = _kg(tmp_path)
    a = kg.create_entity("tool", "git", valid_from="2025-01-01T00:00:00")
    b = kg.create_entity("tool", "svn", valid_from="2025-01-01T00:00:00")
    # Backdate the relation's valid_from to 2025-01-01 (still open: valid_to NULL).
    rel = kg.create_relation(
        a.id, b.id, "related", valid_from="2025-01-01T00:00:00"
    )
    assert rel.valid_from == "2025-01-01T00:00:00"

    state = kg.as_of("2025-06-01T00:00:00")
    rel_ids = [r.id for r in state.relations]
    assert rel.id in rel_ids, (
        "a relation whose interval contains T=2025-06-01 must be present in as_of"
    )
    # The entities backing it are also present (backdated to 2025-01-01).
    ent_ids = {e.id for e in state.entities}
    assert a.id in ent_ids
    assert b.id in ent_ids
    db.close()


# --- as_of: absent ----------------------------------------------------------


def test_as_of_absent(tmp_path: Path) -> None:
    """A relation retired with valid_to 2025-03-01 is absent from
    as_of('2025-06-01')."""
    db, kg = _kg(tmp_path)
    a = kg.create_entity("tool", "git")
    b = kg.create_entity("tool", "svn")
    rel = kg.create_relation(
        a.id, b.id, "related", valid_from="2025-01-01T00:00:00"
    )
    # Retire before T: valid_to = 2025-03-01.
    kg.retire_relation(rel.id, valid_to="2025-03-01T00:00:00")

    state = kg.as_of("2025-06-01T00:00:00")
    rel_ids = [r.id for r in state.relations]
    assert rel.id not in rel_ids, (
        "a relation retired before T=2025-06-01 must be absent from as_of"
    )
    db.close()


def test_as_of_boundary_strict(tmp_path: Path) -> None:
    """as_of uses strict > for valid_to: a relation with valid_to == T is present
    (valid_to > T is False only when strictly greater required — interval is
    [valid_from, valid_to) so valid_to == T means still NOT present at T)."""
    db, kg = _kg(tmp_path)
    a = kg.create_entity("tool", "git")
    b = kg.create_entity("tool", "svn")
    rel = kg.create_relation(
        a.id, b.id, "related", valid_from="2025-01-01T00:00:00"
    )
    kg.retire_relation(rel.id, valid_to="2025-06-01T00:00:00")

    # valid_to == T -> absent at T (interval is half-open [from, to)).
    state = kg.as_of("2025-06-01T00:00:00")
    assert rel.id not in [r.id for r in state.relations]

    # valid_to just after T -> present at T.
    kg2 = KnowledgeGraph(db)
    c = kg2.create_entity("tool", "hg")
    rel2 = kg2.create_relation(
        a.id, c.id, "related", valid_from="2025-01-01T00:00:00"
    )
    kg2.retire_relation(rel2.id, valid_to="2025-06-01T00:00:01")
    state2 = kg2.as_of("2025-06-01T00:00:00")
    assert rel2.id in [r.id for r in state2.relations]
    db.close()


# --- re-assertion after a retirement gap -------------------------------------


def _ab(kg: KnowledgeGraph) -> tuple[int, int]:
    return kg.create_entity("person", "alice").id, kg.create_entity("city", "boston").id


def _holds(kg: KnowledgeGraph, rid: int, iso: str) -> bool:
    return rid in {r.id for r in kg.as_of(iso).relations}


def test_reassert_after_gap_preserves_the_gap(tmp_path: Path) -> None:
    """valid [Jan, Mar), retired, re-asserted from Jun: April is false,
    February and July are true, and the answers survive a reopen."""
    db, kg = _kg(tmp_path)
    a, b = _ab(kg)
    rel = kg.create_relation(a, b, "LIVES_IN", valid_from="2025-01-01T00:00:00")
    kg.retire_relation(rel.id, valid_to="2025-03-01T00:00:00")
    back = kg.reopen_relation(rel.id, valid_from="2025-06-01T00:00:00")

    assert back.id == rel.id and back.valid_to is None
    assert back.valid_from == "2025-06-01T00:00:00"
    assert kg.relation_intervals(rel.id) == [
        ("2025-01-01T00:00:00", "2025-03-01T00:00:00"),
        ("2025-06-01T00:00:00", None),
    ]
    expected = {
        "2024-12-31": False,
        "2025-02-01": True,
        "2025-03-01": False,  # half-open: valid_to excluded
        "2025-04-15": False,  # inside the gap
        "2025-06-01": True,
        "2025-07-01": True,
    }
    assert {d: _holds(kg, rel.id, d) for d in expected} == expected
    snap = kg.as_of("2025-02-01")
    hist = next(r for r in snap.relations if r.id == rel.id)
    assert (hist.valid_from, hist.valid_to) == ("2025-01-01T00:00:00", "2025-03-01T00:00:00")
    db.close()

    db = MindDB(tmp_path / "test.mind", dim=384)
    kg = KnowledgeGraph(db)
    assert {d: _holds(kg, rel.id, d) for d in expected} == expected
    assert kg.find_valid_relations(a, "LIVES_IN")[0].id == rel.id
    db.close()


def test_reassert_overlapping_retirement_continues_interval(tmp_path: Path) -> None:
    """A re-assertion valid from before the old valid_to means the fact never
    stopped holding: no history row, the interval just continues."""
    db, kg = _kg(tmp_path)
    a, b = _ab(kg)
    rel = kg.create_relation(a, b, "LIVES_IN", valid_from="2025-01-01T00:00:00")
    kg.retire_relation(rel.id, valid_to="2025-03-01T00:00:00")
    kg.reopen_relation(rel.id, valid_from="2025-02-15T00:00:00")
    assert kg.relation_intervals(rel.id) == [("2025-01-01T00:00:00", None)]
    assert _holds(kg, rel.id, "2025-04-15")
    db.close()


def test_reopen_valid_relation_is_noop(tmp_path: Path) -> None:
    db, kg = _kg(tmp_path)
    a, b = _ab(kg)
    rel = kg.create_relation(a, b, "LIVES_IN", valid_from="2025-01-01T00:00:00")
    assert kg.reopen_relation(rel.id, valid_from="2025-06-01T00:00:00") == rel
    assert kg.relation_intervals(rel.id) == [("2025-01-01T00:00:00", None)]
    db.close()


def test_offset_aware_validity_is_stored_and_compared_as_utc(tmp_path: Path) -> None:
    db, kg = _kg(tmp_path)
    a, b = _ab(kg)
    rel = kg.create_relation(a, b, "LIVES_IN", valid_from="2025-01-01T02:00:00+02:00")
    assert rel.valid_from == "2025-01-01T00:00:00"
    kg.retire_relation(rel.id, valid_to="2025-03-01T00:00:00Z")
    assert kg.get_relation(rel.id).valid_to == "2025-03-01T00:00:00"
    assert _holds(kg, rel.id, "2025-01-01T01:00:00+01:00")  # = 00:00Z, inclusive start
    assert not _holds(kg, rel.id, "2024-12-31T23:59:59Z")
    assert not _holds(kg, rel.id, "2025-03-01T01:00:00+01:00")  # = valid_to
    db.close()


def test_legacy_file_without_history_table_upgrades_and_reads(tmp_path: Path) -> None:
    """A 0.0.1 file (no history table; a relation reopened the old way, so
    its gap was already lost) opens, reads as before, and records gaps for
    new re-assertions."""
    db, kg = _kg(tmp_path)
    a, b = _ab(kg)
    rel = kg.create_relation(a, b, "LIVES_IN", valid_from="2025-01-01T00:00:00")
    db.conn.execute("DROP TABLE knowledge_relation_history")
    db.conn.commit()
    db.close()

    db = MindDB(tmp_path / "test.mind", dim=384)
    kg = KnowledgeGraph(db)
    assert _holds(kg, rel.id, "2025-02-01")
    kg.retire_relation(rel.id, valid_to="2025-03-01T00:00:00")
    kg.reopen_relation(rel.id, valid_from="2025-06-01T00:00:00")
    assert not _holds(kg, rel.id, "2025-04-15")
    db.close()
