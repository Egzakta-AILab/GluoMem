"""Symmetric BFS distance tests.

``KnowledgeGraph.bfs_distances(seeds, depth)`` performs a SYMMETRIC traversal
following BOTH outgoing and incoming relations (in+out), distance
direction-agnostic, respecting ``depth`` and only over currently-valid
relations.
"""

from __future__ import annotations

from pathlib import Path

from gluomem.kg.graph import KnowledgeGraph
from gluomem.storage.db import MindDB


def _kg(tmp_path: Path, *, dim: int = 384) -> tuple[MindDB, KnowledgeGraph]:
    db = MindDB(tmp_path / "test.mind", dim=dim)
    return db, KnowledgeGraph(db)


# --- symmetric traversal ----------------------------------------------------


def test_bfs_symmetric_in_and_out(tmp_path: Path) -> None:
    """Edges A->B and C->A; bfs([A],3) returns B and C at 1."""
    db, kg = _kg(tmp_path)

    a = kg.create_entity("t", "A")
    b = kg.create_entity("t", "B")
    c = kg.create_entity("t", "C")

    # A -> B (outgoing from A); C -> A (incoming to A).
    kg.create_relation(a.id, b.id, "r")
    kg.create_relation(c.id, a.id, "r")

    dist = kg.bfs_distances([a.id], depth=3)
    # Symmetric: both the outgoing neighbor B and the incoming neighbor C are
    # reachable at distance 1 (direction-agnostic).
    assert dist[b.id] == 1
    assert dist[c.id] == 1
    # The seed itself is at distance 0.
    assert dist[a.id] == 0
    db.close()


# --- depth limit ------------------------------------------------------------


def test_bfs_respects_depth(tmp_path: Path) -> None:
    """Chain A->B->C->D->E; bfs([A],3) returns B,C,D at 1,2,3
    and excludes E (distance 4)."""
    db, kg = _kg(tmp_path)

    a = kg.create_entity("t", "A")
    b = kg.create_entity("t", "B")
    c = kg.create_entity("t", "C")
    d = kg.create_entity("t", "D")
    e = kg.create_entity("t", "E")

    kg.create_relation(a.id, b.id, "r")
    kg.create_relation(b.id, c.id, "r")
    kg.create_relation(c.id, d.id, "r")
    kg.create_relation(d.id, e.id, "r")

    dist = kg.bfs_distances([a.id], depth=3)
    assert dist[a.id] == 0
    assert dist[b.id] == 1
    assert dist[c.id] == 2
    assert dist[d.id] == 3
    # E is at distance 4, beyond depth=3 -> excluded.
    assert e.id not in dist
    db.close()


# --- retired relations are not traversed ------------------------------------


def test_bfs_skips_retired_relations(tmp_path: Path) -> None:
    """A retired relation is not traversed by BFS (only valid relations)."""
    db, kg = _kg(tmp_path)

    a = kg.create_entity("t", "A")
    b = kg.create_entity("t", "B")
    rel = kg.create_relation(a.id, b.id, "r")
    kg.retire_relation(rel.id)

    dist = kg.bfs_distances([a.id], depth=3)
    assert dist == {a.id: 0}
    db.close()
