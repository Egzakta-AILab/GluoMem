"""Bitemporal KnowledgeGraph store, entity seeding, and symmetric BFS."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from gluomem.config.app_config import KgConfig
from gluomem.kg.normalizer import expand_aliases, normalize
from gluomem.storage.backend import DatabaseBackend
from gluomem.storage.db import MindDB
from gluomem.timeutil import now_utc, to_storage, utc_key

__all__ = ["Entity", "KgSnapshot", "KnowledgeGraph", "Relation"]

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass
class Entity:
    """A typed ``knowledge_entities`` row."""

    id: int
    entity_type: str
    name: str
    properties: dict[str, object]
    valid_from: str
    valid_to: str | None
    recorded_at: str


@dataclass
class Relation:
    """A typed ``knowledge_relations`` row."""

    id: int
    source_id: int
    target_id: int
    relation_type: str
    confidence: float
    properties: dict[str, object]
    valid_from: str
    valid_to: str | None
    recorded_at: str


@dataclass
class KgSnapshot:
    """The reconstructed KG state valid at a point in time (``as_of(T)``)."""

    entities: list[Entity]
    relations: list[Relation]


class KnowledgeGraph:
    """Bitemporal knowledge graph store over a ``MindDB`` backend."""

    def __init__(self, db: MindDB) -> None:
        self._db: MindDB = db
        self._backend: DatabaseBackend = db.backend
        self._conn: sqlite3.Connection = db.conn

    def create_entity(
        self,
        entity_type: str,
        name: str,
        properties: dict[str, object] | None = None,
        *,
        valid_from: str | None = None,
    ) -> Entity:
        """Create a KG entity, stamping ``valid_from`` and ``recorded_at``."""
        props = json.dumps(properties or {}, ensure_ascii=False, separators=(",", ":"))
        now_iso = to_storage(now_utc())
        valid_from_iso = to_storage(valid_from) if valid_from is not None else now_iso
        canonical = name.lower()
        with self._backend.transaction():
            row = self._backend.fetchone(
                "INSERT INTO knowledge_entities "
                "(entity_type, name, canonical_name, properties, "
                "valid_from, valid_to, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, NULL, ?) RETURNING id",
                (entity_type, name, canonical, props, valid_from_iso, now_iso),
            )
        if row is None:
            raise RuntimeError("entity INSERT did not return an id")
        return self.get_entity(int(row["id"]))

    def get_entity(self, eid: int) -> Entity:
        """Return the typed ``Entity`` for ``eid`` (raises ``KeyError`` if missing)."""
        row = self._backend.fetchone(
            "SELECT id, entity_type, name, properties, valid_from, valid_to, "
            "recorded_at FROM knowledge_entities WHERE id = ?",
            (eid,),
        )
        if row is None:
            raise KeyError(f"no knowledge entity with id={eid}")
        return self._row_to_entity(row)

    def retire_entity(self, eid: int, *, valid_to: str | None = None) -> None:
        """Retire an entity by setting ``valid_to`` while preserving the row."""
        self.get_entity(eid)
        valid_to_iso = (
            to_storage(valid_to) if valid_to is not None else to_storage(now_utc())
        )
        with self._backend.transaction():
            self._backend.execute(
                "UPDATE knowledge_entities SET valid_to = ? WHERE id = ?",
                (valid_to_iso, eid),
            )

    def create_relation(
        self,
        source_id: int,
        target_id: int,
        relation_type: str,
        confidence: float = 1.0,
        properties: dict[str, object] | None = None,
        *,
        valid_from: str | None = None,
    ) -> Relation:
        """Create a KG relation, honoring ``UNIQUE(source_id, target_id, relation_type)``."""
        props = json.dumps(properties or {}, ensure_ascii=False, separators=(",", ":"))
        now_iso = to_storage(now_utc())
        valid_from_iso = to_storage(valid_from) if valid_from is not None else now_iso
        with self._backend.transaction():
            row = self._backend.fetchone(
                "INSERT INTO knowledge_relations "
                "(source_id, target_id, relation_type, confidence, properties, "
                " valid_from, valid_to, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, ?) RETURNING id",
                (
                    source_id,
                    target_id,
                    relation_type,
                    float(confidence),
                    props,
                    valid_from_iso,
                    now_iso,
                ),
            )
        if row is None:
            raise RuntimeError("relation INSERT did not return an id")
        return self.get_relation(int(row["id"]))

    def get_relation(self, rid: int) -> Relation:
        """Return the typed ``Relation`` for ``rid`` (raises ``KeyError`` if missing)."""
        row = self._backend.fetchone(
            "SELECT id, source_id, target_id, relation_type, confidence, "
            "properties, valid_from, valid_to, recorded_at "
            "FROM knowledge_relations WHERE id = ?",
            (rid,),
        )
        if row is None:
            raise KeyError(f"no knowledge relation with id={rid}")
        return self._row_to_relation(row)

    def retire_relation(self, rid: int, *, valid_to: str | None = None) -> None:
        """Retire a relation by setting ``valid_to`` while preserving the row."""
        self.get_relation(rid)
        valid_to_iso = (
            to_storage(valid_to) if valid_to is not None else to_storage(now_utc())
        )
        with self._backend.transaction():
            self._backend.execute(
                "UPDATE knowledge_relations SET valid_to = ? WHERE id = ?",
                (valid_to_iso, rid),
            )

    def find_valid_entities(
        self,
        name: str,
        kg_cfg: KgConfig,
        *,
        entity_type: str | None = None,
    ) -> list[Entity]:
        """Return currently-valid entities whose normalized name equals ``name``."""
        forms = expand_aliases([name], kg_cfg)
        if not forms:
            return []
        placeholders = ",".join("?" for _ in forms)
        if entity_type is None:
            rows = self._backend.fetchall(
                f"SELECT id, entity_type, name, properties, valid_from, valid_to, "
                f"recorded_at FROM knowledge_entities "
                f"WHERE valid_to IS NULL AND canonical_name IN ({placeholders}) "
                f"ORDER BY id ASC",
                forms,
            )
        else:
            rows = self._backend.fetchall(
                f"SELECT id, entity_type, name, properties, valid_from, valid_to, "
                f"recorded_at FROM knowledge_entities "
                f"WHERE valid_to IS NULL AND entity_type = ? "
                f"AND canonical_name IN ({placeholders}) "
                f"ORDER BY id ASC",
                (entity_type, *forms),
            )
        return [self._row_to_entity(r) for r in rows]

    def find_valid_relations(
        self,
        source_id: int,
        relation_type: str | None = None,
    ) -> list[Relation]:
        """Return currently-valid relations out of ``source_id``."""
        if relation_type is None:
            rows = self._backend.fetchall(
                "SELECT id, source_id, target_id, relation_type, confidence, "
                "properties, valid_from, valid_to, recorded_at "
                "FROM knowledge_relations "
                "WHERE valid_to IS NULL AND source_id = ? ORDER BY id ASC",
                (source_id,),
            )
        else:
            rows = self._backend.fetchall(
                "SELECT id, source_id, target_id, relation_type, confidence, "
                "properties, valid_from, valid_to, recorded_at "
                "FROM knowledge_relations "
                "WHERE valid_to IS NULL AND source_id = ? AND relation_type = ? "
                "ORDER BY id ASC",
                (source_id, relation_type),
            )
        return [self._row_to_relation(r) for r in rows]

    def find_relation_by_triple(
        self, source_id: int, target_id: int, relation_type: str
    ) -> Relation | None:
        """Return the row for one exact triple regardless of validity, or ``None``."""
        row = self._backend.fetchone(
            "SELECT id, source_id, target_id, relation_type, confidence, "
            "properties, valid_from, valid_to, recorded_at "
            "FROM knowledge_relations "
            "WHERE source_id = ? AND target_id = ? AND relation_type = ?",
            (source_id, target_id, relation_type),
        )
        return None if row is None else self._row_to_relation(row)

    def reopen_relation(
        self, rid: int, *, valid_from: str | None = None
    ) -> Relation:
        """Re-assert a retired relation from ``valid_from`` (default: now)."""
        rel = self.get_relation(rid)
        if rel.valid_to is None:
            return rel
        now_iso = to_storage(now_utc())
        new_from = to_storage(valid_from) if valid_from is not None else now_iso
        assert new_from is not None
        with self._backend.transaction():
            if utc_key(new_from) <= utc_key(rel.valid_to):
                start = (
                    new_from
                    if utc_key(new_from) < utc_key(rel.valid_from)
                    else rel.valid_from
                )
                self._backend.execute(
                    "UPDATE knowledge_relations SET valid_from = ?, valid_to = NULL "
                    "WHERE id = ?",
                    (start, rid),
                )
            else:
                self._backend.execute(
                    "INSERT INTO knowledge_relation_history "
                    "(relation_id, valid_from, valid_to, recorded_at, closed_recorded_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (rid, rel.valid_from, rel.valid_to, rel.recorded_at, now_iso),
                )
                self._backend.execute(
                    "UPDATE knowledge_relations SET valid_from = ?, valid_to = NULL, "
                    "recorded_at = ? WHERE id = ?",
                    (new_from, now_iso, rid),
                )
        return self.get_relation(rid)

    def relation_intervals(self, rid: int) -> list[tuple[str, str | None]]:
        """Return every validity interval of ``rid``, oldest first."""
        rel = self.get_relation(rid)
        rows = self._backend.fetchall(
            "SELECT valid_from, valid_to FROM knowledge_relation_history "
            "WHERE relation_id = ? ORDER BY id ASC",
            (rid,),
        )
        return [
            (str(r["valid_from"]), str(r["valid_to"])) for r in rows
        ] + [(rel.valid_from, rel.valid_to)]

    def link_entity_frame(self, entity_id: int, frame_id: int) -> None:
        """Write a ``kg_entity_frames`` bridge row linking ``entity_id`` to ``frame_id``."""
        with self._backend.transaction():
            self._backend.execute(
                "INSERT INTO kg_entity_frames(entity_id, frame_id) "
                "VALUES (?, ?) ON CONFLICT (entity_id, frame_id) DO NOTHING",
                (entity_id, frame_id),
            )

    def frame_ids_for_entity(self, entity_id: int) -> list[int]:
        """Return the ``frame_id``s linked to ``entity_id``."""
        rows = self._backend.fetchall(
            "SELECT frame_id FROM kg_entity_frames WHERE entity_id = ? "
            "ORDER BY frame_id ASC",
            (entity_id,),
        )
        return [int(r["frame_id"]) for r in rows]

    def entity_ids_for_frame(self, frame_id: int) -> list[int]:
        """Return the ``entity_id``s linked to ``frame_id``."""
        rows = self._backend.fetchall(
            "SELECT entity_id FROM kg_entity_frames WHERE frame_id = ? "
            "ORDER BY entity_id ASC",
            (frame_id,),
        )
        return [int(r["entity_id"]) for r in rows]

    def as_of(self, iso: str) -> KgSnapshot:
        """Reconstruct the KG state valid at time ``T = iso``."""
        t = utc_key(iso)
        ent_rows = self._backend.fetchall(
            "SELECT id, entity_type, name, properties, valid_from, valid_to, "
            "recorded_at FROM knowledge_entities "
            "WHERE REPLACE(valid_from, ' ', 'T') <= ? "
            "AND (valid_to IS NULL OR REPLACE(valid_to, ' ', 'T') > ?) "
            "ORDER BY id ASC",
            (t, t),
        )
        rel_rows = self._backend.fetchall(
            "SELECT id, source_id, target_id, relation_type, confidence, "
            "properties, valid_from, valid_to, recorded_at "
            "FROM knowledge_relations "
            "WHERE REPLACE(valid_from, ' ', 'T') <= ? "
            "AND (valid_to IS NULL OR REPLACE(valid_to, ' ', 'T') > ?) "
            "UNION ALL "
            "SELECT r.id, r.source_id, r.target_id, r.relation_type, r.confidence, "
            "r.properties, h.valid_from, h.valid_to, h.recorded_at "
            "FROM knowledge_relation_history h "
            "JOIN knowledge_relations r ON r.id = h.relation_id "
            "WHERE REPLACE(h.valid_from, ' ', 'T') <= ? "
            "AND REPLACE(h.valid_to, ' ', 'T') > ? "
            "ORDER BY 1 ASC",
            (t, t, t, t),
        )
        entities = [
            e
            for r in ent_rows
            if _interval_holds((e := self._row_to_entity(r)).valid_from, e.valid_to, t)
        ]
        relations = [
            rel
            for r in rel_rows
            if _interval_holds(
                (rel := self._row_to_relation(r)).valid_from, rel.valid_to, t
            )
        ]
        return KgSnapshot(entities=entities, relations=relations)

    def seed_entities(self, query: str, kg_cfg: KgConfig) -> list[int]:
        """Seed KG entities from ``query`` via word-boundary token matching."""
        allow = {t.lower() for t in kg_cfg.tech_allowlist}
        min_len = kg_cfg.seed_min_len

        candidates: set[str] = set()
        for raw in _WORD_RE.findall(query):
            tok = raw.lower()
            if len(tok) >= min_len or tok in allow:
                candidates.add(normalize(raw, kg_cfg))

        if not candidates:
            return []

        forms = expand_aliases(candidates, kg_cfg)
        placeholders = ",".join("?" for _ in forms)
        rows = self._backend.fetchall(
            f"SELECT id FROM knowledge_entities "
            f"WHERE valid_to IS NULL AND canonical_name IN ({placeholders}) "
            f"ORDER BY id ASC",
            forms,
        )
        return [int(r["id"]) for r in rows]

    def bfs_distances(self, seeds: list[int], depth: int) -> dict[int, int]:
        """Symmetric BFS over the currently-valid relation subgraph."""
        if depth < 0 or not seeds:
            return {}

        dist: dict[int, int] = {}
        frontier: list[int] = []
        for s in seeds:
            sid = int(s)
            if sid not in dist:
                dist[sid] = 0
                frontier.append(sid)

        for d in range(depth):
            if not frontier:
                break
            ph = ",".join("?" for _ in frontier)
            rows = self._backend.fetchall(
                f"SELECT source_id, target_id FROM knowledge_relations "
                f"WHERE valid_to IS NULL "
                f"AND (source_id IN ({ph}) OR target_id IN ({ph}))",
                (*frontier, *frontier),
            )
            next_frontier: list[int] = []
            for r in rows:
                for nb in (int(r["source_id"]), int(r["target_id"])):
                    if nb not in dist:
                        dist[nb] = d + 1
                        next_frontier.append(nb)
            frontier = next_frontier
        return dist

    def contextual(
        self,
        query: str,
        kg_cfg: KgConfig,
        *,
        recent_entities: list[int] | None = None,
        depth: int,
    ) -> dict[int, int]:
        """Map each candidate frame to its minimum BFS distance from the query."""
        eff_depth = int(depth)
        if eff_depth < 0:
            return {}

        seeds: set[int] = set(self.seed_entities(query, kg_cfg))
        if recent_entities:
            seeds.update(int(e) for e in recent_entities)
        if not seeds:
            return {}

        distances = self.bfs_distances(sorted(seeds), eff_depth)
        if not distances:
            return {}

        reached_ids = sorted(distances)
        ph = ",".join("?" for _ in reached_ids)
        rows = self._backend.fetchall(
            f"SELECT entity_id, frame_id FROM kg_entity_frames "
            f"WHERE entity_id IN ({ph})",
            reached_ids,
        )
        frame_distance: dict[int, int] = {}
        for row in rows:
            eid = int(row["entity_id"])
            fid = int(row["frame_id"])
            d = distances.get(eid)
            if d is None:
                continue
            prev = frame_distance.get(fid)
            if prev is None or d < prev:
                frame_distance[fid] = d
        return frame_distance

    def _row_to_entity(self, row: Mapping[str, Any] | Sequence[Any]) -> Entity:
        if isinstance(row, Mapping):
            return Entity(
                id=int(row["id"]),
                entity_type=row["entity_type"],
                name=row["name"],
                properties=_load_props(row["properties"]),
                valid_from=row["valid_from"],
                valid_to=row["valid_to"],
                recorded_at=row["recorded_at"],
            )
        v = tuple(row)
        return Entity(
            id=int(v[0]),
            entity_type=v[1],
            name=v[2],
            properties=_load_props(v[3]),
            valid_from=v[4],
            valid_to=v[5],
            recorded_at=v[6],
        )

    def _row_to_relation(self, row: Mapping[str, Any] | Sequence[Any]) -> Relation:
        if isinstance(row, Mapping):
            return Relation(
                id=int(row["id"]),
                source_id=int(row["source_id"]),
                target_id=int(row["target_id"]),
                relation_type=row["relation_type"],
                confidence=float(row["confidence"]),
                properties=_load_props(row["properties"]),
                valid_from=row["valid_from"],
                valid_to=row["valid_to"],
                recorded_at=row["recorded_at"],
            )
        v = tuple(row)
        return Relation(
            id=int(v[0]),
            source_id=int(v[1]),
            target_id=int(v[2]),
            relation_type=v[3],
            confidence=float(v[4]),
            properties=_load_props(v[5]),
            valid_from=v[6],
            valid_to=v[7],
            recorded_at=v[8],
        )


def _interval_holds(valid_from: str, valid_to: str | None, t: str) -> bool:
    return utc_key(valid_from) <= t and (valid_to is None or utc_key(valid_to) > t)


def _load_props(raw: str | None) -> dict[str, object]:
    if raw is None:
        return {}
    try:
        loaded = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}
