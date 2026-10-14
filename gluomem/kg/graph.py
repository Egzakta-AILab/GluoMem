"""Bitemporal KnowledgeGraph store, entity seeding, and symmetric BFS."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from gluomem.config.app_config import KgConfig
from gluomem.kg.normalizer import expand_aliases, normalize
from gluomem.storage.backend import DatabaseBackend
from gluomem.storage.db import MindDB
from gluomem.storage.scope import DEFAULT_SCOPE, MemoryScope, eligible_sql, scope_sql
from gluomem.timeutil import now_utc, now_utc_iso, to_storage, utc_key

__all__ = ["Entity", "KgSnapshot", "KnowledgeGraph", "Relation"]

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

_ENTITY_COLS = (
    "id, entity_type, name, properties, valid_from, valid_to, recorded_at, "
    "tenant_id, principal_id, namespace"
)
_REL_COLS = (
    "id, source_id, target_id, relation_type, confidence, properties, "
    "valid_from, valid_to, recorded_at, tenant_id, principal_id, namespace"
)


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
    tenant_id: str = "local"
    principal_id: str = "default"
    namespace: str = "personal"

    @property
    def scope(self) -> MemoryScope:
        return MemoryScope(self.tenant_id, self.principal_id, self.namespace)


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
    tenant_id: str = "local"
    principal_id: str = "default"
    namespace: str = "personal"

    @property
    def scope(self) -> MemoryScope:
        return MemoryScope(self.tenant_id, self.principal_id, self.namespace)


@dataclass
class KgSnapshot:
    """The reconstructed KG state valid at a point in time (``as_of(T)``)."""

    entities: list[Entity]
    relations: list[Relation]


class KnowledgeGraph:
    """Bitemporal knowledge graph store over a ``MindDB`` backend."""

    def __init__(self, db: MindDB, *, scope: MemoryScope = DEFAULT_SCOPE) -> None:
        self._db: MindDB = db
        self._backend: DatabaseBackend = db.backend
        self._conn: sqlite3.Connection = db.conn
        self._scope: MemoryScope = scope

    @property
    def scope(self) -> MemoryScope:
        return self._scope

    def _eff_scope(self, scope: MemoryScope | None = None) -> MemoryScope:
        return scope if scope is not None else self._scope

    def create_entity(
        self,
        entity_type: str,
        name: str,
        properties: dict[str, object] | None = None,
        *,
        valid_from: str | None = None,
        scope: MemoryScope | None = None,
    ) -> Entity:
        """Create a KG entity, stamping ``valid_from`` and ``recorded_at``."""
        eff_scope = self._eff_scope(scope)
        props = json.dumps(properties or {}, ensure_ascii=False, separators=(",", ":"))
        now_iso = to_storage(now_utc())
        valid_from_iso = to_storage(valid_from) if valid_from is not None else now_iso
        canonical = name.lower()
        with self._backend.transaction():
            row = self._backend.fetchone(
                "INSERT INTO knowledge_entities "
                "(tenant_id, principal_id, namespace, entity_type, name, "
                "canonical_name, properties, valid_from, valid_to, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?) RETURNING id",
                (
                    *eff_scope.as_tuple(),
                    entity_type,
                    name,
                    canonical,
                    props,
                    valid_from_iso,
                    now_iso,
                ),
            )
        if row is None:
            raise RuntimeError("entity INSERT did not return an id")
        return self.get_entity(int(row["id"]), scope=eff_scope)

    def get_entity(self, eid: int, *, scope: MemoryScope | None = None) -> Entity:
        """Return the typed ``Entity`` for ``eid`` in ``scope`` (raises ``KeyError`` if missing)."""
        eff_scope = self._eff_scope(scope)
        row = self._backend.fetchone(
            f"SELECT {_ENTITY_COLS} FROM knowledge_entities WHERE id = ? AND {scope_sql()}",
            (
                eid,
                *eff_scope.as_tuple(),
            ),
        )
        if row is None:
            raise KeyError(f"no knowledge entity with id={eid}")
        return self._row_to_entity(row)

    def retire_entity(
        self,
        eid: int,
        *,
        valid_to: str | None = None,
        scope: MemoryScope | None = None,
    ) -> None:
        """Retire an entity by setting ``valid_to`` while preserving the row."""
        eff_scope = self._eff_scope(scope)
        self.get_entity(eid, scope=eff_scope)
        valid_to_iso = to_storage(valid_to) if valid_to is not None else to_storage(now_utc())
        with self._backend.transaction():
            self._backend.execute(
                f"UPDATE knowledge_entities SET valid_to = ? WHERE id = ? AND {scope_sql()}",
                (
                    valid_to_iso,
                    eid,
                    *eff_scope.as_tuple(),
                ),
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
        scope: MemoryScope | None = None,
    ) -> Relation:
        """Create a KG relation in ``scope`` with uniqueness per scope."""
        eff_scope = self._eff_scope(scope)
        self.get_entity(source_id, scope=eff_scope)
        self.get_entity(target_id, scope=eff_scope)
        props = json.dumps(properties or {}, ensure_ascii=False, separators=(",", ":"))
        now_iso = to_storage(now_utc())
        valid_from_iso = to_storage(valid_from) if valid_from is not None else now_iso
        with self._backend.transaction():
            row = self._backend.fetchone(
                "INSERT INTO knowledge_relations "
                "(tenant_id, principal_id, namespace, source_id, target_id, "
                " relation_type, confidence, properties, valid_from, valid_to, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?) RETURNING id",
                (
                    *eff_scope.as_tuple(),
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
        return self.get_relation(int(row["id"]), scope=eff_scope)

    def get_relation(self, rid: int, *, scope: MemoryScope | None = None) -> Relation:
        """Return the typed ``Relation`` for ``rid`` in ``scope``."""
        eff_scope = self._eff_scope(scope)
        row = self._backend.fetchone(
            f"SELECT {_REL_COLS} FROM knowledge_relations WHERE id = ? AND {scope_sql()}",
            (
                rid,
                *eff_scope.as_tuple(),
            ),
        )
        if row is None:
            raise KeyError(f"no knowledge relation with id={rid}")
        return self._row_to_relation(row)

    def retire_relation(
        self,
        rid: int,
        *,
        valid_to: str | None = None,
        scope: MemoryScope | None = None,
    ) -> None:
        """Retire a relation by setting ``valid_to`` while preserving the row."""
        eff_scope = self._eff_scope(scope)
        self.get_relation(rid, scope=eff_scope)
        valid_to_iso = to_storage(valid_to) if valid_to is not None else to_storage(now_utc())
        with self._backend.transaction():
            self._backend.execute(
                f"UPDATE knowledge_relations SET valid_to = ? WHERE id = ? AND {scope_sql()}",
                (
                    valid_to_iso,
                    rid,
                    *eff_scope.as_tuple(),
                ),
            )

    def find_valid_entities(
        self,
        name: str,
        kg_cfg: KgConfig,
        *,
        entity_type: str | None = None,
        scope: MemoryScope | None = None,
    ) -> list[Entity]:
        """Return currently-valid entities in ``scope`` whose normalized name equals ``name``."""
        eff_scope = self._eff_scope(scope)
        forms = expand_aliases([name], kg_cfg)
        if not forms:
            return []
        placeholders = ",".join("?" for _ in forms)
        if entity_type is None:
            rows = self._backend.fetchall(
                f"SELECT {_ENTITY_COLS} FROM knowledge_entities "
                f"WHERE {scope_sql()} "
                f"AND valid_to IS NULL AND canonical_name IN ({placeholders}) "
                f"ORDER BY id ASC",
                (
                    *eff_scope.as_tuple(),
                    *forms,
                ),
            )
        else:
            rows = self._backend.fetchall(
                f"SELECT {_ENTITY_COLS} FROM knowledge_entities "
                f"WHERE {scope_sql()} "
                f"AND valid_to IS NULL AND entity_type = ? "
                f"AND canonical_name IN ({placeholders}) "
                f"ORDER BY id ASC",
                (
                    *eff_scope.as_tuple(),
                    entity_type,
                    *forms,
                ),
            )
        return [self._row_to_entity(r) for r in rows]

    def find_valid_relations(
        self,
        source_id: int,
        relation_type: str | None = None,
        *,
        scope: MemoryScope | None = None,
    ) -> list[Relation]:
        """Return currently-valid relations out of ``source_id`` in ``scope``."""
        eff_scope = self._eff_scope(scope)
        if relation_type is None:
            rows = self._backend.fetchall(
                f"SELECT {_REL_COLS} FROM knowledge_relations "
                f"WHERE {scope_sql()} "
                "AND valid_to IS NULL AND source_id = ? ORDER BY id ASC",
                (
                    *eff_scope.as_tuple(),
                    source_id,
                ),
            )
        else:
            rows = self._backend.fetchall(
                f"SELECT {_REL_COLS} FROM knowledge_relations "
                f"WHERE {scope_sql()} "
                "AND valid_to IS NULL AND source_id = ? AND relation_type = ? "
                "ORDER BY id ASC",
                (
                    *eff_scope.as_tuple(),
                    source_id,
                    relation_type,
                ),
            )
        return [self._row_to_relation(r) for r in rows]

    def find_relation_by_triple(
        self,
        source_id: int,
        target_id: int,
        relation_type: str,
        *,
        scope: MemoryScope | None = None,
    ) -> Relation | None:
        """Return the row for one exact triple in ``scope`` regardless of validity, or ``None``."""
        eff_scope = self._eff_scope(scope)
        row = self._backend.fetchone(
            f"SELECT {_REL_COLS} FROM knowledge_relations "
            f"WHERE {scope_sql()} "
            "AND source_id = ? AND target_id = ? AND relation_type = ?",
            (
                *eff_scope.as_tuple(),
                source_id,
                target_id,
                relation_type,
            ),
        )
        return None if row is None else self._row_to_relation(row)

    def reopen_relation(
        self,
        rid: int,
        *,
        valid_from: str | None = None,
        scope: MemoryScope | None = None,
    ) -> Relation:
        """Re-assert a retired relation from ``valid_from`` (default: now)."""
        eff_scope = self._eff_scope(scope)
        rel = self.get_relation(rid, scope=eff_scope)
        if rel.valid_to is None:
            return rel
        now_iso = to_storage(now_utc())
        new_from = to_storage(valid_from) if valid_from is not None else now_iso
        assert new_from is not None
        with self._backend.transaction():
            if utc_key(new_from) <= utc_key(rel.valid_to):
                start = new_from if utc_key(new_from) < utc_key(rel.valid_from) else rel.valid_from
                self._backend.execute(
                    "UPDATE knowledge_relations SET valid_from = ?, valid_to = NULL "
                    f"WHERE id = ? AND {scope_sql()}",
                    (
                        start,
                        rid,
                        *eff_scope.as_tuple(),
                    ),
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
                    "recorded_at = ? "
                    f"WHERE id = ? AND {scope_sql()}",
                    (
                        new_from,
                        now_iso,
                        rid,
                        *eff_scope.as_tuple(),
                    ),
                )
        return self.get_relation(rid, scope=eff_scope)

    def relation_intervals(
        self, rid: int, *, scope: MemoryScope | None = None
    ) -> list[tuple[str, str | None]]:
        """Return every validity interval of ``rid`` in ``scope``, oldest first."""
        eff_scope = self._eff_scope(scope)
        rel = self.get_relation(rid, scope=eff_scope)
        rows = self._backend.fetchall(
            "SELECT valid_from, valid_to FROM knowledge_relation_history "
            "WHERE relation_id = ? ORDER BY id ASC",
            (rid,),
        )
        return [(str(r["valid_from"]), str(r["valid_to"])) for r in rows] + [
            (rel.valid_from, rel.valid_to)
        ]

    def link_entity_frame(
        self,
        entity_id: int,
        frame_id: int,
        *,
        scope: MemoryScope | None = None,
    ) -> None:
        """Write a ``kg_entity_frames`` bridge row in ``scope``."""
        eff_scope = self._eff_scope(scope)
        self.get_entity(entity_id, scope=eff_scope)
        frame_row = self._backend.fetchone(
            f"SELECT id FROM memory_frames WHERE id = ? AND {scope_sql()}",
            (
                frame_id,
                *eff_scope.as_tuple(),
            ),
        )
        if frame_row is None:
            raise KeyError(f"no memory_frame with id={frame_id} in scope")
        with self._backend.transaction():
            self._backend.execute(
                "INSERT INTO kg_entity_frames(entity_id, frame_id) "
                f"SELECT ?, id FROM memory_frames WHERE id = ? AND {scope_sql()} AND "
                f"{eligible_sql()} "
                "ON CONFLICT (entity_id, frame_id) DO NOTHING",
                (entity_id, frame_id, *eff_scope.as_tuple(), utc_key(now_utc_iso())),
            )

    def link_relation_frame(
        self, relation_id: int, frame_id: int, *, scope: MemoryScope | None = None
    ) -> None:
        """Attach relation evidence only to a recall-eligible frame in its scope."""
        scope = self._eff_scope(scope)
        self.get_relation(relation_id, scope=scope)
        with self._backend.transaction():
            self._backend.execute(
                "INSERT INTO kg_relation_frames(relation_id, frame_id) "
                f"SELECT ?, id FROM memory_frames WHERE id = ? AND {scope_sql()} AND "
                f"{eligible_sql()} "
                "ON CONFLICT (relation_id, frame_id) DO NOTHING",
                (relation_id, frame_id, *scope.as_tuple(), utc_key(now_utc_iso())),
            )

    def frame_ids_for_entity(
        self, entity_id: int, *, scope: MemoryScope | None = None
    ) -> list[int]:
        """Return the ``frame_id``s linked to ``entity_id`` in ``scope``."""
        eff_scope = self._eff_scope(scope)
        rows = self._backend.fetchall(
            "SELECT kef.frame_id AS frame_id "
            "FROM kg_entity_frames kef "
            "JOIN knowledge_entities e ON e.id = kef.entity_id "
            "JOIN memory_frames m ON m.id = kef.frame_id "
            "WHERE kef.entity_id = ? "
            f"AND {scope_sql('e')} "
            f"AND {scope_sql('m')} "
            "ORDER BY kef.frame_id ASC",
            (
                entity_id,
                *eff_scope.as_tuple(),
                *eff_scope.as_tuple(),
            ),
        )
        return [int(r["frame_id"]) for r in rows]

    def entity_ids_for_frame(self, frame_id: int, *, scope: MemoryScope | None = None) -> list[int]:
        """Return the ``entity_id``s linked to ``frame_id`` in ``scope``."""
        eff_scope = self._eff_scope(scope)
        rows = self._backend.fetchall(
            "SELECT kef.entity_id AS entity_id "
            "FROM kg_entity_frames kef "
            "JOIN knowledge_entities e ON e.id = kef.entity_id "
            "JOIN memory_frames m ON m.id = kef.frame_id "
            "WHERE kef.frame_id = ? "
            f"AND {scope_sql('e')} "
            f"AND {scope_sql('m')} "
            "ORDER BY kef.entity_id ASC",
            (
                frame_id,
                *eff_scope.as_tuple(),
                *eff_scope.as_tuple(),
            ),
        )
        return [int(r["entity_id"]) for r in rows]

    def as_of(self, iso: str, *, scope: MemoryScope | None = None) -> KgSnapshot:
        """Reconstruct the KG state in ``scope`` valid at time ``T = iso``."""
        eff_scope = self._eff_scope(scope)
        t = utc_key(iso)
        ent_rows = self._backend.fetchall(
            f"SELECT {_ENTITY_COLS} FROM knowledge_entities "
            f"WHERE {scope_sql()} "
            "AND REPLACE(valid_from, ' ', 'T') <= ? "
            "AND (valid_to IS NULL OR REPLACE(valid_to, ' ', 'T') > ?) "
            "ORDER BY id ASC",
            (
                *eff_scope.as_tuple(),
                t,
                t,
            ),
        )
        rel_rows = self._backend.fetchall(
            "SELECT id, source_id, target_id, relation_type, confidence, "
            "properties, valid_from, valid_to, recorded_at, "
            "tenant_id, principal_id, namespace "
            "FROM knowledge_relations "
            f"WHERE {scope_sql()} "
            "AND REPLACE(valid_from, ' ', 'T') <= ? "
            "AND (valid_to IS NULL OR REPLACE(valid_to, ' ', 'T') > ?) "
            "UNION ALL "
            "SELECT r.id, r.source_id, r.target_id, r.relation_type, r.confidence, "
            "r.properties, h.valid_from, h.valid_to, h.recorded_at, "
            "r.tenant_id, r.principal_id, r.namespace "
            "FROM knowledge_relation_history h "
            "JOIN knowledge_relations r ON r.id = h.relation_id "
            f"WHERE {scope_sql('r')} "
            "AND REPLACE(h.valid_from, ' ', 'T') <= ? "
            "AND REPLACE(h.valid_to, ' ', 'T') > ? "
            "ORDER BY 1 ASC",
            (
                *eff_scope.as_tuple(),
                t,
                t,
                *eff_scope.as_tuple(),
                t,
                t,
            ),
        )
        entities = [
            e
            for r in ent_rows
            if _interval_holds((e := self._row_to_entity(r)).valid_from, e.valid_to, t)
        ]
        relations = [
            rel
            for r in rel_rows
            if _interval_holds((rel := self._row_to_relation(r)).valid_from, rel.valid_to, t)
        ]
        return KgSnapshot(entities=entities, relations=relations)

    def seed_entities(
        self,
        query: str,
        kg_cfg: KgConfig,
        *,
        scope: MemoryScope | None = None,
    ) -> list[int]:
        """Seed KG entities in ``scope`` from ``query`` via word-boundary token matching."""
        eff_scope = self._eff_scope(scope)
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
            f"WHERE {scope_sql()} "
            f"AND valid_to IS NULL AND canonical_name IN ({placeholders}) "
            f"ORDER BY id ASC",
            (
                *eff_scope.as_tuple(),
                *forms,
            ),
        )
        return [int(r["id"]) for r in rows]

    def bfs_distances(
        self,
        seeds: list[int],
        depth: int,
        *,
        scope: MemoryScope | None = None,
    ) -> dict[int, int]:
        """Symmetric BFS over the currently-valid relation subgraph in ``scope``."""
        if depth < 0 or not seeds:
            return {}
        eff_scope = self._eff_scope(scope)
        seed_ph = ",".join("?" for _ in seeds)
        valid_seed_rows = self._backend.fetchall(
            f"SELECT id FROM knowledge_entities "
            f"WHERE {scope_sql()} "
            f"AND valid_to IS NULL AND id IN ({seed_ph}) "
            f"ORDER BY id ASC",
            (
                *eff_scope.as_tuple(),
                *[int(s) for s in seeds],
            ),
        )
        scoped_seeds = [int(r["id"]) for r in valid_seed_rows]
        if not scoped_seeds:
            return {}

        dist: dict[int, int] = {}
        frontier: list[int] = []
        for sid in scoped_seeds:
            if sid not in dist:
                dist[sid] = 0
                frontier.append(sid)

        for d in range(depth):
            if not frontier:
                break
            ph = ",".join("?" for _ in frontier)
            rows = self._backend.fetchall(
                f"SELECT source_id, target_id FROM knowledge_relations "
                f"WHERE {scope_sql()} "
                f"AND valid_to IS NULL "
                f"AND (source_id IN ({ph}) OR target_id IN ({ph}))",
                (
                    *eff_scope.as_tuple(),
                    *frontier,
                    *frontier,
                ),
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
        scope: MemoryScope | None = None,
    ) -> dict[int, int]:
        """Map each candidate frame in ``scope`` to its minimum BFS distance from the query."""
        eff_depth = int(depth)
        if eff_depth < 0:
            return {}
        eff_scope = self._eff_scope(scope)

        seeds: set[int] = set(self.seed_entities(query, kg_cfg, scope=eff_scope))
        if recent_entities:
            seeds.update(int(e) for e in recent_entities)
        if not seeds:
            return {}

        distances = self.bfs_distances(sorted(seeds), eff_depth, scope=eff_scope)
        if not distances:
            return {}

        reached_ids = sorted(distances)
        ph = ",".join("?" for _ in reached_ids)
        rows = self._backend.fetchall(
            f"SELECT kef.entity_id AS entity_id, kef.frame_id AS frame_id "
            f"FROM kg_entity_frames kef "
            f"JOIN memory_frames m ON m.id = kef.frame_id "
            f"WHERE kef.entity_id IN ({ph}) "
            f"AND {scope_sql('m')} AND {eligible_sql('m')}",
            (
                *reached_ids,
                *eff_scope.as_tuple(),
                utc_key(now_utc_iso()),
            ),
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

    def _row_to_entity(self, row: Mapping[str, Any]) -> Entity:
        values = {col: row[col] for col in _ENTITY_COLS.split(", ")}
        values["properties"] = _load_props(values["properties"])
        return Entity(**values)

    def _row_to_relation(self, row: Mapping[str, Any]) -> Relation:
        values = {col: row[col] for col in _REL_COLS.split(", ")}
        values["properties"] = _load_props(values["properties"])
        return Relation(**values)


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
