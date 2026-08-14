"""gluomem.kg.graph — bitemporal KnowledgeGraph store.

A bitemporal knowledge graph over the ``knowledge_entities``,
``knowledge_relations``, and ``kg_entity_frames`` tables created by
``schema.sql``.

Bitemporal model
----------------
- Every entity/relation row stamps ``valid_from`` (defaults to ``now``),
  ``valid_to`` (NULL until retired), and ``recorded_at`` (defaults to ``now``).
- **Retire-not-delete**: ``retire_entity`` / ``retire_relation`` SET
  ``valid_to`` (now or a supplied ISO) and KEEP the row. No row is ever
  ``DELETE``d; history is preserved.
- ``as_of(T)`` reconstructs the state valid at T: rows where
  ``valid_from <= T AND (valid_to IS NULL OR valid_to > T)`` — i.e. the
  half-open validity interval ``[valid_from, valid_to)`` contains T.

The triple-UNIQUE constraint ``UNIQUE(source_id, target_id, relation_type)``
is honored: a duplicate triple insert raises ``sqlite3.IntegrityError`` (the
schema enforces it; this module surfaces it unchanged).

The ``kg_entity_frames`` bridge is written via ``link_entity_frame`` and is
queryable both directions: ``frame_ids_for_entity(eid)`` and
``entity_ids_for_frame(frame_id)``.

Bitemporal CRUD, the bridge, ``as_of`` reconstruction, seeding, BFS, and the
contextual signal live here; the alias normalizer and the relative-date
resolver live in ``normalizer.py`` and ``temporal.py``.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from gluomem.config.app_config import KgConfig
from gluomem.kg.normalizer import normalize
from gluomem.storage.db import MindDB

__all__ = ["Entity", "Relation", "KgSnapshot", "KnowledgeGraph"]


# Word-boundary tokenizer for seeding. ``\b\w+`` over unicode captures
# standalone tokens only — the substring ``ai`` inside ``mountain`` is NOT a
# token, so it never seeds the ``ai`` entity.
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


# ---------------------------------------------------------------------------
# Typed rows.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# KnowledgeGraph.
# ---------------------------------------------------------------------------


class KnowledgeGraph:
    """Bitemporal KG store over a ``MindDB`` connection.

    Parameters
    ----------
    db:
        An open ``MindDB`` (WAL, FK, sqlite-vec loaded, schema applied). The
        graph reuses its connection for all reads/writes.
    """

    def __init__(self, db: MindDB) -> None:
        self._db: MindDB = db
        self._conn: sqlite3.Connection = db.conn
        # Register the pure ISO normalizer as a SQL function so as_of's
        # half-open interval predicate (valid_from <= T AND (valid_to IS NULL
        # OR valid_to > T)) compares equal-format ISO strings server-side.
        # create_function with the same name replaces any prior registration,
        # so this is safe even if the connection is shared across KG instances.
        self._conn.create_function(
            "_gluomem_norm_iso", 1, _normalize_iso, deterministic=True
        )

    # --- entities ----------------------------------------------------------

    def create_entity(
        self,
        entity_type: str,
        name: str,
        properties: dict[str, object] | None = None,
        *,
        valid_from: str | None = None,
    ) -> Entity:
        """Create a KG entity, stamping valid_from=now (or supplied), valid_to=NULL.

        ``properties`` is stored as JSON in the ``properties`` column (defaults
        to ``{}``). ``valid_from`` defaults to SQLite ``datetime('now')``; an
        explicit ISO string backdates the validity start. ``recorded_at`` is
        always ``now`` (the recording time is the ingest time, not the world
        time).
        """
        props = json.dumps(properties or {}, ensure_ascii=False, separators=(",", ":"))
        if valid_from is None:
            cur = self._conn.execute(
                "INSERT INTO knowledge_entities "
                "(entity_type, name, properties, valid_from, valid_to, recorded_at) "
                "VALUES (?, ?, ?, datetime('now'), NULL, datetime('now'))",
                (entity_type, name, props),
            )
        else:
            cur = self._conn.execute(
                "INSERT INTO knowledge_entities "
                "(entity_type, name, properties, valid_from, valid_to, recorded_at) "
                "VALUES (?, ?, ?, ?, NULL, datetime('now'))",
                (entity_type, name, props, valid_from),
            )
        self._conn.commit()
        eid = cur.lastrowid
        if eid is None:  # pragma: no cover — INSERT always sets lastrowid
            raise RuntimeError("entity INSERT did not return a lastrowid")
        return self.get_entity(int(eid))

    def get_entity(self, eid: int) -> Entity:
        """Return the typed ``Entity`` for ``eid``.

        Raises ``KeyError`` if no entity exists with that id.
        """
        row = self._conn.execute(
            "SELECT id, entity_type, name, properties, valid_from, valid_to, "
            "recorded_at FROM knowledge_entities WHERE id=?",
            (eid,),
        ).fetchone()
        if row is None:
            raise KeyError(f"no knowledge entity with id={eid}")
        return self._row_to_entity(row)

    def retire_entity(self, eid: int, *, valid_to: str | None = None) -> None:
        """Retire an entity: SET ``valid_to`` (now or supplied), KEEP the row.

        Retire-not-delete: the row is NEVER removed; it is closed by setting
        ``valid_to`` so ``as_of(T > valid_to)`` excludes it. Raises ``KeyError``
        if the entity does not exist. Idempotent re-retire overwrites
        ``valid_to`` with the new value.
        """
        # Ensure the entity exists (raise KeyError otherwise).
        self.get_entity(eid)
        if valid_to is None:
            self._conn.execute(
                "UPDATE knowledge_entities SET valid_to=datetime('now') WHERE id=?",
                (eid,),
            )
        else:
            self._conn.execute(
                "UPDATE knowledge_entities SET valid_to=? WHERE id=?",
                (valid_to, eid),
            )
        self._conn.commit()

    # --- relations ---------------------------------------------------------

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
        """Create a KG relation, honoring ``UNIQUE(source_id, target_id, relation_type)``.

        ``valid_from`` defaults to ``now``; an explicit ISO string backdates the
        validity start. ``recorded_at`` is always ``now``. A duplicate triple
        raises ``sqlite3.IntegrityError`` (the schema enforces the UNIQUE).
        """
        props = json.dumps(properties or {}, ensure_ascii=False, separators=(",", ":"))
        if valid_from is None:
            cur = self._conn.execute(
                "INSERT INTO knowledge_relations "
                "(source_id, target_id, relation_type, confidence, properties, "
                " valid_from, valid_to, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, datetime('now'), NULL, datetime('now'))",
                (source_id, target_id, relation_type, float(confidence), props),
            )
        else:
            cur = self._conn.execute(
                "INSERT INTO knowledge_relations "
                "(source_id, target_id, relation_type, confidence, properties, "
                " valid_from, valid_to, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, datetime('now'))",
                (source_id, target_id, relation_type, float(confidence), props, valid_from),
            )
        self._conn.commit()
        rid = cur.lastrowid
        if rid is None:  # pragma: no cover — INSERT always sets lastrowid
            raise RuntimeError("relation INSERT did not return a lastrowid")
        return self.get_relation(int(rid))

    def get_relation(self, rid: int) -> Relation:
        """Return the typed ``Relation`` for ``rid``.

        Raises ``KeyError`` if no relation exists with that id.
        """
        row = self._conn.execute(
            "SELECT id, source_id, target_id, relation_type, confidence, "
            "properties, valid_from, valid_to, recorded_at "
            "FROM knowledge_relations WHERE id=?",
            (rid,),
        ).fetchone()
        if row is None:
            raise KeyError(f"no knowledge relation with id={rid}")
        return self._row_to_relation(row)

    def retire_relation(self, rid: int, *, valid_to: str | None = None) -> None:
        """Retire a relation: SET ``valid_to`` (now or supplied), KEEP the row.

        Retire-not-delete. Raises ``KeyError`` if the relation does not exist.
        """
        self.get_relation(rid)
        if valid_to is None:
            self._conn.execute(
                "UPDATE knowledge_relations SET valid_to=datetime('now') WHERE id=?",
                (rid,),
            )
        else:
            self._conn.execute(
                "UPDATE knowledge_relations SET valid_to=? WHERE id=?",
                (valid_to, rid),
            )
        self._conn.commit()

    # --- resolution lookups ------------------------------------------------

    def find_valid_entities(
        self,
        name: str,
        kg_cfg: KgConfig,
        *,
        entity_type: str | None = None,
    ) -> list[Entity]:
        """Return currently-valid entities whose normalized name equals ``name``.

        Whole-name equality through the alias normalizer (``k8s`` finds
        ``kubernetes``; multi-word names match verbatim after normalization) —
        unlike token-based ``seed_entities`` this never token-splits. Optional
        ``entity_type`` narrows the match. Ordered by id ascending.
        """
        target = normalize(name, kg_cfg)
        if entity_type is None:
            rows = self._conn.execute(
                "SELECT id, entity_type, name, properties, valid_from, valid_to, "
                "recorded_at FROM knowledge_entities WHERE valid_to IS NULL "
                "ORDER BY id ASC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, entity_type, name, properties, valid_from, valid_to, "
                "recorded_at FROM knowledge_entities "
                "WHERE valid_to IS NULL AND entity_type=? ORDER BY id ASC",
                (entity_type,),
            ).fetchall()
        return [
            self._row_to_entity(r)
            for r in rows
            if normalize(str(r[2]), kg_cfg) == target
        ]

    def find_valid_relations(
        self,
        source_id: int,
        relation_type: str | None = None,
    ) -> list[Relation]:
        """Return currently-valid relations out of ``source_id``.

        Optional ``relation_type`` narrows to one edge type — the lookup the
        write path uses both to recover idempotently from a duplicate-triple
        ``IntegrityError`` and to find edges to retire on contradiction.
        """
        if relation_type is None:
            rows = self._conn.execute(
                "SELECT id, source_id, target_id, relation_type, confidence, "
                "properties, valid_from, valid_to, recorded_at "
                "FROM knowledge_relations "
                "WHERE valid_to IS NULL AND source_id=? ORDER BY id ASC",
                (source_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, source_id, target_id, relation_type, confidence, "
                "properties, valid_from, valid_to, recorded_at "
                "FROM knowledge_relations "
                "WHERE valid_to IS NULL AND source_id=? AND relation_type=? "
                "ORDER BY id ASC",
                (source_id, relation_type),
            ).fetchall()
        return [self._row_to_relation(r) for r in rows]

    def find_relation_by_triple(
        self, source_id: int, target_id: int, relation_type: str
    ) -> Relation | None:
        """The row for one exact triple REGARDLESS of validity, or None.

        The schema keeps one row per (source, target, type) — this is the
        lookup behind duplicate-triple recovery when the existing row has
        already been retired.
        """
        row = self._conn.execute(
            "SELECT id, source_id, target_id, relation_type, confidence, "
            "properties, valid_from, valid_to, recorded_at "
            "FROM knowledge_relations "
            "WHERE source_id=? AND target_id=? AND relation_type=?",
            (source_id, target_id, relation_type),
        ).fetchone()
        return None if row is None else self._row_to_relation(row)

    def reopen_relation(self, rid: int) -> Relation:
        """Re-assert a retired relation: clear ``valid_to``, keep the row.

        One row per triple means one validity interval — a fact retired and
        later re-asserted becomes valid again from its ORIGINAL
        ``valid_from``; the retirement gap is not representable (documented
        schema limitation). Raises ``KeyError`` if the relation is missing.
        """
        self.get_relation(rid)
        self._conn.execute(
            "UPDATE knowledge_relations SET valid_to=NULL WHERE id=?", (rid,)
        )
        self._conn.commit()
        return self.get_relation(rid)

    # --- entity <-> frame bridge ------------------------------------------

    def link_entity_frame(self, entity_id: int, frame_id: int) -> None:
        """Write an ``kg_entity_frames`` bridge row linking ``entity_id`` to ``frame_id``.

        The bridge table has ``PRIMARY KEY(entity_id, frame_id)`` so a duplicate
        link is a no-op (INSERT OR IGNORE). This is the entity↔frame bridge the
        contextual KG signal walks.
        """
        self._conn.execute(
            "INSERT OR IGNORE INTO kg_entity_frames(entity_id, frame_id) "
            "VALUES (?, ?)",
            (entity_id, frame_id),
        )
        self._conn.commit()

    def frame_ids_for_entity(self, entity_id: int) -> list[int]:
        """Return the frame_ids linked to ``entity_id`` (bridge: entity → frames)."""
        rows = self._conn.execute(
            "SELECT frame_id FROM kg_entity_frames WHERE entity_id=? "
            "ORDER BY frame_id ASC",
            (entity_id,),
        ).fetchall()
        return [int(r[0]) for r in rows]

    def entity_ids_for_frame(self, frame_id: int) -> list[int]:
        """Return the entity_ids linked to ``frame_id`` (bridge: frame → entities)."""
        rows = self._conn.execute(
            "SELECT entity_id FROM kg_entity_frames WHERE frame_id=? "
            "ORDER BY entity_id ASC",
            (frame_id,),
        ).fetchall()
        return [int(r[0]) for r in rows]

    # --- bitemporal reconstruction ----------------------------------------

    def as_of(self, iso: str) -> KgSnapshot:
        """Reconstruct the KG state valid at time ``T = iso``.

        Returns the entities and relations whose half-open validity interval
        ``[valid_from, valid_to)`` contains T:

            valid_from <= T AND (valid_to IS NULL OR valid_to > T)

        T and the validity bounds are compared as ISO-8601 strings of equal
        shape (lexicographic order == chronological order for equal-format ISO
        datetimes). A bare date ``YYYY-MM-DD`` is padded to
        ``YYYY-MM-DDT00:00:00``; an SQLite ``datetime('now')`` value
        ``YYYY-MM-DD HH:MM:SS`` is normalized to ``YYYY-MM-DDTHH:MM:SS``.
        """
        t = _normalize_iso(iso)
        ent_rows = self._conn.execute(
            "SELECT id, entity_type, name, properties, valid_from, valid_to, "
            "recorded_at FROM knowledge_entities "
            "WHERE _gluomem_norm_iso(valid_from) <= ? "
            "AND (valid_to IS NULL OR _gluomem_norm_iso(valid_to) > ?) "
            "ORDER BY id ASC",
            (t, t),
        ).fetchall()
        rel_rows = self._conn.execute(
            "SELECT id, source_id, target_id, relation_type, confidence, "
            "properties, valid_from, valid_to, recorded_at "
            "FROM knowledge_relations "
            "WHERE _gluomem_norm_iso(valid_from) <= ? "
            "AND (valid_to IS NULL OR _gluomem_norm_iso(valid_to) > ?) "
            "ORDER BY id ASC",
            (t, t),
        ).fetchall()
        entities = [self._row_to_entity(r) for r in ent_rows]
        relations = [self._row_to_relation(r) for r in rel_rows]
        return KgSnapshot(entities=entities, relations=relations)

    # --- seeding + symmetric BFS ------------------------------------------

    def seed_entities(self, query: str, kg_cfg: KgConfig) -> list[int]:
        """Seed KG entities from ``query`` via word-boundary token matching.

        Tokenizes ``query`` on WORD BOUNDARIES (``\\b\\w+\\b``), normalizes
        each token through the alias normalizer, and matches tokens against
        entity names (also normalized) of currently-valid (non-retired)
        entities.

        A token is an eligible seed candidate when its length is >=
        ``kg_cfg.seed_min_len`` OR it appears in ``kg_cfg.tech_allowlist``
        (case-insensitive) — the allowlist lets short technical tokens
        (e.g. ``ai``, ``go``) bypass the length floor.

        NO substring matching: the entity ``ai`` is not seeded from inside
        ``mountain``, but standalone ``AI``/``ai`` IS seeded
        (case-insensitive). Aliases seed their canonical entity via the
        normalizer.

        Returns the list of seeded entity ids (deduplicated, ascending).
        """
        allow = {t.lower() for t in kg_cfg.tech_allowlist}
        min_len = kg_cfg.seed_min_len

        # Eligible, normalized query tokens.
        candidates: set[str] = set()
        for raw in _WORD_RE.findall(query):
            tok = raw.lower()
            if len(tok) >= min_len or tok in allow:
                candidates.add(normalize(raw, kg_cfg))

        if not candidates:
            return []

        # Map normalized entity name -> entity id over valid (non-retired)
        # entities. Only the currently-valid subgraph is seedable.
        rows = self._conn.execute(
            "SELECT id, name FROM knowledge_entities WHERE valid_to IS NULL"
        ).fetchall()
        name_to_ids: dict[str, list[int]] = {}
        for row in rows:
            eid = int(row[0])
            norm_name = normalize(row[1], kg_cfg)
            name_to_ids.setdefault(norm_name, []).append(eid)

        seeded: set[int] = set()
        for tok in candidates:
            for eid in name_to_ids.get(tok, ()):
                seeded.add(eid)
        return sorted(seeded)

    def bfs_distances(self, seeds: list[int], depth: int) -> dict[int, int]:
        """Symmetric BFS over the currently-valid relation subgraph.

        Traversal follows BOTH outgoing and incoming relations (in+out),
        distance is direction-agnostic, and only relations with
        ``valid_to IS NULL`` are traversed. ``depth`` bounds the maximum
        distance returned (seeds are at distance 0).

        Returns a mapping ``entity_id -> distance`` for every reachable entity
        within ``depth`` hops (inclusive of the seeds at distance 0).
        """
        if depth < 0 or not seeds:
            return {}

        # Build symmetric adjacency from valid relations: each relation
        # source->target contributes edges in BOTH directions.
        rows = self._conn.execute(
            "SELECT source_id, target_id FROM knowledge_relations "
            "WHERE valid_to IS NULL"
        ).fetchall()
        adj: dict[int, list[int]] = {}
        for row in rows:
            s = int(row[0])
            t = int(row[1])
            adj.setdefault(s, []).append(t)
            adj.setdefault(t, []).append(s)

        dist: dict[int, int] = {}
        queue: deque[int] = deque()
        for s in seeds:
            if s not in dist:
                dist[s] = 0
                queue.append(s)
        while queue:
            node = queue.popleft()
            d = dist[node]
            if d >= depth:
                continue
            for nb in adj.get(node, ()):
                if nb not in dist:
                    dist[nb] = d + 1
                    queue.append(nb)
        return dist

    # --- contextual KG signal ---------------------------------------------

    def contextual(
        self,
        query: str,
        kg_cfg: KgConfig,
        *,
        recent_entities: list[int] | None = None,
        depth: int,
    ) -> dict[int, int]:
        """Map each candidate frame to its minimum BFS distance from the query.

        The KG contextual signal:

        1. Seed entities from ``query`` via word-boundary token matching
           (:meth:`seed_entities`), optionally unioned with ``recent_entities``
           (e.g. recently-seen entities carried across turns).
        2. Run symmetric :meth:`bfs_distances` over the seeded subgraph up to
           ``depth``. The BFS depth is owned by the scoring config; the caller
           MUST pass it — typically ``cfg.scoring.bfs_depth``.
        3. Map every reached entity to its linked frames via
           ``kg_entity_frames`` and return each candidate frame's MINIMUM BFS
           distance over all entities that link to it.

        Returns a mapping ``frame_id -> distance`` for frames linked to at
        least one reached entity. A frame linked to no reached entity (or
        unlinked entirely) is ABSENT from the map — the caller treats absence
        as "beyond depth" (contextual 0.0).
        """
        # The canonical BFS depth lives on the scoring config
        # (``cfg.scoring.bfs_depth``); the caller threads it through so the KG
        # does not import the scoring section.
        eff_depth = int(depth)
        if eff_depth < 0:
            return {}

        # (1) Seed entities from the query, optionally unioned with recent.
        seeds: set[int] = set(self.seed_entities(query, kg_cfg))
        if recent_entities:
            seeds.update(int(e) for e in recent_entities)
        if not seeds:
            return {}

        # (2) Symmetric BFS distances over the valid relation subgraph.
        distances = self.bfs_distances(sorted(seeds), eff_depth)
        if not distances:
            return {}

        # (3) Map reached entities -> linked frames; keep each frame's MINIMUM
        # distance over all entities that link to it.
        rows = self._conn.execute(
            "SELECT entity_id, frame_id FROM kg_entity_frames"
        ).fetchall()
        frame_distance: dict[int, int] = {}
        for row in rows:
            eid = int(row[0])
            fid = int(row[1])
            d = distances.get(eid)
            if d is None:
                continue
            prev = frame_distance.get(fid)
            if prev is None or d < prev:
                frame_distance[fid] = d
        return frame_distance

    # --- row mappers -------------------------------------------------------

    def _row_to_entity(self, row: sqlite3.Row | tuple) -> Entity:
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

    def _row_to_relation(self, row: sqlite3.Row | tuple) -> Relation:
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


# ---------------------------------------------------------------------------
# Helpers (pure).
# ---------------------------------------------------------------------------


def _load_props(raw: str | None) -> dict[str, object]:
    """Parse a JSON ``properties`` column into a dict (empty on NULL/invalid)."""
    if raw is None:
        return {}
    try:
        loaded = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _normalize_iso(value: str) -> str:
    """Normalize an ISO-ish datetime/date to a fixed-width comparable string.

    A bare date ``YYYY-MM-DD`` is padded to ``YYYY-MM-DDT00:00:00``; an SQLite
    ``datetime('now')`` value ``YYYY-MM-DD HH:MM:SS`` is converted to the
    ``T``-separated ISO form so it compares lexicographically against full
    datetimes on equal footing.
    """
    s = value.strip()
    if len(s) == 10 and s[4] == "-":
        return s + "T00:00:00"
    if " " in s and "T" not in s:
        return s.replace(" ", "T")
    return s


def _now_iso() -> str:
    """Return the current local time as ``YYYY-MM-DDTHH:MM:SS`` (debug helper)."""
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
