"""gluomem.write.cognify — resolve extracted entities/relations into the KG.

The extraction facade returns entity/relation candidates by NAME
(``gluomem.llm.outputs.KGEntity``/``KGRelation``); this module resolves them
into KG row ids:

- Entity resolution is alias-aware whole-name matching
  (``kg.find_valid_entities``) with create-if-missing — idempotent across
  re-ingest (``k8s`` resolves to an existing ``kubernetes`` entity).
- Relation endpoints resolve through the same entity resolution (an endpoint
  the extractor did not list as an entity is created untyped). A duplicate
  triple raises ``sqlite3.IntegrityError`` (UNIQUE source/target/type); the
  gate recovers idempotently by looking the existing edge up.
- ``retire_conflicting_relations`` implements functional-edge invalidation on
  the contradiction path: of ``(s, rtype, t_new)`` and a valid
  ``(s, rtype, t_old != t_new)`` the one valid from earlier is closed when the
  other begins — bitemporal retire-not-delete, ordered by valid time.

dspy-free: only the ``Extractors`` protocol and pydantic output types cross
this boundary.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gluomem.kg.normalizer import predicate_family
from gluomem.timeutil import utc_key

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from gluomem.config.app_config import AppConfig, KgConfig
    from gluomem.kg.graph import KnowledgeGraph, Relation
    from gluomem.llm.outputs import CognifyResult, Extractors, KGEntity


_UNTYPED = "unknown"


@dataclass
class ResolvedGraph:
    """Ids of the entities/relations a cognify pass resolved or created."""

    entity_ids: list[int] = field(default_factory=list)
    relation_ids: list[int] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)


def _allow_set(cfg: AppConfig) -> frozenset[str]:
    return frozenset(t.strip().lower() for t in cfg.kg.entity_type_allowlist)


def _resolve_entity(
    candidate: KGEntity, *, kg: KnowledgeGraph, cfg: AppConfig
) -> int | None:
    """Resolve one entity candidate to a valid KG entity id (create if new).

    Returns ``None`` when the entity-type gate DROPS the candidate
    (``kg.entity_type_mode='drop'`` and its type is not allowlisted) — the
    caller prunes any relation that touched it. ``mode='off'`` (default) skips
    the gate entirely.
    """
    entity_type = candidate.entity_type
    if cfg.kg.entity_type_mode != "off":
        if entity_type.strip().lower() not in _allow_set(cfg):
            if cfg.kg.entity_type_mode == "drop":
                return None
            entity_type = cfg.kg.retype_to  # 'retype': relabel, keep the node
    existing = kg.find_valid_entities(candidate.name, cfg.kg, entity_type=entity_type)
    if existing:
        return existing[0].id
    created = kg.create_entity(
        entity_type,
        candidate.name,
        properties=dict(candidate.properties) or None,
    )
    return created.id


def _resolve_endpoint(name: str, *, kg: KnowledgeGraph, cfg: AppConfig) -> int | None:
    """Resolve a relation endpoint by name regardless of type (create untyped).

    Returns ``None`` under ``kg.entity_type_mode='drop'`` when the name is not
    already a valid entity — materializing it would create exactly the untyped
    junk node the drop gate removes, so the relation is pruned instead.
    """
    existing = kg.find_valid_entities(name, cfg.kg)
    if existing:
        return existing[0].id
    if cfg.kg.entity_type_mode == "drop":
        return None
    return kg.create_entity(_UNTYPED, name).id


def cognify_text(
    text: str,
    *,
    facade: Extractors,
    kg: KnowledgeGraph,
    cfg: AppConfig,
    valid_from: str | None = None,
) -> ResolvedGraph:
    """Extract + resolve entities/relations from ``text`` into the KG.

    ``valid_from`` backdates new relations to the observation's event time
    (bitemporal): the edge is valid from when the fact HAPPENED, not from
    ingest time. Returns the resolved ids (idempotent on re-ingest).
    """
    if not cfg.write.cognify.enabled:
        return ResolvedGraph()
    return resolve_cognify_result(
        facade.cognify(text), kg=kg, cfg=cfg, valid_from=valid_from
    )


def cognify_texts(
    texts: Sequence[str],
    *,
    facade: Extractors,
    kg: KnowledgeGraph,
    cfg: AppConfig,
    valid_from_list: Sequence[str | None],
    max_workers: int,
) -> list[ResolvedGraph | Exception]:
    """Batch cognify: LLM calls fan out; KG resolution stays sequential.

    The extraction (facade) phase is pure LLM I/O and runs on up to
    ``max_workers`` threads; the resolution phase writes the KG and runs on
    the calling thread IN INPUT ORDER, so entity/relation ids are identical
    to a fully sequential run. Per-text failures come back as the exception
    at that index (the caller records them; one bad text never aborts the
    batch).
    """
    if not cfg.write.cognify.enabled:
        return [ResolvedGraph() for _ in texts]

    def call(text: str) -> Callable[[], CognifyResult]:
        return lambda: facade.cognify(text)

    thunks = [(str(i), call(text)) for i, text in enumerate(texts)]
    raw: list[CognifyResult | Exception] = []
    if max_workers <= 1 or len(thunks) <= 1:
        for _, thunk in thunks:
            try:
                raw.append(thunk())
            except Exception as exc:  # noqa: BLE001 - recorded per index
                raw.append(exc)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(max_workers, len(thunks))) as pool:
            futures = [pool.submit(thunk) for _, thunk in thunks]
            for future in futures:
                try:
                    raw.append(future.result())
                except Exception as exc:  # noqa: BLE001 - recorded per index
                    raw.append(exc)

    out: list[ResolvedGraph | Exception] = []
    for result, valid_from in zip(raw, valid_from_list, strict=True):
        if isinstance(result, Exception):
            out.append(result)
            continue
        out.append(
            resolve_cognify_result(result, kg=kg, cfg=cfg, valid_from=valid_from)
        )
    return out


def resolve_cognify_result(
    result: CognifyResult,
    *,
    kg: KnowledgeGraph,
    cfg: AppConfig,
    valid_from: str | None = None,
) -> ResolvedGraph:
    """Resolve one extraction result into KG ids (single-threaded DB phase)."""
    entity_ids: list[int] = []
    name_to_id: dict[str, int] = {}
    for candidate in result.entities:
        eid = _resolve_entity(candidate, kg=kg, cfg=cfg)
        if eid is None:
            continue  # dropped by the entity-type gate (drop mode)
        if eid not in entity_ids:
            entity_ids.append(eid)
        name_to_id.setdefault(candidate.name, eid)

    relation_ids: list[int] = []
    relations: list[Relation] = []
    for rel in result.relations:
        source_id = name_to_id.get(rel.source)
        if source_id is None:
            source_id = _resolve_endpoint(rel.source, kg=kg, cfg=cfg)
            if source_id is not None:
                name_to_id[rel.source] = source_id
        target_id = name_to_id.get(rel.target)
        if target_id is None:
            target_id = _resolve_endpoint(rel.target, kg=kg, cfg=cfg)
            if target_id is not None:
                name_to_id[rel.target] = target_id
        if source_id is None or target_id is None:
            continue  # an endpoint was dropped → prune the relation
        try:
            created = kg.create_relation(
                source_id,
                target_id,
                rel.relation_type,
                confidence=rel.confidence,
                valid_from=valid_from,
            )
        except sqlite3.IntegrityError:
            # Duplicate triple. Valid row -> recover it (idempotent re-ingest).
            # Retired row -> the fact is being RE-ASSERTED after a
            # contradiction retired it (real LoCoMo does this): reopen the
            # edge from this assertion's valid time (the retirement gap is
            # kept in the relation history), feeding contradiction detection
            # like any fresh edge.
            existing = [
                r
                for r in kg.find_valid_relations(source_id, rel.relation_type)
                if r.target_id == target_id
            ]
            if existing:
                created = existing[0]
            else:
                retired_row = kg.find_relation_by_triple(
                    source_id, target_id, rel.relation_type
                )
                if retired_row is None:
                    raise
                created = kg.reopen_relation(retired_row.id, valid_from=valid_from)
        if created.id not in relation_ids:
            relation_ids.append(created.id)
            relations.append(created)

    return ResolvedGraph(
        entity_ids=entity_ids, relation_ids=relation_ids, relations=relations
    )


def retire_conflicting_relations(
    new_relations: list[Relation],
    *,
    kg: KnowledgeGraph,
    kg_cfg: KgConfig | None = None,
) -> list[int]:
    """Functional-edge invalidation for ``(s, rtype, t_old != t_new)``.

    For each new relation, every OTHER currently-valid relation with the same
    source and type but a different target conflicts with it. With
    ``kg_cfg.predicate_families`` configured, "same type" widens to the whole
    predicate family, so ``MOVED_TO`` can retire a stale ``LIVES_IN``.

    The conflict is resolved by VALID time, not by recording order:

    - the new edge is valid from on or after the old one: the old edge is
      closed at the new edge's ``valid_from`` (when the change happened);
    - the new edge is valid from strictly earlier (a backdated, older fact
      recorded late): the old edge stays current and the new edge is closed
      at the old edge's ``valid_from``.

    Retire-not-delete either way. Returns the retired relation ids.
    """
    retired: list[int] = []
    for rel in new_relations:
        if kg_cfg is not None:
            rtypes = predicate_family(rel.relation_type, kg_cfg)
        else:
            rtypes = (rel.relation_type,)
        for rtype in rtypes:
            for existing in kg.find_valid_relations(rel.source_id, rtype):
                if (
                    existing.id == rel.id
                    or existing.target_id == rel.target_id
                    or existing.id in retired
                    or rel.id in retired
                ):
                    continue
                if utc_key(rel.valid_from) >= utc_key(existing.valid_from):
                    kg.retire_relation(existing.id, valid_to=rel.valid_from)
                    retired.append(existing.id)
                else:
                    kg.retire_relation(rel.id, valid_to=existing.valid_from)
                    retired.append(rel.id)
    return retired
