"""Resolve extracted entities and relations into the knowledge graph."""

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


def _is_integrity_error(exc: Exception) -> bool:
    return isinstance(exc, sqlite3.IntegrityError) or type(exc).__name__ in (
        "IntegrityError",
        "UniqueViolation",
    )


@dataclass
class ResolvedGraph:
    """IDs of the entities and relations a cognify pass resolved or created."""

    entity_ids: list[int] = field(default_factory=list)
    relation_ids: list[int] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)


def _allow_set(cfg: AppConfig) -> frozenset[str]:
    return frozenset(t.strip().lower() for t in cfg.kg.entity_type_allowlist)


def _resolve_entity(
    candidate: KGEntity, *, kg: KnowledgeGraph, cfg: AppConfig
) -> int | None:
    """Resolve one entity candidate to a valid KG entity ID (create if new)."""
    entity_type = candidate.entity_type
    if cfg.kg.entity_type_mode != "off":
        if entity_type.strip().lower() not in _allow_set(cfg):
            if cfg.kg.entity_type_mode == "drop":
                return None
            entity_type = cfg.kg.retype_to
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
    """Resolve a relation endpoint by name regardless of type (create untyped)."""
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
    """Extract and resolve entities and relations from ``text`` into the KG."""
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
    """Batch cognify: LLM calls fan out; KG resolution stays sequential."""
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
            except Exception as exc:  # noqa: BLE001
                raw.append(exc)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(max_workers, len(thunks))) as pool:
            futures = [pool.submit(thunk) for _, thunk in thunks]
            for future in futures:
                try:
                    raw.append(future.result())
                except Exception as exc:  # noqa: BLE001
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
    """Resolve one extraction result into KG IDs (single-threaded DB phase)."""
    entity_ids: list[int] = []
    name_to_id: dict[str, int] = {}
    for candidate in result.entities:
        eid = _resolve_entity(candidate, kg=kg, cfg=cfg)
        if eid is None:
            continue
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
            continue
        try:
            created = kg.create_relation(
                source_id,
                target_id,
                rel.relation_type,
                confidence=rel.confidence,
                valid_from=valid_from,
            )
        except Exception as exc:
            if not _is_integrity_error(exc):
                raise
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
    """Functional-edge invalidation for ``(s, rtype, t_old != t_new)`` ordered by valid time."""
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
