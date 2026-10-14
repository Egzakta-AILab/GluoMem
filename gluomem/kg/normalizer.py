"""Config-driven alias and predicate-family normalizers for the knowledge graph."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

from gluomem.config.app_config import KgConfig

__all__ = ["alias_map", "expand_aliases", "normalize", "predicate_family"]


def alias_map(kg_cfg: KgConfig) -> dict[str, str]:
    """Build the lowercased ``alias -> canonical`` map from ``KgConfig.alias_groups``."""
    return _alias_map(tuple(tuple(g) for g in kg_cfg.alias_groups))


@lru_cache(maxsize=128)
def _alias_map(groups: tuple[tuple[str, ...], ...]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for group in groups:
        if not group:
            continue
        canonical = group[0].lower()
        for member in group:
            mapping[member.lower()] = canonical
    return mapping


def normalize(token: str, kg_cfg: KgConfig) -> str:
    """Lowercase ``token`` and map alias-group members to their canonical form."""
    lowered = token.lower()
    return alias_map(kg_cfg).get(lowered, lowered)


def expand_aliases(targets: Iterable[str], kg_cfg: KgConfig) -> list[str]:
    """Return ``targets`` together with any alias surface forms that normalize to them."""
    canon_set = {normalize(t, kg_cfg) for t in targets}
    if not canon_set:
        return []
    forms = set(canon_set)
    for group in kg_cfg.alias_groups:
        if not group:
            continue
        canonical = group[0].lower()
        if canonical in canon_set:
            for member in group:
                forms.add(member.lower())
    return sorted(forms)


def predicate_family(relation_type: str, kg_cfg: KgConfig) -> tuple[str, ...]:
    """Return all relation types in ``relation_type``'s predicate family (including itself)."""
    families = _family_map(tuple(tuple(g) for g in kg_cfg.predicate_families))
    return families.get(relation_type.strip().upper(), (relation_type,))


@lru_cache(maxsize=128)
def _family_map(
    groups: tuple[tuple[str, ...], ...],
) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, tuple[str, ...]] = {}
    for group in groups:
        for member in group:
            mapping[member.strip().upper()] = group
    return mapping
