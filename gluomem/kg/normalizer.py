"""gluomem.kg.normalizer — alias normalizer.

A pure function ``normalize(token, kg_cfg)`` that lowercases a token and maps
any member of an ``alias_groups`` group to that group's CANONICAL (first)
member. Tokens that belong to no alias group are returned unchanged (still
lowercased) — this preserves ``tech_allowlist`` tokens like ``go``/``ml`` that
have no canonical form.

Example
-------
With ``alias_groups = [["postgresql","postgres","pg"],
                      ["kubernetes","k8s"]]``:

    normalize("pg")       -> "postgresql"   # alias -> canonical
    normalize("postgres") -> "postgresql"
    normalize("k8s")      -> "kubernetes"
    normalize("go")       -> "go"           # no group -> unchanged (lowered)
    normalize("ML")       -> "ml"

The normalizer is config-driven: it reads ``alias_groups`` from the passed
``KgConfig`` and builds the alias->canonical map lazily. Nothing is hardcoded.
"""

from __future__ import annotations

from functools import lru_cache

from gluomem.config.app_config import KgConfig

__all__ = ["normalize", "alias_map", "predicate_family"]


def alias_map(kg_cfg: KgConfig) -> dict[str, str]:
    """Build the alias -> canonical map from ``KgConfig.alias_groups``.

    Each group's first member is the canonical form; every member (including
    the canonical itself) maps to the lowercased canonical. Groups are
    lowercased on construction so lookups are case-insensitive.
    """
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
    """Lowercase ``token`` and map alias-group members to their canonical form.

    Parameters
    ----------
    token:
        A single word-boundary token from a query (or an entity name).
    kg_cfg:
        The ``KgConfig`` carrying ``alias_groups`` (and ``tech_allowlist``,
        which is honored implicitly: tokens in no alias group are returned
        unchanged after lowercasing).

    Returns
    -------
    The normalized (lowercased, canonicalized) token.
    """
    lowered = token.lower()
    return alias_map(kg_cfg).get(lowered, lowered)


def predicate_family(relation_type: str, kg_cfg: KgConfig) -> tuple[str, ...]:
    """All relation types in ``relation_type``'s predicate family (incl. itself).

    Mirrors ``alias_groups`` for relation predicates: with
    ``predicate_families = [["LIVES_IN", "MOVED_TO", "RESIDES_IN"]]``, any
    member returns the whole group (members as configured; lookup is
    case-insensitive). A type in no family returns a 1-tuple of itself, so
    callers can iterate unconditionally — empty config = exact-type matching.
    """
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
