"""Alias normalizer tests.

The normalizer is a pure function ``normalize(token, cfg.kg)``:
lowercase the token, then map any member of an ``alias_groups`` group to that
group's CANONICAL (first) member. Tokens in ``tech_allowlist`` that belong to
no alias group are returned unchanged (still lowercased): ``go -> go``,
``ml -> ml``.
"""

from __future__ import annotations

from gluomem.config.app_config import KgConfig, load_config
from gluomem.kg.normalizer import normalize, predicate_family


def _kg_cfg():
    return load_config().kg


# --- alias groups -----------------------------------------------------------


def test_normalizer_alias_groups_map_to_canonical() -> None:
    """postgres/pg -> postgresql; k8s -> kubernetes."""
    cfg = _kg_cfg()

    # Canonical members map to themselves (lowercased).
    assert normalize("postgresql", cfg) == "postgresql"
    assert normalize("kubernetes", cfg) == "kubernetes"

    # Alias members map to the group's first (canonical) member.
    assert normalize("postgres", cfg) == "postgresql"
    assert normalize("pg", cfg) == "postgresql"
    assert normalize("k8s", cfg) == "kubernetes"

    # Case-insensitive: uppercase aliases still canonicalize.
    assert normalize("Postgres", cfg) == "postgresql"
    assert normalize("PG", cfg) == "postgresql"
    assert normalize("K8S", cfg) == "kubernetes"


# --- allowlist --------------------------------------------------------------


def test_normalizer_preserves_allowlist_tokens() -> None:
    """go -> go, ml -> ml (no alias group, unchanged)."""
    cfg = _kg_cfg()

    # Allowlist tokens that belong to no alias group are returned unchanged
    # (lowercased — the normalizer always lowercases).
    assert normalize("go", cfg) == "go"
    assert normalize("ml", cfg) == "ml"
    assert normalize("GO", cfg) == "go"
    assert normalize("ML", cfg) == "ml"

    # A token in neither an alias group nor the allowlist is still lowercased
    # and returned as-is (the normalizer never drops information).
    assert normalize("Mountain", cfg) == "mountain"


def test_predicate_family_returns_whole_group() -> None:
    cfg = KgConfig(predicate_families=[["LIVES_IN", "MOVED_TO", "RESIDES_IN"]])
    group = ("LIVES_IN", "MOVED_TO", "RESIDES_IN")
    assert predicate_family("MOVED_TO", cfg) == group
    assert predicate_family("lives_in", cfg) == group  # case-insensitive lookup


def test_predicate_family_unknown_type_is_singleton() -> None:
    cfg = KgConfig(predicate_families=[["LIVES_IN", "MOVED_TO"]])
    assert predicate_family("WORKS_AT", cfg) == ("WORKS_AT",)
    # Empty config (the default) degrades to exact-type matching everywhere.
    assert predicate_family("LIVES_IN", KgConfig()) == ("LIVES_IN",)
