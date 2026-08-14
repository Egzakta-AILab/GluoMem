"""Word-boundary entity seeding tests.

``KnowledgeGraph.seed_entities(query, cfg.kg)`` tokenizes the query on WORD
BOUNDARIES and matches tokens against entity names with length >=
``seed_min_len(2)``, respecting the ``tech_allowlist``. It MUST NOT do
substring matching — so the entity ``"ai"`` is NOT seeded from inside
``"mountain"``, but standalone ``"AI"``/``"ai"`` IS seeded (case-insensitive).
The normalizer is applied so aliases seed their canonical entity.
"""

from __future__ import annotations

from pathlib import Path

from gluomem.config.app_config import load_config
from gluomem.kg.graph import KnowledgeGraph
from gluomem.storage.db import MindDB


def _kg(tmp_path: Path, *, dim: int = 384) -> tuple[MindDB, KnowledgeGraph]:
    db = MindDB(tmp_path / "test.mind", dim=dim)
    return db, KnowledgeGraph(db)


# --- no substring matching --------------------------------------------------


def test_seeding_no_substring_match(tmp_path: Path) -> None:
    """'ai' entity NOT seeded by 'I climbed a mountain'."""
    db, kg = _kg(tmp_path)
    cfg = load_config()

    ai = kg.create_entity("concept", "ai")

    seeds = kg.seed_entities("I climbed a mountain", cfg.kg)
    # The substring 'ai' inside 'mountain' must NOT seed the 'ai' entity.
    assert ai.id not in seeds
    db.close()


# --- standalone tokens ------------------------------------------------------


def test_seeding_standalone_token_case_insensitive(tmp_path: Path) -> None:
    """'AI' and 'ai' both seed the 'ai' entity."""
    db, kg = _kg(tmp_path)
    cfg = load_config()

    ai = kg.create_entity("concept", "ai")

    # Uppercase standalone token.
    seeds_upper = kg.seed_entities("I use AI daily", cfg.kg)
    assert ai.id in seeds_upper

    # Lowercase standalone token.
    seeds_lower = kg.seed_entities("I use ai daily", cfg.kg)
    assert ai.id in seeds_lower
    db.close()


# --- alias-aware seeding ----------------------------------------------------


def test_seeding_normalizes_aliases_to_canonical(tmp_path: Path) -> None:
    """An alias token seeds the canonical entity (pg -> postgresql entity)."""
    db, kg = _kg(tmp_path)
    cfg = load_config()

    # The canonical entity is named 'postgresql'; a query mentioning 'pg'
    # (an alias group member) should seed it after normalization.
    pg_entity = kg.create_entity("tool", "postgresql")

    seeds = kg.seed_entities("we deploy pg in production", cfg.kg)
    assert pg_entity.id in seeds
    db.close()
