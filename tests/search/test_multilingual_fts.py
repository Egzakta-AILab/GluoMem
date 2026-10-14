"""Multilingual Serbian (Cyrillic, Latin with diacritics, ASCII-folded) and English FTS tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gluomem.config.app_config import load_config
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.search.hybrid import HybridSearch
from gluomem.search.multilingual import (
    cyrillic_to_latin,
    expand_fts5_query,
    fold_diacritics,
    fold_multilingual_text,
    fold_query_text,
    fold_to_ascii,
)
from gluomem.search.sanitize import sanitize
from gluomem.storage.db import MindDB
from gluomem.storage.postgres_backend import PgTsVectorStore
from gluomem.storage.scope import MemoryScope


def test_cyrillic_latin_and_ascii_folding_primitives() -> None:
    cyr = "Састанак у Београду: извештај је послао Ђорђе Човић за џеп и љубав."
    lat = cyrillic_to_latin(cyr)
    assert "Beogradu" in lat
    assert "izveštaj" in lat
    assert "Đorđe" in lat
    assert "Čović" in lat
    assert "džep" in lat
    assert "ljubav" in lat

    folded = fold_diacritics(lat)
    assert "izvestaj" in folded
    assert "Djordje" in folded
    assert "Covic" in folded
    assert "dzep" in folded

    assert fold_to_ascii("Ђорђе извештај Београд") == "djordje izvestaj beograd"
    assert fold_to_ascii("Đorđe izveštaj Beograd") == "djordje izvestaj beograd"
    assert fold_to_ascii("Djordje izvestaj Beograd") == "djordje izvestaj beograd"

    idx_text = fold_multilingual_text("Састанак у Београду: извештај је послао Ђорђе")
    expected_tokens = (
        "београду",
        "beogradu",
        "beograd",
        "извештај",
        "izveštaj",
        "izvestaj",
        "ђорђе",
        "đorđe",
        "djordje",
        "dorde",
    )
    for expected in expected_tokens:
        assert expected in idx_text.split()


@pytest.mark.parametrize(
    ("stored_script", "stored_text"),
    [
        (
            "cyrillic",
            "Квартални извештај за пројекат Београд потписао је Ђорђе Петровић.",
        ),
        (
            "latin_diacritic",
            "Kvartalni izveštaj za projekat Beograd potpisao je Đorđe Petrović.",
        ),
        (
            "ascii_folded",
            "Kvartalni izvestaj za projekat Beograd potpisao je Djordje Petrovic.",
        ),
    ],
)
@pytest.mark.parametrize(
    ("query_script", "query_text"),
    [
        ("cyrillic", "Београд извештај Ђорђе"),
        ("latin_diacritic", "Beograd izveštaj Đorđe"),
        ("ascii_folded", "Beograd izvestaj Djordje"),
    ],
)
def test_sqlite_cross_script_serbian_fts_and_hybrid_search(
    tmp_path: Path,
    stored_script: str,
    stored_text: str,
    query_script: str,
    query_text: str,
) -> None:
    cfg = load_config().with_overrides(
        {
            "components.allow_fake": True,
            "embedding.model": "mock",
            "models.mock.dim": 16,
            "storage.vector_dim": 16,
        }
    )
    scope = MemoryScope("egzakta", "mikhail", "personal")
    db = MindDB(tmp_path / f"{stored_script}_{query_script}.mind", dim=16)
    emb = MockEmbedder(dim=16)
    store = FrameStore(db, embedder=emb, scope=scope)

    target = store.create_i_frame("gop-sr", stored_text, scope=scope)
    _distractor = store.create_i_frame(
        "gop-other",
        "Unrelated deployment log for cluster us-east-1.",
        scope=scope,
    )

    # 1. Direct FTS5 search matches across all 3x3 script combinations
    sanitized = sanitize(query_text)
    fts_ids = db.fts.search(sanitized, limit=5, scope=scope)
    assert target.id in fts_ids

    # 2. HybridSearch retrieves target frame as top hit
    hs = HybridSearch(db, store, cfg, embedder=emb, scope=scope)
    res = hs.search(query_text, limit=5, scope=scope)
    assert [h.frame.id for h in res.hits][:1] == [target.id]

    # 3. index_state remains ready (get_text returns original un-folded content)
    assert store.index_state(target.id).ready
    db.close()


def test_sqlite_like_fallback_matches_across_scripts(tmp_path: Path) -> None:
    scope = MemoryScope("egzakta", "mikhail", "personal")
    db = MindDB(tmp_path / "like.mind", dim=16)
    emb = MockEmbedder(dim=16)
    store = FrameStore(db, embedder=emb, scope=scope)

    f_cyr = store.create_i_frame(
        "gop-1", "Поверљиви извештај Ђорђе", scope=scope
    )
    assert db.fts.search_like(["izvestaj", "djordje"], limit=5, scope=scope) == [
        f_cyr.id
    ]
    assert db.fts.search_like(["извештај", "Ђорђе"], limit=5, scope=scope) == [
        f_cyr.id
    ]
    db.close()


def test_english_stemming_and_technical_tokens_preserved(tmp_path: Path) -> None:
    cfg = load_config().with_overrides(
        {
            "components.allow_fake": True,
            "embedding.model": "mock",
            "models.mock.dim": 16,
            "storage.vector_dim": 16,
        }
    )
    db = MindDB(tmp_path / "en.mind", dim=16)
    emb = MockEmbedder(dim=16)
    store = FrameStore(db, embedder=emb)

    f1 = store.create_i_frame(
        "gop-en",
        "The worker service was running GPT-4 v1.2.3 on node.alpha.",
    )
    # Porter stemming: 'run' matches 'running', technical tokens stay intact
    assert f1.id in db.fts.search(sanitize("run GPT-4 v1.2.3"), limit=5)

    hs = HybridSearch(db, store, cfg, embedder=emb)
    res = hs.search("run GPT-4 v1.2.3", limit=5)
    assert any(h.frame.id == f1.id for h in res.hits)
    db.close()


def test_postgres_tsvector_store_combines_simple_folded_and_english_queries() -> None:
    executed: list[tuple[str, tuple[Any, ...]]] = []

    class _FakePgBackend:
        dialect = "postgres"

        def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
            executed.append((sql, tuple(params)))

        def fetchall(
            self, sql: str, params: tuple[Any, ...] = ()
        ) -> list[dict[str, Any]]:
            executed.append((sql, tuple(params)))
            return [{"rowid": 42}]

        def commit(self) -> None:
            pass

    backend = _FakePgBackend()
    pg_fts = PgTsVectorStore(backend)  # type: ignore[arg-type]
    scope = MemoryScope("egzakta", "mikhail", "workspace")

    pg_fts.upsert(42, "Извештај за Београд је послао Ђорђе.")
    upsert_sql, upsert_params = executed[-1]
    assert "INSERT INTO memory_frames_fts(rowid, content, folded_content)" in upsert_sql
    assert upsert_params[0] == 42
    assert "izvestaj" in upsert_params[2]
    assert "beograd" in upsert_params[2]
    assert "djordje" in upsert_params[2]

    for query in (
        "Београд извештај Ђорђе",
        "Beograd izveštaj Đorđe",
        "Beograd izvestaj Djordje",
    ):
        hits = pg_fts.search(
            sanitize(query), limit=5, scope=scope, now_iso="2026-06-01T00:00:00"
        )
        assert hits == [42]
        search_sql, search_params = executed[-1]
        assert "websearch_to_tsquery('simple', ?)" in search_sql
        assert "websearch_to_tsquery('english', ?)" in search_sql
        # All three scripts normalize to the exact same folded_terms parameter!
        assert search_params[0] == "beograd izvestaj djordje"
        assert search_params[2:5] == ("egzakta", "mikhail", "workspace")

    assert fold_query_text("Београд извештај Ђорђе") == "beograd izvestaj djordje"
    assert (
        expand_fts5_query(sanitize("GPT-4 v1.2.3"))
        == '"GPT-4" "v1.2.3"'
    )
