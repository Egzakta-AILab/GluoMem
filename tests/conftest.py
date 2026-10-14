"""Root test fixtures and environment setup."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from gluomem.config.app_config import AppConfig, load_config
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.storage.db import MindDB

_dspy_cache = tempfile.mkdtemp(prefix="gluomem-dspy-cache-")
os.environ.setdefault("DSPY_CACHEDIR", _dspy_cache)


@pytest.fixture
def cfg() -> AppConfig:
    return load_config()


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder(dim=384)


@pytest.fixture
def mind_db(tmp_path: Path) -> Iterator[MindDB]:
    db = MindDB(tmp_path / "test.mind", dim=384)
    yield db
    db.close()


@pytest.fixture
def frame_store(mind_db: MindDB, mock_embedder: MockEmbedder, cfg: AppConfig) -> FrameStore:
    return FrameStore(mind_db, embedder=mock_embedder, chunking=cfg.chunking)


@pytest.fixture
def kg(mind_db: MindDB) -> KnowledgeGraph:
    return KnowledgeGraph(mind_db)


@pytest.fixture
def assert_no_recall(cfg: AppConfig):
    """Assert removed plaintext is absent from recall and recoverable SQL surfaces."""
    from gluomem.frames.lifecycle import export_memories, list_memories
    from gluomem.search.hybrid import HybridSearch
    from gluomem.storage.scope import scope_sql

    def check(
        text: str, *, db: MindDB, store: FrameStore, kg: KnowledgeGraph, search: Any = None
    ) -> None:
        scope = store.scope
        frames = store.all_frames()
        assert text not in str(frames)
        fts_rows = db.fetchall("SELECT content FROM memory_frames_fts")
        assert text not in str([dict(r) for r in fts_rows])
        for table in ("source_events", "memory_jobs", "memory_provenance"):
            rows = db.fetchall(f"SELECT * FROM {table} WHERE {scope_sql()}", scope.as_tuple())
            assert text not in str([dict(row) for row in rows])
        assert text not in str([dict(r) for r in db.fetchall("SELECT * FROM memory_frame_chunks")])
        invalid = {f.id for f in frames if f.status not in ("active", "candidate")}
        assert invalid.isdisjoint(db.frame_vec.ids())
        chunk_ids = {int(r["id"]) for r in db.fetchall("SELECT id FROM memory_frame_chunks")}
        assert db.chunk_vec.ids() <= chunk_ids
        qvec = store.embedder.embed(text)
        assert invalid.isdisjoint(fid for fid, _ in db.frame_vec.search(qvec, 100, scope=scope))
        assert {cid for cid, _ in db.chunk_vec.search(qvec, 100, scope=scope)} <= chunk_ids
        search = search or HybridSearch(db, store, cfg, embedder=store.embedder, kg=kg)
        assert all(text not in hit.frame.content for hit in search.search(text, scope=scope).hits)
        assert text not in str([dict(r) for r in db.fetchall("SELECT * FROM knowledge_entities")])
        for at in ("1900-01-01", "2026-03-01", "2100-01-01"):
            assert text not in str(kg.as_of(at, scope=scope))
        for gop in {f.gop_id for f in frames}:
            assert text not in str(store.reconstruct_state(gop))
        assert text not in str(list_memories(store))
        assert text not in str(export_memories(store))
        assert text not in str([store.lineage(f.id) for f in frames])

    return check
