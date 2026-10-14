"""Root test fixtures and environment setup."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

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
