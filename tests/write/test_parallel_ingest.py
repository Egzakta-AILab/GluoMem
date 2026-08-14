"""Parallel extraction equivalence — write.parallelism must not change results.

LLM calls fan out to threads; DB writes stay single-threaded and are applied
in observation order, so a parallel run must be byte-identical to a
sequential one (frames, entities, relations, ids).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gluomem.config.app_config import load_config
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.llm import FakeExtractors
from gluomem.storage.db import MindDB
from gluomem.write.cognify import cognify_texts
from gluomem.write.gate import InjectionScanner
from gluomem.write.ingest import ingest_conversation, load_locomo
from gluomem.write.lanes import run_ordered

FIXTURE = Path(__file__).parent.parent / "fixtures" / "locomo_tiny.json"


def _ingest_snapshot(tmp_path: Path, parallelism: int) -> tuple[list, list, list]:
    cfg = load_config().with_overrides({"write.parallelism": parallelism})
    db = MindDB(tmp_path / f"p{parallelism}.mind", dim=cfg.storage.vector_dim)
    store = FrameStore(db, embedder=MockEmbedder(dim=db.dim), chunking=cfg.chunking)
    kg = KnowledgeGraph(db)
    conv = load_locomo(FIXTURE)[0]
    ingest_conversation(
        conv,
        facade=FakeExtractors(banner=False),
        scanner=InjectionScanner(cfg.safety),
        store=store,
        kg=kg,
        cfg=cfg,
    )
    frames = db.conn.execute(
        "SELECT id, frame_type, gop_id, content, importance, trust, event_time "
        "FROM memory_frames ORDER BY id"
    ).fetchall()
    entities = db.conn.execute(
        "SELECT id, entity_type, name FROM knowledge_entities ORDER BY id"
    ).fetchall()
    relations = db.conn.execute(
        "SELECT id, source_id, target_id, relation_type FROM knowledge_relations "
        "ORDER BY id"
    ).fetchall()
    return frames, entities, relations


def test_parallel_ingest_identical_to_sequential(tmp_path: Path) -> None:
    sequential = _ingest_snapshot(tmp_path, parallelism=1)
    parallel = _ingest_snapshot(tmp_path, parallelism=8)
    assert parallel == sequential


def test_run_ordered_preserves_input_order() -> None:
    thunks = [(str(i), lambda i=i: i * 10) for i in range(20)]
    seq = run_ordered(thunks, max_workers=1)
    par = run_ordered(thunks, max_workers=8)
    assert par == seq == [(str(i), i * 10) for i in range(20)]


def test_run_ordered_propagates_first_exception() -> None:
    def boom() -> int:
        raise RuntimeError("lane failed")

    thunks = [("ok", lambda: 1), ("bad", boom), ("ok2", lambda: 2)]
    with pytest.raises(RuntimeError, match="lane failed"):
        run_ordered(thunks, max_workers=4)


def test_cognify_texts_records_per_index_failures(env) -> None:
    class _Facade(FakeExtractors):
        def cognify(self, text):
            if "bad" in text:
                raise RuntimeError("cognify exploded")
            return super().cognify(text)

    results = cognify_texts(
        ["fine one", "bad one", "fine two"],
        facade=_Facade(banner=False),
        kg=env.kg,
        cfg=env.cfg,
        valid_from_list=[None, None, None],
        max_workers=4,
    )
    assert not isinstance(results[0], Exception)
    assert isinstance(results[1], Exception)
    assert not isinstance(results[2], Exception)


def test_parallelism_validator_floor() -> None:
    cfg = load_config()
    with pytest.raises(Exception, match="parallelism|greater"):
        cfg.with_overrides({"write.parallelism": 0})
