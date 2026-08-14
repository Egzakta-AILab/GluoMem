"""Frozen-extraction integration: a replay ingest reproduces the record ingest.

This is the property that makes an ingest-time A/B trustworthy — replaying the
cached extraction into a fresh DB yields byte-identical frames, so any per-arm
delta is the ablation, not extraction nondeterminism.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from gluomem.config.app_config import load_config
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.llm import CachedExtractors, FakeExtractors
from gluomem.storage.db import MindDB
from gluomem.write.gate import InjectionScanner
from gluomem.write.ingest import Conversation, Session, Turn, ingest_conversation


def _conv() -> Conversation:
    return Conversation(
        conv_id="c1",
        speakers=("Alice", "Bob"),
        sessions=[
            Session(
                index=1,
                date_time=datetime(2023, 5, 8),
                turns=[Turn("Alice", "Alice moved to Denver. She likes tea.")],
            ),
            Session(
                index=2,
                date_time=datetime(2023, 5, 9),
                turns=[Turn("Bob", "Bob lives in Boston now.")],
            ),
        ],
    )


def _env(tmp_path: Path, name: str):
    cfg = load_config()
    db = MindDB(tmp_path / name, dim=cfg.storage.vector_dim)
    store = FrameStore(db, embedder=MockEmbedder(dim=db.dim), chunking=cfg.chunking)
    return cfg, store, KnowledgeGraph(db), db


def _frames(db: MindDB):
    # Exclude wall-clock columns (created_at/recorded_at) — compare the
    # extraction-determined content only.
    return db.conn.execute(
        "SELECT frame_type, gop_id, content, content_hash, event_time "
        "FROM memory_frames ORDER BY id"
    ).fetchall()


def test_record_then_replay_reproduces_db(tmp_path: Path) -> None:
    cache = str(tmp_path / "extraction.jsonl")

    cfg, store_a, kg_a, db_a = _env(tmp_path, "a.mind")
    rec = CachedExtractors(FakeExtractors(banner=False), path=cache, mode="record")
    ingest_conversation(
        _conv(),
        facade=rec,
        scanner=InjectionScanner(cfg.safety),
        store=store_a,
        kg=kg_a,
        cfg=cfg,
    )

    cfg, store_b, kg_b, db_b = _env(tmp_path, "b.mind")
    inner = FakeExtractors(banner=False)  # must NOT be touched on replay
    rep = CachedExtractors(inner, path=cache, mode="replay")
    ingest_conversation(
        _conv(),
        facade=rep,
        scanner=InjectionScanner(cfg.safety),
        store=store_b,
        kg=kg_b,
        cfg=cfg,
    )

    assert _frames(db_a) == _frames(db_b)  # replay reproduces the recorded DB
    assert rep.cache_misses == 0
    assert sum(inner.calls.values()) == 0  # every extraction served from cache
