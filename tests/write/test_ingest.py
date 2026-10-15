"""Ingest pipeline tests: lanes, frame models, LoCoMo e2e, backdating, parallelism, and cache."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from gluomem.config.app_config import load_config
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.llm import (
    CachedExtractors,
    CognifyResult,
    ContradictionVerdict,
    ExtractedEvent,
    ExtractedFact,
    FakeExtractors,
    KGEntity,
    KGRelation,
)
from gluomem.storage.db import MindDB
from gluomem.write.cognify import cognify_texts
from gluomem.write.gate import InjectionScanner, admit
from gluomem.write.ingest import (
    Conversation,
    Session,
    Turn,
    _parse_locomo_datetime,
    ingest_conversation,
    load_locomo,
)
from gluomem.write.lanes import extract_observations, lane_importance, run_ordered

FIXTURE = Path(__file__).parent.parent / "fixtures" / "locomo_tiny.json"
_REF = datetime(2023, 5, 8)


def test_lanes_extraction_order_event_time_and_disablement(env) -> None:
    obs = extract_observations(
        "Alice got promoted yesterday. Bob likes tea.",
        facade=FakeExtractors(banner=False),
        reference_date=_REF,
        speakers=["Alice", "Bob"],
        speaker_texts={"Alice": "I got promoted.", "Bob": "I like tea."},
        cfg=env.cfg,
    )
    lanes = [o.lane for o in obs]
    assert lanes == sorted(lanes, key=["fact", "event", "profile"].index)
    events = [o for o in obs if o.lane == "event"]
    assert len(events) == 1 and events[0].event_time == "2023-05-07T00:00:00"
    assert [o.speaker for o in obs if o.lane == "profile"] == ["Alice", "Bob"]

    no_date = extract_observations(
        "text",
        facade=FakeExtractors(
            events=[[ExtractedEvent(text="happened", event_date=None)]], banner=False
        ),
        reference_date=_REF,
        speakers=[],
        speaker_texts={},
        cfg=env.cfg,
    )
    assert [o for o in no_date if o.lane == "event"][0].event_time is None

    disabled_cfg = env.cfg.with_overrides(
        {
            "write.lanes.fact.enabled": False,
            "write.lanes.event.enabled": False,
            "write.lanes.profile.enabled": False,
        }
    )
    facade = FakeExtractors(banner=False)
    assert (
        extract_observations(
            "text",
            facade=facade,
            reference_date=_REF,
            speakers=["Alice"],
            speaker_texts={"Alice": "hi"},
            cfg=disabled_cfg,
        )
        == []
    )
    assert sum(facade.calls.values()) == 0
    assert lane_importance("fact", env.cfg) == "normal"
    assert lane_importance("profile", env.cfg) == "important"


_ALICE = KGEntity(entity_type="person", name="alice")
_BOSTON = KGEntity(entity_type="city", name="boston")
_SEATTLE = KGEntity(entity_type="city", name="seattle")
_F1 = "Alice lives in Boston."
_F2 = "Alice enjoys hiking around Boston."
_F3 = "Alice moved from Boston to Seattle."


@pytest.mark.parametrize(
    ("mode", "expected_types", "expected_i", "expected_p", "expected_b", "expected_demoted"),
    [
        ("ipb", ["I", "P", "I", "B"], 2, 1, 1, 0),
        ("i_only", ["I", "I", "I"], 3, 0, 0, 0),
        ("overwrite", ["I", "I", "I"], 3, 0, 0, 2),
    ],
)
def test_frame_models_ablation(
    env,
    scripted_facade_cls,
    mode: str,
    expected_types: list[str],
    expected_i: int,
    expected_p: int,
    expected_b: int,
    expected_demoted: int,
) -> None:
    cfg = env.cfg.with_overrides(
        {
            "write.frame_model": mode,
            "write.raw_turns.enabled": False,
            "write.lanes.event.enabled": False,
            "write.lanes.profile.enabled": False,
        }
    )
    facade = scripted_facade_cls(
        facts={
            "session one": [ExtractedFact(text=_F1)],
            "session two": [ExtractedFact(text=_F2)],
            "session three": [ExtractedFact(text=_F3)],
        },
        cognify_map={
            _F1: CognifyResult(
                entities=[_ALICE, _BOSTON],
                relations=[KGRelation(source="alice", target="boston", relation_type="LIVES_IN")],
            ),
            _F2: CognifyResult(entities=[_ALICE, _BOSTON]),
            _F3: CognifyResult(
                entities=[_ALICE, _BOSTON, _SEATTLE],
                relations=[KGRelation(source="alice", target="seattle", relation_type="LIVES_IN")],
            ),
        },
        verdicts={
            "moved from Boston": ContradictionVerdict(
                is_contradiction=True, description="Alice moved cities"
            )
        },
    )
    conv = Conversation(
        conv_id="c1",
        speakers=("Alice", "Bob"),
        sessions=[
            Session(
                index=i,
                date_time=datetime(2023, 5, 8 + i),
                turns=[Turn(speaker="Alice", text=f"chatter in session {w}")],
            )
            for i, w in ((1, "one"), (2, "two"), (3, "three"))
        ],
    )
    report = ingest_conversation(
        conv,
        facade=facade,
        scanner=InjectionScanner(cfg.safety),
        store=env.store,
        kg=env.kg,
        cfg=cfg,
    )
    assert (report.i_frames, report.p_frames, report.b_frames, report.demoted) == (
        expected_i,
        expected_p,
        expected_b,
        expected_demoted,
    )
    rows = env.db.conn.execute(
        "SELECT frame_type, importance FROM memory_frames WHERE gop_id='c1:facts' ORDER BY id"
    ).fetchall()
    assert [r[0] for r in rows] == expected_types
    if mode == "overwrite":
        assert [r[1] for r in rows] == ["deprecated", "deprecated", "normal"]


def test_locomo_parse_and_ingest_e2e(env, tmp_path: Path) -> None:
    assert _parse_locomo_datetime("1:56 pm on 8 May, 2023").hour == 13
    with pytest.raises(ValueError, match="unrecognized"):
        _parse_locomo_datetime("mai huitieme, treize heures")

    bad = tmp_path / "bad.json"
    bad.write_text('[{"sample_id": "x"}]', encoding="utf-8")
    with pytest.raises(ValueError, match="missing sample_id/conversation"):
        load_locomo(bad)

    convs = load_locomo(FIXTURE)
    assert len(convs) == 1
    conv = convs[0]
    assert conv.conv_id == "conv-tiny-1" and conv.speakers == ("Alice", "Bob")
    assert "shares photo: a golden retriever puppy" in conv.sessions[0].turns[3].text

    report = ingest_conversation(
        conv,
        facade=FakeExtractors(banner=False),
        scanner=InjectionScanner(env.cfg.safety),
        store=env.store,
        kg=env.kg,
        cfg=env.cfg,
    )
    assert (report.conversations, report.sessions, report.turns, report.quarantined) == (
        1,
        3,
        9,
        1,
    )
    assert report.deduped > 0

    leaked = env.db.conn.execute(
        "SELECT COUNT(*) FROM memory_frames "
        "WHERE content LIKE '%ignore all previous%' AND gop_id NOT LIKE '%:session:%'"
    ).fetchone()[0]
    assert leaked == 0

    latest_i, _ = env.store.reconstruct_state("conv-tiny-1:profile:Alice")
    assert latest_i is not None

    # Re-ingest is idempotent.
    count_before = env.db.conn.execute("SELECT COUNT(*) FROM memory_frames").fetchone()[0]
    report2 = ingest_conversation(
        conv,
        facade=FakeExtractors(banner=False),
        scanner=InjectionScanner(env.cfg.safety),
        store=env.store,
        kg=env.kg,
        cfg=env.cfg,
    )
    count_after = env.db.conn.execute("SELECT COUNT(*) FROM memory_frames").fetchone()[0]
    assert count_after == count_before and report2.i_frames == 0


def test_backdate_created_at_in_store_gate_and_ingest(env) -> None:
    ts = "2023-05-08T13:56:00"
    scanner = InjectionScanner(env.cfg.safety)
    kw = dict(
        gop="g1",
        scanner=scanner,
        store=env.store,
        kg=env.kg,
        cfg=env.cfg,
        created_at=ts,
    )
    base = admit("base", **kw)
    assert base.frame is not None and base.frame.created_at == ts
    p = admit("delta", frame_type="P", base_id=base.frame.id, **kw)
    b = admit(
        "corr",
        frame_type="B",
        base_id=base.frame.id,
        references=[base.frame.id],
        **kw,
    )
    assert p.frame is not None and p.frame.created_at == ts
    assert b.frame is not None and b.frame.created_at == ts

    cfg = env.cfg.with_overrides({"write.backdate_created_at": True})
    conv = load_locomo(FIXTURE)[0]
    ingest_conversation(
        conv,
        facade=FakeExtractors(banner=False),
        scanner=InjectionScanner(cfg.safety),
        store=env.store,
        kg=env.kg,
        cfg=cfg,
    )
    rows = env.db.conn.execute(
        "SELECT DISTINCT created_at FROM memory_frames WHERE gop_id LIKE '%:session:%'"
    ).fetchall()
    assert {r[0] for r in rows} == {
        "2023-05-08T13:56:00",
        "2023-06-21T10:15:00",
        "2023-08-02T19:30:00",
    }


def test_parallel_ingest_and_run_ordered(env, tmp_path: Path) -> None:
    thunks = [(str(i), lambda i=i: i * 10) for i in range(10)]
    assert run_ordered(thunks, max_workers=1) == run_ordered(thunks, max_workers=4)

    def boom() -> int:
        raise RuntimeError("lane failed")

    with pytest.raises(RuntimeError, match="lane failed"):
        run_ordered([("ok", lambda: 1), ("bad", boom)], max_workers=2)

    class _BoomFacade(FakeExtractors):
        def cognify(self, text: str):
            if "bad" in text:
                raise RuntimeError("cognify exploded")
            return super().cognify(text)

    results = cognify_texts(
        ["fine one", "bad one", "fine two"],
        facade=_BoomFacade(banner=False),
        kg=env.kg,
        cfg=env.cfg,
        valid_from_list=[None, None, None],
        max_workers=2,
    )
    assert not isinstance(results[0], Exception)
    assert isinstance(results[1], Exception)
    assert not isinstance(results[2], Exception)

    def _snap(p: int):
        cfg = load_config().with_overrides({"write.parallelism": p})
        db = MindDB(tmp_path / f"p{p}.mind", dim=cfg.storage.vector_dim)
        store = FrameStore(db, embedder=MockEmbedder(dim=db.dim), chunking=cfg.chunking)
        kg = KnowledgeGraph(db)
        ingest_conversation(
            load_locomo(FIXTURE)[0],
            facade=FakeExtractors(banner=False),
            scanner=InjectionScanner(cfg.safety),
            store=store,
            kg=kg,
            cfg=cfg,
        )
        return db.conn.execute(
            "SELECT id, frame_type, gop_id, content FROM memory_frames ORDER BY id"
        ).fetchall()

    assert _snap(1) == _snap(4)


def test_record_then_replay_extraction_cache_reproduces_db(tmp_path: Path) -> None:
    cache = str(tmp_path / "extraction.jsonl")
    conv = Conversation(
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
    cfg = load_config()
    db_a = MindDB(tmp_path / "a.mind", dim=cfg.storage.vector_dim)
    store_a = FrameStore(db_a, embedder=MockEmbedder(dim=db_a.dim), chunking=cfg.chunking)
    rec = CachedExtractors(FakeExtractors(banner=False), path=cache, mode="record")
    ingest_conversation(
        conv,
        facade=rec,
        scanner=InjectionScanner(cfg.safety),
        store=store_a,
        kg=KnowledgeGraph(db_a),
        cfg=cfg,
    )

    db_b = MindDB(tmp_path / "b.mind", dim=cfg.storage.vector_dim)
    store_b = FrameStore(db_b, embedder=MockEmbedder(dim=db_b.dim), chunking=cfg.chunking)
    inner = FakeExtractors(banner=False)
    rep = CachedExtractors(inner, path=cache, mode="replay")
    ingest_conversation(
        conv,
        facade=rep,
        scanner=InjectionScanner(cfg.safety),
        store=store_b,
        kg=KnowledgeGraph(db_b),
        cfg=cfg,
    )

    q = (
        "SELECT frame_type, gop_id, content, content_hash, event_time "
        "FROM memory_frames ORDER BY id"
    )
    assert db_a.conn.execute(q).fetchall() == db_b.conn.execute(q).fetchall()
    assert rep.cache_misses == 0 and sum(inner.calls.values()) == 0
