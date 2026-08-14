"""Frame-model ablation — the SAME observation sequence under ipb / i_only /
overwrite, switched purely by config (no code edit).

Scenario (fact lane only):
  1. "Alice lives in Boston."            -> first I (entities: alice+boston)
  2. "Alice enjoys hiking around Boston."-> same subject (overlap 2), no
     contradiction  -> ipb: P | i_only: I | overwrite: I
  3. "Alice moved from Boston to Seattle."-> same subject (overlap 2),
     CONTRADICTION -> ipb: new I + B + edge retired | i_only: plain I |
     overwrite: new I + base/P demoted to deprecated + edge retired
"""

from __future__ import annotations

from datetime import datetime

from gluomem.llm import (
    CognifyResult,
    ContradictionVerdict,
    ExtractedFact,
    KGEntity,
    KGRelation,
)
from gluomem.write.gate import InjectionScanner
from gluomem.write.ingest import Conversation, Session, Turn, ingest_conversation

_ALICE = KGEntity(entity_type="person", name="alice")
_BOSTON = KGEntity(entity_type="city", name="boston")
_SEATTLE = KGEntity(entity_type="city", name="seattle")

_F1 = "Alice lives in Boston."
_F2 = "Alice enjoys hiking around Boston."
_F3 = "Alice moved from Boston to Seattle."


def _facade(cls):
    return cls(
        facts={
            "session one": [ExtractedFact(text=_F1)],
            "session two": [ExtractedFact(text=_F2)],
            "session three": [ExtractedFact(text=_F3)],
        },
        cognify_map={
            _F1: CognifyResult(
                entities=[_ALICE, _BOSTON],
                relations=[
                    KGRelation(source="alice", target="boston", relation_type="LIVES_IN")
                ],
            ),
            _F2: CognifyResult(entities=[_ALICE, _BOSTON]),
            _F3: CognifyResult(
                entities=[_ALICE, _BOSTON, _SEATTLE],
                relations=[
                    KGRelation(source="alice", target="seattle", relation_type="LIVES_IN")
                ],
            ),
        },
        verdicts={
            "moved from Boston": ContradictionVerdict(
                is_contradiction=True, description="Alice moved cities"
            )
        },
    )


def _conv() -> Conversation:
    sessions = [
        Session(
            index=i,
            date_time=datetime(2023, 5, 8 + i),
            turns=[Turn(speaker="Alice", text=f"chatter in session {word}")],
        )
        for i, word in ((1, "one"), (2, "two"), (3, "three"))
    ]
    return Conversation(conv_id="c1", speakers=("Alice", "Bob"), sessions=sessions)


def _run(env, facade_cls, mode: str):
    cfg = env.cfg.with_overrides(
        {
            "write.frame_model": mode,
            "write.raw_turns.enabled": False,
            "write.lanes.event.enabled": False,
            "write.lanes.profile.enabled": False,
        }
    )
    facade = _facade(facade_cls)
    report = ingest_conversation(
        _conv(),
        facade=facade,
        scanner=InjectionScanner(cfg.safety),
        store=env.store,
        kg=env.kg,
        cfg=cfg,
    )
    return report, facade


def _fact_frames(env):
    rows = env.db.conn.execute(
        "SELECT id, frame_type, importance, content FROM memory_frames "
        "WHERE gop_id='c1:facts' ORDER BY id"
    ).fetchall()
    return [(r[1], r[2], r[3]) for r in rows]


def test_ipb_mode_emits_p_and_b(env, scripted_facade_cls) -> None:
    report, _ = _run(env, scripted_facade_cls, "ipb")
    assert report.i_frames == 2  # F1 + the new truth F3
    assert report.p_frames == 1  # F2 delta on F1
    assert report.b_frames == 1  # correction on F1
    frames = _fact_frames(env)
    types = [t for t, _, _ in frames]
    assert types == ["I", "P", "I", "B"]
    # B payload references base + new I.
    b_row = env.db.conn.execute(
        "SELECT id, base_frame_id FROM memory_frames "
        "WHERE gop_id='c1:facts' AND frame_type='B'"
    ).fetchone()
    correction = env.store.get_b_correction(int(b_row[0]))
    assert correction.description == "Alice moved cities"
    assert len(correction.references) == 2
    # The old LIVES_IN boston edge is retired; seattle edge valid.
    alice = env.kg.find_valid_entities("alice", env.cfg.kg)[0]
    valid = env.kg.find_valid_relations(alice.id, "LIVES_IN")
    assert len(valid) == 1
    assert report.retired_edges == 1


def test_i_only_mode_never_checks_contradiction(env, scripted_facade_cls) -> None:
    report, facade = _run(env, scripted_facade_cls, "i_only")
    assert report.i_frames == 3
    assert report.p_frames == 0 and report.b_frames == 0
    assert facade.calls["check_contradiction"] == 0
    assert [t for t, _, _ in _fact_frames(env)] == ["I", "I", "I"]


def test_overwrite_mode_demotes_superseded_cluster(env, scripted_facade_cls) -> None:
    report, _ = _run(env, scripted_facade_cls, "overwrite")
    assert report.p_frames == 0 and report.b_frames == 0
    assert report.i_frames == 3
    # In overwrite mode the F2 delta was admitted as a sibling I sharing the
    # subject, so the contradiction demotes the WHOLE stale cluster (F1 + F2).
    assert report.demoted == 2
    frames = _fact_frames(env)
    assert frames[0] == ("I", "deprecated", _F1)
    assert frames[1] == ("I", "deprecated", _F2)
    assert frames[2][1] == "normal"  # the new truth stays at lane importance
    assert report.retired_edges == 1


def test_mode_switch_is_pure_config_flip(env, scripted_facade_cls) -> None:
    """Same inputs, different write.frame_model -> different frame shapes."""
    report, _ = _run(env, scripted_facade_cls, "ipb")
    assert (report.p_frames, report.b_frames) == (1, 1)
