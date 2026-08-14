"""kg_edge_conflict contradiction trigger.

The structural miss this trigger closes: at ``min_entity_overlap=2`` an update-type
fact shares only ONE entity with the stale truth ({bob, seattle} vs stored
{bob, boston} — the old and new city differ *by construction*), so the
entity_overlap trigger never nominates the stale frame and the contradiction
detector never runs. The same-family KG edge conflict (a fresh MOVED_TO from
a source that still holds a valid LIVES_IN to a different target) is exact.

Scenario (fact lane only, default min_entity_overlap=2):
  1. "Bob lives in Boston."   -> I (entities: bob+boston, edge LIVES_IN)
  2. "Bob moved to Seattle."  -> shares only `bob` (overlap 1 < 2):
     entity_overlap: plain I, NO contradiction check (the documented miss)
     kg_edge_conflict + LIVES_IN/MOVED_TO family: contradiction -> new I + B
     on the stale frame + the boston edge retired; P-frames stay 0.
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

_BOB = KGEntity(entity_type="person", name="bob")
_BOSTON = KGEntity(entity_type="city", name="boston")
_SEATTLE = KGEntity(entity_type="city", name="seattle")

_F1 = "Bob lives in Boston."
_F2 = "Bob moved to Seattle."

_FAMILIES = [["LIVES_IN", "MOVED_TO", "RESIDES_IN"]]


def _facade(cls):
    return cls(
        facts={
            "session one": [ExtractedFact(text=_F1)],
            "session two": [ExtractedFact(text=_F2)],
        },
        cognify_map={
            _F1: CognifyResult(
                entities=[_BOB, _BOSTON],
                relations=[
                    KGRelation(source="bob", target="boston", relation_type="LIVES_IN")
                ],
            ),
            _F2: CognifyResult(
                entities=[_BOB, _SEATTLE],
                relations=[
                    KGRelation(source="bob", target="seattle", relation_type="MOVED_TO")
                ],
            ),
        },
        verdicts={
            "moved to Seattle": ContradictionVerdict(
                is_contradiction=True, description="Bob moved cities"
            )
        },
    )


def _conv() -> Conversation:
    sessions = [
        Session(
            index=i,
            date_time=datetime(2023, 5, 8 + i),
            turns=[Turn(speaker="Bob", text=f"chatter in session {word}")],
        )
        for i, word in ((1, "one"), (2, "two"))
    ]
    return Conversation(conv_id="c1", speakers=("Bob", "Ann"), sessions=sessions)


def _run(env, facade_cls, overrides: dict):
    cfg = env.cfg.with_overrides(
        {
            "write.raw_turns.enabled": False,
            "write.lanes.event.enabled": False,
            "write.lanes.profile.enabled": False,
            **overrides,
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


def test_entity_overlap_trigger_misses_single_shared_entity_update(
    env, scripted_facade_cls
) -> None:
    """The documented baseline miss: overlap 1 < 2 means no candidate, no
    contradiction call, no B-frame — old and new truth coexist live."""
    report, facade = _run(env, scripted_facade_cls, {})
    assert facade.calls["check_contradiction"] == 0
    assert report.b_frames == 0 and report.p_frames == 0
    assert report.i_frames == 2
    bob = env.kg.find_valid_entities("bob", env.cfg.kg)[0]
    assert len(env.kg.find_valid_relations(bob.id, "LIVES_IN")) == 1  # stale, live


def test_kg_edge_conflict_trigger_catches_update(env, scripted_facade_cls) -> None:
    report, facade = _run(
        env,
        scripted_facade_cls,
        {
            "write.subject.trigger": "kg_edge_conflict",
            "kg.predicate_families": _FAMILIES,
        },
    )
    # The stale frame was nominated and checked exactly once.
    assert facade.calls["check_contradiction"] == 1
    assert facade.contradiction_inputs == [(_F2, _F1)]
    # ipb contradiction path: new truth I + B on the stale frame; and the
    # trigger never feeds P-attachment, so P stays 0.
    assert report.i_frames == 2 and report.b_frames == 1 and report.p_frames == 0
    b_row = env.db.conn.execute(
        "SELECT id FROM memory_frames WHERE gop_id='c1:facts' AND frame_type='B'"
    ).fetchone()
    correction = env.store.get_b_correction(int(b_row[0]))
    assert correction.description == "Bob moved cities"
    assert len(correction.references) == 2
    # Family-aware retirement: the fresh MOVED_TO retired the stale LIVES_IN.
    bob = env.kg.find_valid_entities("bob", env.cfg.kg)[0]
    assert env.kg.find_valid_relations(bob.id, "LIVES_IN") == []
    assert len(env.kg.find_valid_relations(bob.id, "MOVED_TO")) == 1
    assert report.retired_edges == 1


def test_kg_edge_conflict_needs_predicate_family(env, scripted_facade_cls) -> None:
    """Without the LIVES_IN/MOVED_TO family the exact-type edge check finds no
    conflict (MOVED_TO has no prior same-type edge) — families are the lever."""
    report, facade = _run(
        env, scripted_facade_cls, {"write.subject.trigger": "kg_edge_conflict"}
    )
    assert facade.calls["check_contradiction"] == 0
    assert report.b_frames == 0


def test_both_trigger_unions_candidates(env, scripted_facade_cls) -> None:
    """`both` catches the update too (edge candidates fill in when the
    entity-overlap base is absent)."""
    report, facade = _run(
        env,
        scripted_facade_cls,
        {
            "write.subject.trigger": "both",
            "kg.predicate_families": _FAMILIES,
        },
    )
    assert facade.calls["check_contradiction"] == 1
    assert report.b_frames == 1
