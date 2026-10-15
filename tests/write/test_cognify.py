"""Cognify entity/relation resolution, bitemporal retirement, and contradiction triggers."""

from __future__ import annotations

from datetime import datetime

import pytest

from gluomem.config.app_config import load_config
from gluomem.llm import (
    CognifyResult,
    ContradictionVerdict,
    ExtractedFact,
    FakeExtractors,
    KGEntity,
    KGRelation,
)
from gluomem.write.cognify import cognify_text, retire_conflicting_relations
from gluomem.write.conversation import Conversation, Session, Turn, ingest_conversation
from gluomem.write.gate import InjectionScanner


def _facade(*results: CognifyResult) -> FakeExtractors:
    return FakeExtractors(cognify_results=list(results), banner=False)


def _lives_in(city: str) -> CognifyResult:
    return CognifyResult(
        entities=[
            KGEntity(entity_type="person", name="alice"),
            KGEntity(entity_type="city", name=city),
        ],
        relations=[KGRelation(source="alice", target=city, relation_type="LIVES_IN")],
    )


def test_entity_resolution_alias_and_implicit_endpoints(env) -> None:
    res = _lives_in("boston")
    facade = _facade(res, res)
    g1 = cognify_text("t", facade=facade, kg=env.kg, cfg=env.cfg)
    g2 = cognify_text("t", facade=facade, kg=env.kg, cfg=env.cfg)
    assert g1.entity_ids == g2.entity_ids
    assert g1.relation_ids == g2.relation_ids
    assert len(env.kg.find_valid_entities("alice", env.cfg.kg)) == 1

    existing = env.kg.create_entity("technology", "kubernetes")
    g_alias = cognify_text(
        "t",
        facade=_facade(CognifyResult(entities=[KGEntity(entity_type="technology", name="k8s")])),
        kg=env.kg,
        cfg=env.cfg,
    )
    assert g_alias.entity_ids == [existing.id]

    g_imp = cognify_text(
        "t",
        facade=_facade(
            CognifyResult(
                entities=[KGEntity(entity_type="person", name="alice")],
                relations=[KGRelation(source="alice", target="acme", relation_type="WORKS_AT")],
            )
        ),
        kg=env.kg,
        cfg=env.cfg,
    )
    assert len(g_imp.relation_ids) == 1
    acme = env.kg.find_valid_entities("acme", env.cfg.kg)
    assert len(acme) == 1 and acme[0].entity_type == "unknown"


def test_cognify_disabled_is_noop(env) -> None:
    cfg = env.cfg.with_overrides({"write.cognify.enabled": False})
    facade = _facade(_lives_in("boston"))
    g = cognify_text("t", facade=facade, kg=env.kg, cfg=cfg)
    assert g.entity_ids == [] and g.relation_ids == []
    assert facade.calls["cognify"] == 0


def test_bitemporal_retirement_reassertion_and_valid_time_ordering(env) -> None:
    def _rel_on(city: str, when: str):
        g = cognify_text(
            f"t-{city}-{when}",
            facade=_facade(_lives_in(city)),
            kg=env.kg,
            cfg=env.cfg,
            valid_from=when,
        )
        return env.kg.get_relation(g.relation_ids[0]), g

    boston, _ = _rel_on("boston", "2023-01-01T00:00:00")
    seattle, g_sea = _rel_on("seattle", "2023-03-01T00:00:00")
    assert retire_conflicting_relations(g_sea.relations, kg=env.kg) == [boston.id]
    assert env.kg.get_relation(boston.id).valid_to == "2023-03-01T00:00:00"

    # Re-asserting retired triple reopens edge and preserves intermediate gap.
    again = cognify_text(
        "t-back",
        facade=_facade(_lives_in("boston")),
        kg=env.kg,
        cfg=env.cfg,
        valid_from="2023-07-01T00:00:00",
    )
    assert again.relation_ids == [boston.id]
    retire_conflicting_relations(again.relations, kg=env.kg)

    def city_at(iso: str) -> set[int]:
        return {r.target_id for r in env.kg.as_of(iso).relations if r.relation_type == "LIVES_IN"}

    assert city_at("2023-02-01") == {boston.target_id}
    assert city_at("2023-04-01") == {seattle.target_id}
    assert city_at("2023-08-01") == {boston.target_id}


def test_late_recorded_older_fact_does_not_retire_newer_one(env) -> None:
    g_sea = cognify_text(
        "t-sea",
        facade=_facade(_lives_in("seattle")),
        kg=env.kg,
        cfg=env.cfg,
        valid_from="2023-05-01T00:00:00",
    )
    g_bos = cognify_text(
        "t-bos",
        facade=_facade(_lives_in("boston")),
        kg=env.kg,
        cfg=env.cfg,
        valid_from="2023-01-01T00:00:00",
    )
    seattle_id, boston_id = g_sea.relation_ids[0], g_bos.relation_ids[0]
    assert retire_conflicting_relations(g_bos.relations, kg=env.kg) == [boston_id]
    assert env.kg.get_relation(seattle_id).valid_to is None
    assert env.kg.get_relation(boston_id).valid_to == "2023-05-01T00:00:00"


def test_entity_type_modes_off_retype_and_drop(env) -> None:
    assert load_config().kg.entity_type_mode == "retype"
    polluted = CognifyResult(
        entities=[
            KGEntity(entity_type="person", name="alice"),
            KGEntity(entity_type="coffee scene", name="the coffee scene"),
        ],
        relations=[
            KGRelation(source="alice", target="the coffee scene", relation_type="ENJOYS"),
            KGRelation(source="alice", target="golden retriever puppy", relation_type="HAS_PET"),
        ],
    )

    cfg_off = env.cfg.with_overrides({"kg.entity_type_mode": "off"})
    g_off = cognify_text("t", facade=_facade(polluted), kg=env.kg, cfg=cfg_off)
    assert len(g_off.entity_ids) == 2 and len(g_off.relation_ids) == 2

    cfg_retype = env.cfg.with_overrides({"kg.entity_type_mode": "retype"})
    g_ret = cognify_text("t2", facade=_facade(polluted), kg=env.kg, cfg=cfg_retype)
    assert len(g_ret.entity_ids) == 2 and len(g_ret.relation_ids) == 2

    acme = env.kg.create_entity("organization", "acme")
    dup_junk = CognifyResult(
        entities=[
            KGEntity(entity_type="person", name="alice"),
            KGEntity(entity_type="vibes", name="acme"),
        ],
        relations=[KGRelation(source="alice", target="acme", relation_type="WORKS_AT")],
    )
    cfg_drop = env.cfg.with_overrides({"kg.entity_type_mode": "drop"})
    g_drop = cognify_text("t3", facade=_facade(dup_junk), kg=env.kg, cfg=cfg_drop)
    assert len(g_drop.relation_ids) == 1
    assert env.kg.get_relation(g_drop.relation_ids[0]).target_id == acme.id


_BOB = KGEntity(entity_type="person", name="bob")
_BOSTON = KGEntity(entity_type="city", name="boston")
_SEATTLE = KGEntity(entity_type="city", name="seattle")
_F1 = "Bob lives in Boston."
_F2 = "Bob moved to Seattle."
_FAMILIES = [["LIVES_IN", "MOVED_TO", "RESIDES_IN"]]


@pytest.mark.parametrize(
    ("overrides", "expected_checks", "expected_b", "expected_retired"),
    [
        ({}, 0, 0, 0),
        ({"write.subject.trigger": "kg_edge_conflict"}, 0, 0, 0),
        (
            {
                "write.subject.trigger": "kg_edge_conflict",
                "kg.predicate_families": _FAMILIES,
            },
            1,
            1,
            1,
        ),
        (
            {
                "write.subject.trigger": "both",
                "kg.predicate_families": _FAMILIES,
            },
            1,
            1,
            1,
        ),
    ],
)
def test_contradiction_triggers(
    env,
    scripted_facade_cls,
    overrides: dict,
    expected_checks: int,
    expected_b: int,
    expected_retired: int,
) -> None:
    cfg = env.cfg.with_overrides(
        {
            "write.raw_turns.enabled": False,
            "write.lanes.event.enabled": False,
            "write.lanes.profile.enabled": False,
            **overrides,
        }
    )
    facade = scripted_facade_cls(
        facts={
            "session one": [ExtractedFact(text=_F1)],
            "session two": [ExtractedFact(text=_F2)],
        },
        cognify_map={
            _F1: CognifyResult(
                entities=[_BOB, _BOSTON],
                relations=[KGRelation(source="bob", target="boston", relation_type="LIVES_IN")],
            ),
            _F2: CognifyResult(
                entities=[_BOB, _SEATTLE],
                relations=[KGRelation(source="bob", target="seattle", relation_type="MOVED_TO")],
            ),
        },
        verdicts={
            "moved to Seattle": ContradictionVerdict(
                is_contradiction=True, description="Bob moved cities"
            )
        },
    )
    conv = Conversation(
        conv_id="c1",
        speakers=("Bob", "Ann"),
        sessions=[
            Session(
                index=i,
                date_time=datetime(2023, 5, 8 + i),
                turns=[Turn(speaker="Bob", text=f"chatter in session {w}")],
            )
            for i, w in ((1, "one"), (2, "two"))
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
    assert facade.calls["check_contradiction"] == expected_checks
    assert report.b_frames == expected_b
    assert report.retired_edges == expected_retired
