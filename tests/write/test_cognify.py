"""Cognify resolution tests — idempotent entities, dup-triple recovery, retirement."""

from __future__ import annotations

from gluomem.llm import CognifyResult, FakeExtractors, KGEntity, KGRelation
from gluomem.write.cognify import cognify_text, retire_conflicting_relations


def _facade(*results: CognifyResult) -> FakeExtractors:
    return FakeExtractors(cognify_results=list(results), banner=False)


def _lives_in(city: str) -> CognifyResult:
    return CognifyResult(
        entities=[
            KGEntity(entity_type="person", name="alice"),
            KGEntity(entity_type="city", name=city),
        ],
        relations=[
            KGRelation(source="alice", target=city, relation_type="LIVES_IN")
        ],
    )


def test_entity_resolution_idempotent(env) -> None:
    result = _lives_in("boston")
    facade = _facade(result, result)
    g1 = cognify_text("t", facade=facade, kg=env.kg, cfg=env.cfg)
    g2 = cognify_text("t", facade=facade, kg=env.kg, cfg=env.cfg)
    assert g1.entity_ids == g2.entity_ids
    assert g1.relation_ids == g2.relation_ids  # dup triple recovered, same edge
    assert len(env.kg.find_valid_entities("alice", env.cfg.kg)) == 1


def test_alias_resolves_to_existing_entity(env) -> None:
    existing = env.kg.create_entity("technology", "kubernetes")
    facade = _facade(
        CognifyResult(entities=[KGEntity(entity_type="technology", name="k8s")])
    )
    g = cognify_text("t", facade=facade, kg=env.kg, cfg=env.cfg)
    assert g.entity_ids == [existing.id]


def test_relation_endpoint_created_when_not_listed_as_entity(env) -> None:
    facade = _facade(
        CognifyResult(
            entities=[KGEntity(entity_type="person", name="alice")],
            relations=[
                KGRelation(source="alice", target="acme", relation_type="WORKS_AT")
            ],
        )
    )
    g = cognify_text("t", facade=facade, kg=env.kg, cfg=env.cfg)
    assert len(g.relation_ids) == 1
    acme = env.kg.find_valid_entities("acme", env.cfg.kg)
    assert len(acme) == 1 and acme[0].entity_type == "unknown"


def test_retire_conflicting_relations_functional_edge(env) -> None:
    old = cognify_text(
        "t", facade=_facade(_lives_in("boston")), kg=env.kg, cfg=env.cfg
    )
    new = cognify_text(
        "t2", facade=_facade(_lives_in("seattle")), kg=env.kg, cfg=env.cfg
    )
    retired = retire_conflicting_relations(new.relations, kg=env.kg)
    assert retired == old.relation_ids
    # Retire-not-delete: the row survives with valid_to set.
    old_rel = env.kg.get_relation(old.relation_ids[0])
    assert old_rel.valid_to is not None
    # The new edge stays valid.
    assert env.kg.get_relation(new.relation_ids[0]).valid_to is None


def test_reasserted_retired_triple_reopens_edge(env) -> None:
    """boston -> seattle -> boston again. The third assertion hits the
    UNIQUE triple whose row was RETIRED — recovery must reopen it, not
    crash (real LoCoMo re-mentions retired facts)."""
    first = cognify_text(
        "t", facade=_facade(_lives_in("boston")), kg=env.kg, cfg=env.cfg
    )
    moved = cognify_text(
        "t2", facade=_facade(_lives_in("seattle")), kg=env.kg, cfg=env.cfg
    )
    retire_conflicting_relations(moved.relations, kg=env.kg)
    assert env.kg.get_relation(first.relation_ids[0]).valid_to is not None

    back = cognify_text(
        "t3", facade=_facade(_lives_in("boston")), kg=env.kg, cfg=env.cfg
    )
    # Same row identity, valid again.
    assert back.relation_ids == first.relation_ids
    assert env.kg.get_relation(back.relation_ids[0]).valid_to is None


def test_relation_backdating_via_valid_from(env) -> None:
    g = cognify_text(
        "t",
        facade=_facade(_lives_in("boston")),
        kg=env.kg,
        cfg=env.cfg,
        valid_from="2023-05-07T00:00:00",
    )
    rel = env.kg.get_relation(g.relation_ids[0])
    assert rel.valid_from == "2023-05-07T00:00:00"


def test_cognify_disabled_is_noop(env) -> None:
    cfg = env.cfg.with_overrides({"write.cognify.enabled": False})
    facade = _facade(_lives_in("boston"))
    g = cognify_text("t", facade=facade, kg=env.kg, cfg=cfg)
    assert g.entity_ids == [] and g.relation_ids == []
    assert facade.calls["cognify"] == 0  # facade never invoked


# --- entity-type pollution gate (kg.entity_type_mode) -----------------------

# Alice (allowlisted 'person') enjoys a junk-typed entity, and has a relation to
# an endpoint never listed as an entity (resolves untyped 'unknown').
def _polluted() -> CognifyResult:
    return CognifyResult(
        entities=[
            KGEntity(entity_type="person", name="alice"),
            KGEntity(entity_type="coffee scene", name="the coffee scene"),
        ],
        relations=[
            KGRelation(source="alice", target="the coffee scene", relation_type="ENJOYS"),
            KGRelation(
                source="alice", target="golden retriever puppy", relation_type="HAS_PET"
            ),
        ],
    )


def test_entity_type_off_keeps_junk_unchanged(env) -> None:
    """`off` applies no hygiene: junk type + untyped endpoint pass through."""
    cfg = env.cfg.with_overrides({"kg.entity_type_mode": "off"})
    g = cognify_text("t", facade=_facade(_polluted()), kg=env.kg, cfg=cfg)
    assert len(g.entity_ids) == 2 and len(g.relation_ids) == 2
    junk = env.kg.find_valid_entities("the coffee scene", cfg.kg)
    assert junk and junk[0].entity_type == "coffee scene"
    puppy = env.kg.find_valid_entities("golden retriever puppy", cfg.kg)
    assert puppy and puppy[0].entity_type == "unknown"


def test_entity_type_default_is_retype() -> None:
    """The shipped default is `retype`."""
    from gluomem.config.app_config import load_config

    assert load_config().kg.entity_type_mode == "retype"


def test_entity_type_retype_relabels_but_keeps_graph(env) -> None:
    cfg = env.cfg.with_overrides({"kg.entity_type_mode": "retype"})  # explicit, though default
    g = cognify_text("t", facade=_facade(_polluted()), kg=env.kg, cfg=cfg)
    # Node + both relations survive; only the junk type is relabelled.
    assert len(g.entity_ids) == 2 and len(g.relation_ids) == 2
    junk = env.kg.find_valid_entities("the coffee scene", cfg.kg)
    assert junk and junk[0].entity_type == "other"
    # Allowlisted entity untouched.
    alice = env.kg.find_valid_entities("alice", cfg.kg)
    assert alice and alice[0].entity_type == "person"


def test_entity_type_drop_prunes_entity_and_its_relations(env) -> None:
    cfg = env.cfg.with_overrides({"kg.entity_type_mode": "drop"})
    g = cognify_text("t", facade=_facade(_polluted()), kg=env.kg, cfg=cfg)
    # Only allowlisted Alice survives; both relations pruned (one endpoint is the
    # dropped junk entity, the other an unlisted untyped endpoint).
    assert len(g.entity_ids) == 1 and len(g.relation_ids) == 0
    assert not env.kg.find_valid_entities("the coffee scene", cfg.kg)
    assert not env.kg.find_valid_entities("golden retriever puppy", cfg.kg)
    alice = env.kg.find_valid_entities("alice", cfg.kg)
    assert alice and alice[0].entity_type == "person"


def test_entity_type_drop_relinks_relation_to_existing_allowlisted(env) -> None:
    """A junk-typed duplicate of an existing GOOD node is dropped, but a relation
    naming it reconnects to the real node (find-by-name, no type filter)."""
    acme = env.kg.create_entity("organization", "acme")
    result = CognifyResult(
        entities=[
            KGEntity(entity_type="person", name="alice"),
            KGEntity(entity_type="vibes", name="acme"),  # junk-typed dup of acme
        ],
        relations=[KGRelation(source="alice", target="acme", relation_type="WORKS_AT")],
    )
    cfg = env.cfg.with_overrides({"kg.entity_type_mode": "drop"})
    g = cognify_text("t", facade=_facade(result), kg=env.kg, cfg=cfg)
    assert len(g.relation_ids) == 1  # relation kept, wired to the real acme
    rel = env.kg.get_relation(g.relation_ids[0])
    assert rel.target_id == acme.id
    # No junk-typed 'vibes' duplicate was created.
    assert all(
        e.entity_type != "vibes" for e in env.kg.find_valid_entities("acme", cfg.kg)
    )
