"""V4's DSPy-free contract. Full authored native workflow is a separate offline gate."""

import pytest
from pydantic import ValidationError

from gluomem.config.app_config import AppConfig
from gluomem.llm.outputs import ProjectedCandidate, projected_decision_model
from gluomem.llm.semantic import (
    SEMANTIC_WRITER_CONTRACT,
    AssertionKey,
    ClaimMeaning,
    SemanticClaimSpec,
    SemanticProjection,
    TargetMeaning,
    permits,
    source_witness,
)
from gluomem.storage.scope import MemoryScope
from gluomem.write.events import SourceEvent, SourceEventMessage


def proposal(value="old", *, predicate="state", anchor=None):
    return ClaimMeaning(
        key=AssertionKey(subject="authored-subject", predicate=predicate, perspective="user"),
        value_key=value,
        independently_versioned=anchor is None,
        refines=anchor,
    )


def test_v4_is_explicit_and_keeps_unknown_legacy_unknown():
    cfg = AppConfig(
        dspy={"model": "mock", "cache": {"disk": False}},
        write={
            "writer_contract": SEMANTIC_WRITER_CONTRACT,
            "source_capture_policy": "attributed-interaction",
        },
    )
    assert cfg.write.writer_contract == SEMANTIC_WRITER_CONTRACT
    legacy = AppConfig(dspy={"model": "mock"})
    assert "writer_contract" not in legacy.write.model_dump()
    assert not permits("SUPERSEDE", proposal("new"), None)


def test_independence_and_explicit_anchor_are_not_kind_date_or_topic_similarity():
    root = proposal()
    successor = proposal("new")
    detail = proposal("qualifier", predicate="detail", anchor=root.key)
    assert not permits("REFINE", successor, root)
    assert permits("TRANSITION", successor, root)
    assert permits("REFINE", detail, root)
    assert not permits("SUPERSEDE", detail, root)
    assert not permits("NOOP", successor, root)
    with pytest.raises(ValidationError):
        ClaimMeaning(key=root.key, value_key="detail", independently_versioned=False, refines=None)
    with pytest.raises(ValidationError):
        # Raw malformed input deliberately omits the required meaning field.
        SemanticClaimSpec.model_validate({"text": "A legacy claim has no meaning proposal."})


def test_v4_unknown_target_menu_still_allows_explicit_add_not_a_targeted_repair():
    menu = SemanticProjection(
        claim_index=0,
        meaning=proposal("new"),
        targets=(ProjectedCandidate(frame_id=7, content="Unknown legacy state."),),
        target_meanings=(TargetMeaning(frame_id=7, meaning=None),),
    )
    model = projected_decision_model(menu, writer_contract=SEMANTIC_WRITER_CONTRACT)
    assert model.model_validate({"action": "ADD"}).target_frame_id is None
    with pytest.raises(ValueError, match="outside this claim's legal projection"):
        model.model_validate({"action": "SUPERSEDE", "target_frame_id": 7})


def test_proposal_witness_is_literal_scoped_and_role_bound_not_an_entailment_certificate():
    scope = MemoryScope("authored-tenant", "authored-principal", "authored-namespace")
    text = "An exact authored source statement."
    event = SourceEvent(
        event_id="authored",
        source_ref="authored/source",
        scope=scope,
        capture_policy="attributed-interaction",
        messages=[SourceEventMessage(message_id="m1", role="user", content=text)],
    )
    spec = SemanticClaimSpec(text=text, meaning=proposal(), message_id="m1", verbatim_span=text)
    witness = source_witness(event, spec)
    assert witness.namespace == scope.namespace and witness.verbatim_span == text
    assert witness.source_role == "user" and witness.observed_at is None
    wrong_role = event.model_copy(
        update={"messages": [SourceEventMessage(message_id="m1", role="assistant", content=text)]}
    )
    with pytest.raises(ValueError, match="perspective"):
        source_witness(wrong_role, spec)
