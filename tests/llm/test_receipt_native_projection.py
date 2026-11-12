"""Authored offline calls through the native consolidation facade."""

from contextlib import contextmanager
from typing import Literal

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from gluomem.config.app_config import AppConfig
from gluomem.llm.assertion_scope import (
    AssertionScopedContext,
    AssertionScopedIdentityContext,
    AssertionScopedMeaning,
    AssertionScopedProjection,
)
from gluomem.llm.modules import DspyExtractors
from gluomem.llm.observed_identity import PropertyIdentityProposal
from gluomem.llm.outputs import ConsolidationProjection, ExtractionError
from gluomem.llm.router import LMRouter
from gluomem.llm.semantic import AssertionKey, SemanticSourceWitness
from gluomem.write.receipt_coherence import ReceiptCoherentProjection

POLICY = "same-event-corroboration-v1"
PRODUCER = "observed-assertion-scope-v4"


class AuthoredRouter(LMRouter):
    def __init__(self):
        self.lm = DummyLM([{"decision": {"action": "CANDIDATE"}}], adapter=dspy.JSONAdapter())

    @contextmanager
    def bind(self, task):
        assert task == "consolidate"
        with dspy.context(lm=self.lm, adapter=dspy.JSONAdapter()):
            yield


def native(policy):
    write = {
        "writer_contract": "semantic-state-writer-v4",
        "semantic_producer_contract": PRODUCER,
        "source_capture_policy": "attributed-interaction",
    }
    if policy:
        write["semantic_reuse_policy"] = policy
    cfg = AppConfig(
        dspy={
            "model": "mock",
            "cache": {"disk": False},
            "adapter": "json",
            "schema_mode": "required",
        },
        write=write,
        _env_file=None,
        _secrets_dir=None,
    )
    router = AuthoredRouter()
    return DspyExtractors(cfg, router=router), router


def projection(
    cls,
    actions=("ADD", "CANDIDATE"),
    *,
    assertion_act: Literal["direct_assertion", "unresolved"] = "direct_assertion",
):
    meaning = AssertionScopedMeaning(
        key=AssertionKey(subject="cabinet", predicate="colour", perspective="user"),
        value_key="blue",
        independently_versioned=True,
        refines=None,
        assertion_act=assertion_act,
    )
    context = AssertionScopedContext(
        kind="fact",
        source_role="user",
        observed_at=None,
        effective_at=None,
        meaning=meaning,
        witness=SemanticSourceWitness(
            tenant_id="t",
            principal_id="p",
            namespace="n",
            event_id="e",
            revision=1,
            source_ref="authored",
            message_id="m",
            source_role="user",
            observed_at=None,
            verbatim_span="The cabinet is blue.",
        ),
    )
    extra = {"feasible_untargeted": actions} if issubclass(cls, ReceiptCoherentProjection) else {}
    return cls(
        claim_index=0,
        targets=(),
        meaning=meaning,
        context=context,
        target_meanings=(),
        structural=ConsolidationProjection(claim_index=0, targets=()),
        identity_context=AssertionScopedIdentityContext(claim_index=0),
        proposal=PropertyIdentityProposal(
            relation="independent",
            assertion_relation="unrelated",
            subject_alignment="not_aligned",
            property_alignment="not_aligned",
        ),
        identity_call="empty-context",
        **extra,
    )


@pytest.mark.parametrize(
    "policy,cls", [(None, AssertionScopedProjection), (POLICY, ReceiptCoherentProjection)]
)
def test_native_empty_projection_accepts_only_configured_family(policy, cls):
    extractor, router = native(policy)
    menu = projection(cls)
    result = extractor.consolidate_claim_with_projection(
        "The cabinet is blue.",
        [],
        context=menu.context,
        projection=menu,
    )
    assert result.action == "ADD"
    assert router.lm.history == []


@pytest.mark.parametrize(
    "policy,cls", [(None, ReceiptCoherentProjection), (POLICY, AssertionScopedProjection)]
)
def test_native_rejects_cross_policy_family_before_send(policy, cls):
    extractor, router = native(policy)
    menu = projection(cls)
    with pytest.raises(ExtractionError, match="matching bound meanings"):
        extractor.consolidate_claim_with_projection(
            "The cabinet is blue.",
            [],
            context=menu.context,
            projection=menu,
        )
    assert router.lm.history == []


@pytest.mark.parametrize(
    "policy,cls,act",
    [
        (None, AssertionScopedProjection, "unresolved"),
        (POLICY, ReceiptCoherentProjection, "direct_assertion"),
    ],
)
def test_native_menu_without_add_runs_authored_typed_output(policy, cls, act):
    extractor, router = native(policy)
    menu = projection(cls, ("CANDIDATE",), assertion_act=act)
    result = extractor.consolidate_claim_with_projection(
        "The cabinet is blue.",
        [],
        context=menu.context,
        projection=menu,
    )
    assert result.action == "CANDIDATE"
    assert len(router.lm.history) == 1
    assert "feasible_untargeted" not in str(router.lm.history[0]["messages"])


@pytest.mark.parametrize(
    "policy,cls", [(None, AssertionScopedProjection), (POLICY, ReceiptCoherentProjection)]
)
def test_preparation_resolves_same_exact_family_as_native(policy, cls):
    from gluomem.write.observed_identity import ObservedProjectionState

    extractor, _ = native(policy)
    state = ObservedProjectionState(
        extractor,
        producer_contract=PRODUCER,
        event=object(),
        cfg=extractor._cfg,
    )
    assert state.projection_type is cls


def test_predecessor_projection_families_are_unchanged():
    from gluomem.llm.observed_identity import ObservedSemanticProjection
    from gluomem.llm.projection_contract import semantic_projection_type
    from gluomem.llm.semantic import SemanticProjection

    assert semantic_projection_type(None) is SemanticProjection
    for producer in (
        "observed-assertion-date-v1",
        "observed-assertion-date-v2",
        "observed-property-identity-v3",
    ):
        assert semantic_projection_type(producer) is ObservedSemanticProjection


@pytest.mark.parametrize(
    "producer,policy",
    [
        (None, POLICY),
        ("observed-property-identity-v3", POLICY),
        (PRODUCER, "unknown"),
        ("unknown", None),
    ],
)
def test_projection_family_resolution_fails_closed(producer, policy):
    from gluomem.llm.projection_contract import semantic_projection_type

    with pytest.raises(ValueError, match="Unsupported semantic projection"):
        semantic_projection_type(producer, policy)


def test_native_rejects_subclass_even_with_receipt_policy():
    class UntrustedProjection(ReceiptCoherentProjection):
        pass

    extractor, router = native(POLICY)
    menu = projection(UntrustedProjection)
    with pytest.raises(ExtractionError, match="matching bound meanings"):
        extractor.consolidate_claim_with_projection(
            "The cabinet is blue.",
            [],
            context=menu.context,
            projection=menu,
        )
    assert router.lm.history == []
