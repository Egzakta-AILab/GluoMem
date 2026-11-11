"""Pure predecessor contract snapshots; no DSPy, model, transport or service imports."""

from gluomem.config.app_config import AppConfig, WriteConfig
from gluomem.llm.assertion_scope import (
    AssertionScopedContext,
    AssertionScopedIdentityContext,
    AssertionScopedMeaning,
    scoped_claim_type,
)
from gluomem.llm.cache import _key
from gluomem.llm.observed_identity import (
    IdentityContext,
    identity_proposal_model,
    observed_claim_type,
)
from gluomem.llm.outputs import (
    ConsolidationDecision,
    ConsolidationProjection,
    ExtractedClaimSpec,
    ProjectedCandidate,
    ProjectedConsolidationDecision,
    projected_decision_model,
)
from gluomem.llm.semantic import (
    ClaimMeaning,
    SemanticClaimContext,
    SemanticClaimSpec,
    SemanticSourceWitness,
)

PRODUCERS = (
    "observed-assertion-date-v1",
    "observed-assertion-date-v2",
    "observed-property-identity-v3",
    "observed-assertion-scope-v4",
)


def predecessor_snapshot():
    result = {"modes": {}, "leaves": {}, "cache_keys": {}}
    modes = [
        ("default", None, None),
        ("bounded-v2", "bounded-writer-v2", None),
        ("action-target-v3", "action-target-writer-v3", None),
        ("semantic-v4", "semantic-state-writer-v4", None),
    ]
    modes += [(p, "semantic-state-writer-v4", p) for p in PRODUCERS]
    projection = ConsolidationProjection(
        claim_index=0,
        targets=(
            ProjectedCandidate(
                frame_id=7,
                content="The cabinet is blue.",
                allowed_actions=("NOOP", "SUPERSEDE", "REFINE"),
            ),
        ),
    )
    witness = SemanticSourceWitness(
        tenant_id="t",
        principal_id="p",
        namespace="n",
        event_id="e",
        revision=1,
        source_ref="authored:compat",
        message_id="m",
        source_role="user",
        observed_at=None,
        verbatim_span="The cabinet is blue.",
    )
    for label, writer, producer in modes:
        write = {}
        if writer:
            write.update(writer_contract=writer, source_capture_policy="attributed-interaction")
        if producer:
            write["semantic_producer_contract"] = producer
        cfg = AppConfig(
            dspy={"model": "mock", "adapter": "json", "schema_mode": "required"},
            write=write,
            _env_file=None,
            _secrets_dir=None,
        )
        assert "semantic_reuse_policy" not in cfg.write.model_dump(mode="json")
        row = {
            "config": cfg.model_dump(mode="json"),
            "menu": projection.model_dump(mode="json"),
            "decision_schema": projected_decision_model(
                projection, writer_contract=writer
            ).model_json_schema(),
        }
        if producer:
            meaning = {
                "key": {"subject": "cabinet", "predicate": "colour", "perspective": "user"},
                "value_key": "blue",
                "independently_versioned": True,
                "refines": None,
            }
            if producer == PRODUCERS[-1]:
                meaning["assertion_act"] = "direct_assertion"
                context = AssertionScopedContext(
                    kind="fact",
                    source_role="user",
                    observed_at=None,
                    effective_at=None,
                    meaning=AssertionScopedMeaning.model_validate(meaning),
                    witness=witness,
                )
                observed = AssertionScopedIdentityContext(claim_index=0)
            else:
                context = SemanticClaimContext(
                    kind="fact",
                    source_role="user",
                    observed_at=None,
                    effective_at=None,
                    meaning=ClaimMeaning.model_validate(meaning),
                    witness=witness,
                )
                observed = IdentityContext(claim_index=0)
            row["identity_schema"] = identity_proposal_model(
                observed, context, producer_contract=producer
            ).model_json_schema()
            row["identity_context"] = observed.model_dump(mode="json")
        result["modes"][label] = row
    for cls in (
        ExtractedClaimSpec,
        SemanticClaimSpec,
        observed_claim_type(),
        scoped_claim_type(),
        ConsolidationDecision,
        ProjectedConsolidationDecision,
    ):
        result["leaves"][cls.__name__] = cls.model_json_schema()
    for method in (
        "extract_event_claims",
        "consolidate_claim",
        "consolidate_claim_with_projection",
        "align_claim_identity",
    ):
        result["cache_keys"][method] = _key(method, "Authored frozen input.", "{}")
    return result


def assert_optin_configuration():
    import pytest

    assert "semantic_reuse_policy" not in WriteConfig().model_dump()
    write = {
        "semantic_reuse_policy": "same-event-corroboration-v1",
        "writer_contract": "semantic-state-writer-v4",
        "semantic_producer_contract": PRODUCERS[-1],
        "source_capture_policy": "attributed-interaction",
    }
    assert (
        WriteConfig.model_validate(write).model_dump()["semantic_reuse_policy"]
        == "same-event-corroboration-v1"
    )
    for change in (
        {"semantic_producer_contract": None},
        {"semantic_producer_contract": PRODUCERS[0]},
        {"writer_contract": None},
        {"semantic_reuse_policy": "unknown"},
        {"source_capture_policy": None},
    ):
        with pytest.raises(ValueError):
            # Raw invalid configurations must still reach the production validator unchanged.
            WriteConfig.model_validate({**write, **change})


def test_predecessor_contracts_and_explicit_policy():
    assert len(predecessor_snapshot()["modes"]) == 8
    assert_optin_configuration()
