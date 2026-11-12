"""Untrusted proposal references must be legal in native schema and parser."""

import jsonschema
import pytest

from gluomem.llm.observed_identity import (
    IdentityContext,
    IdentityTarget,
    identity_proposal_model,
)
from gluomem.llm.outputs import ProjectedCandidate
from gluomem.llm.semantic import SemanticClaimContext, SemanticSourceWitness
from tests.contracts.test_semantic_backend_parity import assertion


def inputs():
    spec = assertion("Nila's compass is blue.", "compass-hue", "blue")
    witness = SemanticSourceWitness(
        tenant_id="t",
        principal_id="p",
        namespace="n",
        event_id="e",
        revision=1,
        source_ref="s",
        message_id="m",
        source_role="user",
        observed_at=None,
        verbatim_span=spec.text,
    )
    context = SemanticClaimContext(
        kind="fact",
        source_role="user",
        observed_at=None,
        effective_at=None,
        meaning=spec.meaning,
        witness=witness,
    )
    projection = IdentityContext(
        claim_index=2,
        targets=(
            IdentityTarget(
                target=ProjectedCandidate(
                    frame_id=7, content="Nila's compass is red.", allowed_actions=("SUPERSEDE",)
                ),
                meaning=assertion("Nila's compass is red.", "compass-colour", "red").meaning,
                receipt_tokens=("opaque",),
            ),
        ),
    )
    return projection, context


def test_native_identity_schema_rejects_incompatible_shape():
    projection, context = inputs()
    model = identity_proposal_model(projection, context)
    invalid = {"relation": "refines_assertion", "target_frame_id": 7, "target_claim_index": None}
    with pytest.raises(ValueError, match="dependent detail"):
        model.model_validate(invalid)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, model.model_json_schema())


@pytest.mark.parametrize(
    "bad",
    [
        {"relation": "same_assertion", "target_frame_id": 99},
        {"relation": "same_assertion", "target_claim_index": 2},
        {"relation": "same_assertion", "target_frame_id": 7, "target_claim_index": 0},
        {"relation": "independent", "target_frame_id": 7},
        {"relation": "unresolved", "target_claim_index": 0},
    ],
)
def test_illegal_references_are_not_repaired(bad):
    projection, context = inputs()
    with pytest.raises(ValueError):
        identity_proposal_model(projection, context).model_validate(bad)


def test_exact_named_compatible_proposal_is_fallible_not_entailment():
    projection, context = inputs()
    raw = {"relation": "same_assertion", "target_frame_id": 7, "target_claim_index": None}
    model = identity_proposal_model(projection, context)
    jsonschema.validate(raw, model.model_json_schema())
    assert model.model_validate(raw).target_frame_id == 7
