"""Opt-in dates; legacy wire and configuration are not upgraded implicitly."""

from typing import get_args

import pytest
from pydantic import ValidationError

from gluomem.config.app_config import AppConfig
from gluomem.llm.bootstrap import load_or_build
from gluomem.llm.outputs import ExtractedClaimSpec
from gluomem.llm.semantic import SemanticClaimSpec

POLICY = "observed-assertion-date-v1"


def config(**write):
    return AppConfig(
        dspy={"model": "mock", "adapter": "json", "schema_mode": "required"},
        write={
            "writer_contract": "semantic-state-writer-v4",
            "source_capture_policy": "attributed-interaction",
            **write,
        },
        _env_file=None,
        _secrets_dir=None,
    )


def test_optin_required_nullable_date_after_loaded_artifact():
    cfg = config(semantic_producer_contract=POLICY)
    assert getattr(cfg.write, "semantic_producer_contract", None) == POLICY
    module = load_or_build("event_claims", cfg)
    loaded = module.predictors()[0].signature
    assert loaded is not None
    annotation = loaded.fields["claims"].annotation
    assert annotation is not None
    claim = get_args(annotation)[0]
    # Loaded overlays extend the extraction root, not the public semantic leaf.
    assert issubclass(claim, ExtractedClaimSpec)
    assert claim.model_fields["event_date"].is_required()
    raw = {
        "text": "Tessa stores the parcel in bay C.",
        "message_id": "m",
        "verbatim_span": "Tessa stores the parcel in bay C.",
        "meaning": {
            "key": {"subject": "Tessa", "predicate": "parcel-place", "perspective": "user"},
            "value_key": "C",
            "independently_versioned": True,
            "refines": None,
        },
    }
    with pytest.raises(ValidationError, match="event_date"):
        claim.model_validate(raw)
    assert claim.model_validate({**raw, "event_date": None}).event_date is None
    assert SemanticClaimSpec.model_validate(raw).event_date is None
    old = load_or_build("event_claims", config()).predictors()[0].signature
    assert old is not None
    old_annotation = old.fields["claims"].annotation
    assert old_annotation is not None
    old_claim = get_args(old_annotation)[0]
    assert issubclass(old_claim, ExtractedClaimSpec)
    assert not old_claim.model_fields["event_date"].is_required()
    assert "semantic_producer_contract" not in config().write.model_dump()
    assert "identity_alignment" not in config().dspy.tasks
    assert cfg.dspy.tasks["identity_alignment"].model is None
    assert "world-effective" in loaded.instructions


@pytest.mark.parametrize(
    "changes",
    [{"writer_contract": None}, {"extraction_backend": "fake"}, {"source_capture_policy": None}],
)
def test_optin_rejects_incompatible_write_policy(changes):
    with pytest.raises(ValueError):
        config(semantic_producer_contract=POLICY, **changes)
