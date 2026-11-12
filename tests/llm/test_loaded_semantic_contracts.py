"""Loaded optional producer contracts, without provider sends or captured responses."""

from typing import get_args

import pytest

from gluomem.llm.bootstrap import load_or_build
from gluomem.llm.cache import (
    CachedExtractors,
    ExtractionCacheMiss,
    _ObservedCached,
    clear_semantic_caches,
)
from gluomem.llm.observed_identity import PropertyIdentityProposal
from gluomem.llm.observed_signatures import identity_alignment_module
from gluomem.llm.outputs import ExtractedClaimSpec
from gluomem.llm.semantic import ClaimMeaning
from gluomem.llm.structured_adapter import output_schema
from tests.llm.test_observed_identity_date import config
from tests.llm.test_observed_identity_proposal import inputs
from tests.write.test_receipt_coherence import Computation


@pytest.mark.parametrize(
    "producer",
    [
        "observed-assertion-date-v1",
        "observed-assertion-date-v2",
        "observed-property-identity-v3",
        "observed-assertion-scope-v4",
    ],
)
def test_loaded_producer_requires_explicit_nullable_date_and_scope_when_selected(producer):
    cfg = config(semantic_producer_contract=producer)
    cfg.dspy.artifacts.require = True
    loaded = load_or_build("event_claims", cfg).predictors()[0].signature
    assert loaded is not None
    annotation = loaded.fields["claims"].annotation
    assert annotation is not None
    claim = get_args(annotation)[0]
    # Loaded overlays extend the extraction root, not the public semantic leaf.
    assert issubclass(claim, ExtractedClaimSpec)
    assert "source_context" in loaded.input_fields
    assert claim.model_fields["event_date"].is_required()
    raw = {
        "text": "The cabinet is blue.",
        "message_id": "m",
        "verbatim_span": "The cabinet is blue.",
        "meaning": {
            "key": {"subject": "cabinet", "predicate": "colour", "perspective": "user"},
            "value_key": "blue",
            "independently_versioned": True,
            "refines": None,
        },
    }
    if producer == "observed-assertion-scope-v4":
        with pytest.raises(ValueError, match="assertion_act"):
            claim.model_validate({**raw, "event_date": None})
        raw["meaning"]["assertion_act"] = "direct_assertion"
    else:
        meaning_type = claim.model_fields["meaning"].annotation
        assert meaning_type is not None and issubclass(meaning_type, ClaimMeaning)
        assert "assertion_act" not in meaning_type.model_fields
    with pytest.raises(ValueError, match="event_date"):
        claim.model_validate(raw)
    assert claim.model_validate({**raw, "event_date": None}).event_date is None


def test_property_identity_schema_keeps_relation_separate_from_values_and_dates():
    producer = "observed-property-identity-v3"
    cfg = config(semantic_producer_contract=producer)
    observed, context = inputs()
    base = load_or_build("identity_alignment", cfg)
    signature = (
        identity_alignment_module(base, observed, context, producer_contract=producer)
        .predictors()[0]
        .signature
    )
    assert signature is not None
    schema = output_schema(signature)
    proposal = schema["properties"]["proposal"]
    if "$ref" in proposal:
        proposal = schema["$defs"][proposal["$ref"].rsplit("/", 1)[1]]
    assert "assertion_relation" in proposal["required"]
    assert "property_slot_context" in signature.input_fields
    assert "value-independent and date-independent" in signature.instructions
    assert "source_alias" in signature.instructions
    assert "Never rewrite" in signature.instructions


def test_semantic_cache_is_wrapper_private_revocable_and_never_uses_disk(tmp_path):
    path = tmp_path / "forbidden-transcript.jsonl"
    path.write_text("This is not a cache and must not be parsed.", encoding="utf-8")

    def wrapper(mode):
        cache = CachedExtractors(
            Computation([], [], []),
            path=str(path),
            mode=mode,
            writer_contract="semantic-state-writer-v4",
            semantic_producer_contract="observed-assertion-scope-v4",
        )
        # The factory must select the observed, invocation-private wrapper.
        assert isinstance(cache, _ObservedCached)
        return cache

    cache = wrapper("record")
    assert not cache.persistent_cache_supported
    value = PropertyIdentityProposal(
        relation="independent",
        assertion_relation="unrelated",
        subject_alignment="not_aligned",
        property_alignment="not_aligned",
    )
    calls = []

    def produce():
        calls.append(True)
        return value

    assert cache._serve("align_claim_identity", "authored", produce) == value
    assert cache._serve("align_claim_identity", "authored", produce) == value
    assert calls == [True]
    cold = wrapper("replay")
    with pytest.raises(ExtractionCacheMiss):
        cold._serve("align_claim_identity", "authored", produce)
    generation = cache._semantic_generation
    clear_semantic_caches()
    assert cache._store == {} and cache._semantic_generation > generation
    cache._persist("align_claim_identity", "stale-fill", value, generation=generation)
    assert cache._store == {}
    assert path.read_text(encoding="utf-8") == "This is not a cache and must not be parsed."


def test_typed_text_omits_native_grammar_but_keeps_local_target_and_character_guards():
    import json
    from types import SimpleNamespace

    from dspy.utils.exceptions import AdapterParseError

    from gluomem.config.app_config import AppConfig
    from gluomem.llm.structured_adapter import make_adapter

    cfg = AppConfig(
        dspy={
            "model": "mock",
            "adapter": "json",
            "schema_mode": "typed-text",
            "output_field_limits": {"consolidate": {"decision.rationale": 4}},
        },
        _env_file=None,
        _secrets_dir=None,
    )
    signature = load_or_build("consolidate", cfg).predictors()[0].signature
    assert signature is not None
    adapter = make_adapter("json", schema_mode="typed-text")
    options = {}
    lm = SimpleNamespace(kwargs={})
    adapter._prepare_response_format(lm, options, signature)
    assert options == {}
    for supplied in (options, lm.kwargs):
        supplied["response_format"] = {"type": "json_schema"}
        with pytest.raises(ValueError, match="forbids"):
            adapter._prepare_response_format(lm, options, signature)
        supplied.clear()
    decision = {"action": "SUPERSEDE", "target_frame_id": 7, "rationale": "é" * 4}
    assert (
        adapter.parse(signature, json.dumps({"decision": decision}))["decision"].target_frame_id
        == 7
    )
    for change in ({"target_frame_id": None}, {"rationale": "é" * 5}):
        with pytest.raises(AdapterParseError):
            adapter.parse(signature, json.dumps({"decision": {**decision, **change}}))
