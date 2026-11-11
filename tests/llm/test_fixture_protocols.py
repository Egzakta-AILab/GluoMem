"""Offline checks for the complete, capability-narrow authored extractor fixtures."""

from datetime import UTC, datetime

import pytest

from gluomem.llm.fake import FakeExtractors
from gluomem.llm.outputs import (
    ClaimConsolidationContext,
    CognifyResult,
    ConsolidationDecision,
    ContextualConsolidation,
    ContradictionVerdict,
    ExtractedClaimSpec,
    ExtractedEvent,
    ExtractedFact,
    Extractors,
    ProjectedConsolidation,
)
from tests._source_helpers import AuthoredProjection
from tests.llm._helpers import ContextOnlyExtractors
from tests.write.test_cumulative_stored_reuse import AuthoredSelections
from tests.write.test_receipt_coherence import Computation

SCRIPTED_FACADES: tuple[Extractors, ...] = (
    AuthoredProjection([], []),
    AuthoredSelections([], []),
    Computation([], [], []),
)


@pytest.mark.parametrize("facade", SCRIPTED_FACADES, ids=["projection", "selection", "receipt"])
def test_scripted_fixtures_complete_legacy_protocol_without_unexpected_fallbacks(
    facade: Extractors,
):
    assert isinstance(facade, Extractors)
    assert facade.extract_event_claims("", reference_date=None) == []
    with pytest.raises(AssertionError, match="Unexpected extract_facts"):
        facade.extract_facts("Authored text.")
    with pytest.raises(AssertionError, match="Unexpected extract_events"):
        facade.extract_events("Authored text.", datetime(2026, 1, 1, tzinfo=UTC))
    with pytest.raises(AssertionError, match="Unexpected extract_profile"):
        facade.extract_profile("Authored text.", speaker="user")
    with pytest.raises(AssertionError, match="Unexpected check_contradiction"):
        facade.check_contradiction("New text.", "Prior text.")
    with pytest.raises(AssertionError, match="Unexpected legacy consolidation"):
        facade.consolidate_claim("Authored text.", [])
    if isinstance(facade, AuthoredSelections):
        assert facade.cognify("Authored text.") == CognifyResult()
    else:
        with pytest.raises(AssertionError, match="Unexpected cognify"):
            facade.cognify("Authored text.")


def test_context_only_fixture_delegates_all_legacy_lanes_without_projection_capability():
    fact = ExtractedFact(text="Authored fact.")
    profile = ExtractedFact(text="Authored profile.")
    event = ExtractedEvent(text="Authored event.")
    claim = ExtractedClaimSpec(text="Authored claim.")
    cognified = CognifyResult()
    verdict = ContradictionVerdict(is_contradiction=True, description="Authored contrast.")
    legacy = ConsolidationDecision(action="ADD", rationale="Authored legacy decision.")
    contextual = ConsolidationDecision(action="ADD", rationale="Authored context decision.")
    inner = FakeExtractors(
        facts=[[fact], [profile]],
        events=[[event]],
        event_claims=[[claim]],
        cognify_results=[cognified],
        verdicts=[verdict],
        consolidation_decisions=[legacy, contextual],
        banner=False,
    )
    facade: Extractors = ContextOnlyExtractors(inner)
    assert isinstance(facade, Extractors)
    assert not isinstance(facade, ProjectedConsolidation)
    assert facade.extract_facts("Authored input.") == [fact]
    assert facade.extract_events("Authored input.", datetime(2026, 1, 1, tzinfo=UTC)) == [event]
    assert facade.extract_profile("Authored input.", speaker="user") == [profile]
    assert facade.cognify("Authored input.") is cognified
    assert facade.check_contradiction("New text.", "Prior text.") is verdict
    assert facade.extract_event_claims("Authored input.", reference_date=None) == [claim]
    assert facade.consolidate_claim("Authored input.", []) is legacy
    assert isinstance(facade, ContextualConsolidation)
    context = ClaimConsolidationContext(
        kind="fact", source_role="user", observed_at=None, effective_at=None
    )
    assert (
        facade.consolidate_claim_with_context("Authored input.", [], context=context) is contextual
    )
    assert inner.calls == {
        "extract_facts": 1,
        "extract_events": 1,
        "extract_profile": 1,
        "cognify": 1,
        "check_contradiction": 1,
        "extract_event_claims": 1,
        "consolidate_claim": 2,
    }
