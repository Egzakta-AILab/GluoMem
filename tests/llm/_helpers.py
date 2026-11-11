"""Authored transport doubles shared by LLM contract tests."""

from collections.abc import Sequence
from datetime import datetime

from dspy.utils.dummies import DummyLM

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
    StoredCandidate,
)


class Scripted(DummyLM):
    def __init__(self, responses):
        super().__init__([])
        self.responses = iter(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        value = next(self.responses)
        if isinstance(value, BaseException):
            raise value
        return [value]

    async def acall(self, prompt=None, *, messages=None, **kwargs):
        return self(prompt=prompt, messages=messages, **kwargs)


class ContextOnlyExtractors:
    """Exercise the unchanged legacy/context API, not the optional projection capability."""

    def __init__(self, inner: Extractors):
        self.inner = inner

    def extract_facts(self, text: str) -> list[ExtractedFact]:
        return self.inner.extract_facts(text)

    def extract_events(self, source_text: str, reference_date: datetime) -> list[ExtractedEvent]:
        return self.inner.extract_events(source_text, reference_date)

    def extract_profile(self, text: str, *, speaker: str) -> list[ExtractedFact]:
        return self.inner.extract_profile(text, speaker=speaker)

    def cognify(self, text: str) -> CognifyResult:
        return self.inner.cognify(text)

    def check_contradiction(
        self, new_statement: str, stored_statement: str
    ) -> ContradictionVerdict:
        return self.inner.check_contradiction(new_statement, stored_statement)

    def extract_event_claims(
        self, source_text: str, *, reference_date: datetime | None = None
    ) -> list[ExtractedClaimSpec]:
        return self.inner.extract_event_claims(source_text, reference_date=reference_date)

    def consolidate_claim(
        self, new_claim: str, candidates: Sequence[StoredCandidate]
    ) -> ConsolidationDecision:
        return self.inner.consolidate_claim(new_claim, candidates)

    def consolidate_claim_with_context(
        self,
        new_claim: str,
        candidates: Sequence[StoredCandidate],
        *,
        context: ClaimConsolidationContext,
    ) -> ConsolidationDecision:
        assert isinstance(self.inner, ContextualConsolidation)
        return self.inner.consolidate_claim_with_context(new_claim, candidates, context=context)
