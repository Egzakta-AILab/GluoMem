"""Portable critical contracts; authored inputs and deterministic embeddings only."""

from collections.abc import Sequence
from datetime import datetime

from gluomem.config.app_config import AppConfig
from gluomem.llm.outputs import (
    CognifyResult,
    ConsolidationDecision,
    ContradictionVerdict,
    ExtractedClaimSpec,
    ExtractedEvent,
    ExtractedFact,
    StoredCandidate,
    projected_decision_model,
)
from gluomem.write.gate import ScanVerdict


class SafeScanner:
    def scan(self, content, *, context=""):
        return ScanVerdict(score=0.0)


class UnusedExtractorLanes:
    """Complete legacy lanes, refusing any call outside an authored fixture's script."""

    def extract_facts(self, text: str) -> list[ExtractedFact]:
        raise AssertionError("Unexpected extract_facts call in authored projection fixture")

    def extract_events(self, source_text: str, reference_date: datetime) -> list[ExtractedEvent]:
        raise AssertionError("Unexpected extract_events call in authored projection fixture")

    def extract_profile(self, text: str, *, speaker: str) -> list[ExtractedFact]:
        raise AssertionError("Unexpected extract_profile call in authored projection fixture")

    def cognify(self, text: str) -> CognifyResult:
        raise AssertionError("Unexpected cognify call in authored projection fixture")

    def check_contradiction(
        self, new_statement: str, stored_statement: str
    ) -> ContradictionVerdict:
        raise AssertionError("Unexpected check_contradiction call in authored projection fixture")

    def consolidate_claim(
        self, new_claim: str, candidates: Sequence[StoredCandidate]
    ) -> ConsolidationDecision:
        raise AssertionError("Unexpected legacy consolidation call in authored projection fixture")


class AuthoredProjection(UnusedExtractorLanes):
    """Scripted authored actions, not model-quality evidence."""

    consolidation_projection_protocol = "evolving-per-claim-v1"

    def __init__(self, specs, decisions):
        self.specs = specs
        self.decisions = decisions
        self.observed = []

    def extract_event_claims(
        self, source_text: str, *, reference_date: datetime | None = None
    ) -> list[ExtractedClaimSpec]:
        return self.specs

    def consolidate_claim_with_projection(self, text, candidates, *, context, projection):
        selected = self.decisions[projection.claim_index]
        self.observed.append(selected.copy())
        return projected_decision_model(projection).model_validate(selected)


def choice(action, *, frame_id=None, claim_index=None):
    return {
        "action": action,
        "target_frame_id": frame_id,
        "target_claim_index": claim_index,
        "rationale": "Explicit authored offline selection.",
    }


def config():
    cfg = AppConfig(dspy={"model": "mock"}, _env_file=None, _secrets_dir=None)
    cfg.write.cognify.enabled = False
    return cfg
