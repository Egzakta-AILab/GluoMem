"""gluomem.llm.fake — deterministic, dspy-free ``Extractors`` implementations.

``FakeExtractors`` mirrors the ``MockEmbedder`` precedent: fully offline,
deterministic, and LOUD about not being the real thing. Scripted queues can be
passed per method so write-path tests drive exact extraction and consolidation
scenarios without regex/keyword heuristics.

``NullExtractors`` extracts nothing at all — the raw-turns-only ablation
baseline (``write.extraction_backend = "none"``).
"""

from __future__ import annotations

import logging
import re
import sys
import threading
from typing import TYPE_CHECKING

from gluomem.llm.outputs import (
    CognifyResult,
    ConsolidationDecision,
    ContradictionVerdict,
    ExtractedClaimSpec,
    ExtractedEvent,
    ExtractedFact,
    StoredCandidate,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

_log = logging.getLogger(__name__)

_FAKE_BANNER = (
    "!!! GLUOMEM IS USING THE FAKE EXTRACTORS (DETERMINISTIC, NO LLM) — "
    "EXTRACTION OUTPUT IS SCRIPTED/DETERMINISTIC, NOT MODEL-BASED !!!"
)

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")
_LINE_PREFIX_RE = re.compile(r"^\s*\[(?P<mid>[^\]]+)\]\s*(?P<speaker>[^:]+):\s*(?P<body>.+)$")


def _emit_fake_banner() -> None:
    """Emit the LOUD all-caps FAKE banner to stderr and logging.warning."""
    print(_FAKE_BANNER, file=sys.stderr, flush=True)
    _log.warning(_FAKE_BANNER)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]


class FakeExtractors:
    """Deterministic ``Extractors`` test double (offline tests / dev).

    Scripted queues: any of ``facts`` / ``events`` / ``cognify_results`` /
    ``verdicts`` / ``event_claims`` / ``consolidation_decisions`` may be passed
    as a list; each call pops from the front and falls back to a deterministic
    default when the queue is exhausted.
    """

    def __init__(
        self,
        *,
        facts: list[list[ExtractedFact]] | None = None,
        events: list[list[ExtractedEvent]] | None = None,
        cognify_results: list[CognifyResult] | None = None,
        verdicts: list[ContradictionVerdict] | None = None,
        event_claims: list[list[ExtractedClaimSpec]] | None = None,
        consolidation_decisions: list[ConsolidationDecision] | None = None,
        banner: bool = True,
    ) -> None:
        self._facts_queue = list(facts or [])
        self._events_queue = list(events or [])
        self._cognify_queue = list(cognify_results or [])
        self._verdicts_queue = list(verdicts or [])
        self._event_claims_queue = list(event_claims or [])
        self._consolidation_queue = list(consolidation_decisions or [])
        self.calls: dict[str, int] = {
            "extract_facts": 0,
            "extract_events": 0,
            "extract_profile": 0,
            "cognify": 0,
            "check_contradiction": 0,
            "extract_event_claims": 0,
            "consolidate_claim": 0,
        }
        self._lock = threading.Lock()
        if banner:
            _emit_fake_banner()

    # --- Extractors protocol -------------------------------------------------

    def extract_facts(self, text: str) -> list[ExtractedFact]:
        with self._lock:
            self.calls["extract_facts"] += 1
            if self._facts_queue:
                return self._facts_queue.pop(0)
        return [ExtractedFact(text=s, confidence=1.0) for s in _sentences(text)]

    def extract_events(
        self,
        source_text: str,  # noqa: ARG002
        reference_date: datetime,  # noqa: ARG002
    ) -> list[ExtractedEvent]:
        with self._lock:
            self.calls["extract_events"] += 1
            if self._events_queue:
                return self._events_queue.pop(0)
        return []

    def extract_profile(self, text: str, *, speaker: str) -> list[ExtractedFact]:
        with self._lock:
            self.calls["extract_profile"] += 1
            if self._facts_queue:
                return self._facts_queue.pop(0)
        return [ExtractedFact(text=f"{speaker}: {s}", confidence=1.0) for s in _sentences(text)]

    def cognify(self, text: str) -> CognifyResult:  # noqa: ARG002
        with self._lock:
            self.calls["cognify"] += 1
            if self._cognify_queue:
                return self._cognify_queue.pop(0)
        return CognifyResult()

    def check_contradiction(
        self,
        new_statement: str,  # noqa: ARG002
        stored_statement: str,  # noqa: ARG002
    ) -> ContradictionVerdict:
        with self._lock:
            self.calls["check_contradiction"] += 1
            if self._verdicts_queue:
                return self._verdicts_queue.pop(0)
        return ContradictionVerdict(is_contradiction=False)

    def extract_event_claims(
        self,
        source_text: str,
        *,
        reference_date: datetime | None = None,
    ) -> list[ExtractedClaimSpec]:
        with self._lock:
            self.calls["extract_event_claims"] += 1
            if self._event_claims_queue:
                return self._event_claims_queue.pop(0)
            cognify_res = self._cognify_queue.pop(0) if self._cognify_queue else None

        claims: list[ExtractedClaimSpec] = []
        for raw_line in source_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            m = _LINE_PREFIX_RE.match(line)
            if m is not None:
                mid: str | None = m.group("mid").strip() or None
                body = m.group("body").strip()
            else:
                mid = None
                body = line
            for sentence in _sentences(body):
                claims.append(
                    ExtractedClaimSpec(
                        text=sentence,
                        kind="fact",
                        confidence=1.0,
                        event_date=reference_date,
                        verbatim_span=sentence,
                        message_id=mid,
                        entities=list(cognify_res.entities) if cognify_res is not None else [],
                        relations=list(cognify_res.relations) if cognify_res is not None else [],
                    )
                )
        return claims

    def consolidate_claim(
        self,
        new_claim: str,
        candidates: Sequence[StoredCandidate],
    ) -> ConsolidationDecision:
        with self._lock:
            self.calls["consolidate_claim"] += 1
            if self._consolidation_queue:
                return self._consolidation_queue.pop(0)
        norm_new = " ".join(new_claim.split()).lower()
        for cand in candidates:
            if " ".join(cand.content.split()).lower() == norm_new:
                return ConsolidationDecision(
                    action="NOOP",
                    target_frame_id=cand.frame_id,
                    rationale="Exact normalized match with existing frame.",
                )
        return ConsolidationDecision(action="ADD")


class NullExtractors:
    """No extraction at all — the raw-turns-only ablation baseline."""

    def extract_facts(self, text: str) -> list[ExtractedFact]:  # noqa: ARG002
        return []

    def extract_events(
        self,
        source_text: str,  # noqa: ARG002
        reference_date: datetime,  # noqa: ARG002
    ) -> list[ExtractedEvent]:
        return []

    def extract_profile(
        self,
        text: str,  # noqa: ARG002
        *,
        speaker: str,  # noqa: ARG002
    ) -> list[ExtractedFact]:
        return []

    def cognify(self, text: str) -> CognifyResult:  # noqa: ARG002
        return CognifyResult()

    def check_contradiction(
        self,
        new_statement: str,  # noqa: ARG002
        stored_statement: str,  # noqa: ARG002
    ) -> ContradictionVerdict:
        return ContradictionVerdict(is_contradiction=False)

    def extract_event_claims(
        self,
        source_text: str,  # noqa: ARG002
        *,
        reference_date: datetime | None = None,  # noqa: ARG002
    ) -> list[ExtractedClaimSpec]:
        return []

    def consolidate_claim(
        self,
        new_claim: str,  # noqa: ARG002
        candidates: Sequence[StoredCandidate],  # noqa: ARG002
    ) -> ConsolidationDecision:
        return ConsolidationDecision(action="ADD")


__all__ = ["FakeExtractors", "NullExtractors"]
