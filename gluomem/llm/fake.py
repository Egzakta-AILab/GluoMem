"""gluomem.llm.fake — deterministic, dspy-free ``Extractors`` implementations.

``FakeExtractors`` mirrors the ``MockEmbedder`` precedent: fully offline,
deterministic, and LOUD about not being the real thing. Rule-based defaults
(sentence-split facts, cue-resolved events via the relative-date resolver,
empty cognify, no-contradiction) can be overridden per-method with scripted
queues so write-path tests can drive exact scenarios.

``NullExtractors`` extracts nothing at all — the raw-turns-only ablation
baseline (``write.extraction_backend = "none"``).
"""

from __future__ import annotations

import logging
import re
import sys
import threading
from datetime import datetime

from gluomem.kg.temporal import resolve_relative_date
from gluomem.llm.outputs import (
    CognifyResult,
    ContradictionVerdict,
    ExtractedEvent,
    ExtractedFact,
)

_log = logging.getLogger(__name__)

_FAKE_BANNER = (
    "!!! GLUOMEM IS USING THE FAKE EXTRACTORS (DETERMINISTIC, NO LLM) — "
    "EXTRACTION OUTPUT IS RULE-BASED, NOT MODEL-BASED !!!"
)

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")

# Relative-time cues the resolver understands, longest-first so
# "day before yesterday" wins over "yesterday".
_TIME_CUES = (
    "day before yesterday",
    "day after tomorrow",
    "yesterday",
    "tomorrow",
    "today",
    "last week",
    "next week",
)
_N_DAYS_RE = re.compile(r"\b(?:(\d+) days ago|in (\d+) days)\b", re.IGNORECASE)


def _emit_fake_banner() -> None:
    """Emit the LOUD all-caps FAKE banner to stderr and logging.warning."""
    print(_FAKE_BANNER, file=sys.stderr, flush=True)
    _log.warning(_FAKE_BANNER)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]


def _find_cue(sentence: str) -> str | None:
    low = sentence.lower()
    for cue in _TIME_CUES:
        if cue in low:
            return cue
    m = _N_DAYS_RE.search(low)
    if m is not None:
        return m.group(0)
    return None


class FakeExtractors:
    """Deterministic rule-based ``Extractors`` (offline tests / dev).

    Scripted queues: any of ``facts``/``events``/``cognify_results``/
    ``verdicts`` may be passed as a list; each call pops from the front and
    falls back to the rule-based default when the queue is exhausted.
    """

    def __init__(
        self,
        *,
        facts: list[list[ExtractedFact]] | None = None,
        events: list[list[ExtractedEvent]] | None = None,
        cognify_results: list[CognifyResult] | None = None,
        verdicts: list[ContradictionVerdict] | None = None,
        banner: bool = True,
    ) -> None:
        self._facts_queue = list(facts or [])
        self._events_queue = list(events or [])
        self._cognify_queue = list(cognify_results or [])
        self._verdicts_queue = list(verdicts or [])
        self.calls: dict[str, int] = {
            "extract_facts": 0,
            "extract_events": 0,
            "extract_profile": 0,
            "cognify": 0,
            "check_contradiction": 0,
        }
        # Ingest fans extraction calls out to threads (write.parallelism);
        # queue pops and counters must stay atomic.
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
        self, source_text: str, reference_date: datetime
    ) -> list[ExtractedEvent]:
        with self._lock:
            self.calls["extract_events"] += 1
            if self._events_queue:
                return self._events_queue.pop(0)
        out: list[ExtractedEvent] = []
        for sentence in _sentences(source_text):
            cue = _find_cue(sentence)
            if cue is None:
                continue
            iso = resolve_relative_date(cue, reference_date.date())
            out.append(
                ExtractedEvent(
                    text=sentence,
                    event_date=datetime.fromisoformat(iso),
                    confidence=1.0,
                )
            )
        return out

    def extract_profile(self, text: str, *, speaker: str) -> list[ExtractedFact]:
        with self._lock:
            self.calls["extract_profile"] += 1
            if self._facts_queue:
                return self._facts_queue.pop(0)
        return [
            ExtractedFact(text=f"{speaker}: {s}", confidence=1.0)
            for s in _sentences(text)
        ]

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


__all__ = ["FakeExtractors", "NullExtractors"]
