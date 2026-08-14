"""Shared write-path test env: MindDB + FrameStore(MockEmbedder) + KG + config.

Everything offline and dspy-free; extraction is driven by FakeExtractors
scripted queues or the substring-keyed ScriptedFacade.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

import pytest

from gluomem.config.app_config import AppConfig, load_config
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.llm import (
    CognifyResult,
    ContradictionVerdict,
    ExtractedEvent,
    ExtractedFact,
)
from gluomem.storage.db import MindDB


class WriteEnv:
    """One in-memory-ish write environment over a tmp .mind file."""

    def __init__(self, tmp_path: Path) -> None:
        self.cfg: AppConfig = load_config()
        self.db = MindDB(tmp_path / "test.mind", dim=self.cfg.storage.vector_dim)
        self.store = FrameStore(
            self.db, embedder=MockEmbedder(dim=self.db.dim), chunking=self.cfg.chunking
        )
        self.kg = KnowledgeGraph(self.db)


@pytest.fixture
def env(tmp_path: Path) -> WriteEnv:
    return WriteEnv(tmp_path)


@pytest.fixture
def scripted_facade_cls() -> type[ScriptedFacade]:
    """Class handle for tests (conftest symbols aren't importable directly)."""
    return ScriptedFacade


class ScriptedFacade:
    """Deterministic ``Extractors`` keyed on content substrings.

    Each map is ``{substring: outputs}``; a method returns the concatenated
    outputs of every key found in its input text (insertion order), so
    multi-session e2e scripts stay readable. Unmatched text extracts nothing.
    """

    def __init__(
        self,
        *,
        facts: dict[str, list[ExtractedFact]] | None = None,
        events: dict[str, list[ExtractedEvent]] | None = None,
        cognify_map: dict[str, CognifyResult] | None = None,
        verdicts: dict[str, ContradictionVerdict] | None = None,
    ) -> None:
        self._facts = facts or {}
        self._events = events or {}
        self._cognify = cognify_map or {}
        self._verdicts = verdicts or {}
        self.calls: dict[str, int] = {
            "extract_facts": 0,
            "extract_events": 0,
            "extract_profile": 0,
            "cognify": 0,
            "check_contradiction": 0,
        }
        self.contradiction_inputs: list[tuple[str, str]] = []
        self.seen_texts: list[str] = []
        # Ingest fans calls out to threads (write.parallelism).
        self._lock = threading.Lock()

    def extract_facts(self, text: str) -> list[ExtractedFact]:
        with self._lock:
            self.calls["extract_facts"] += 1
            self.seen_texts.append(text)
        out: list[ExtractedFact] = []
        for key, facts in self._facts.items():
            if key in text:
                out.extend(facts)
        return out

    def extract_events(
        self, source_text: str, reference_date: datetime
    ) -> list[ExtractedEvent]:  # noqa: ARG002
        with self._lock:
            self.calls["extract_events"] += 1
        out: list[ExtractedEvent] = []
        for key, events in self._events.items():
            if key in source_text:
                out.extend(events)
        return out

    def extract_profile(self, text: str, *, speaker: str) -> list[ExtractedFact]:
        with self._lock:
            self.calls["extract_profile"] += 1
        out: list[ExtractedFact] = []
        for key, facts in self._facts.items():
            if key in text:
                out.extend(
                    ExtractedFact(text=f"{speaker}: {f.text}", confidence=f.confidence)
                    for f in facts
                )
        return out

    def cognify(self, text: str) -> CognifyResult:
        with self._lock:
            self.calls["cognify"] += 1
        for key, result in self._cognify.items():
            if key in text:
                return result
        return CognifyResult()

    def check_contradiction(
        self, new_statement: str, stored_statement: str
    ) -> ContradictionVerdict:
        with self._lock:
            self.calls["check_contradiction"] += 1
            self.contradiction_inputs.append((new_statement, stored_statement))
        for key, verdict in self._verdicts.items():
            if key in new_statement:
                return verdict
        return ContradictionVerdict(is_contradiction=False)
