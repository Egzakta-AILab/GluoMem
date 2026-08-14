"""FakeExtractors / NullExtractors / make_extractors — deterministic seam.

All offline; the fake reuses the rule-based relative-date resolver so
event extraction is deterministic without any LLM.
"""

from __future__ import annotations

from datetime import datetime

from gluomem.config.app_config import load_config
from gluomem.llm import (
    CognifyResult,
    ContradictionVerdict,
    ExtractedFact,
    FakeExtractors,
    KGEntity,
    NullExtractors,
    make_extractors,
)


def test_fake_facts_sentence_split_deterministic() -> None:
    fx = FakeExtractors(banner=False)
    text = "Alice lives in Boston. She works at Acme."
    first = fx.extract_facts(text)
    second = fx.extract_facts(text)
    assert [f.text for f in first] == [f.text for f in second]
    assert [f.text for f in first] == ["Alice lives in Boston.", "She works at Acme."]


def test_fake_events_resolve_relative_cues() -> None:
    fx = FakeExtractors(banner=False)
    ref = datetime(2023, 5, 8)
    events = fx.extract_events("We met yesterday. The sky is blue.", ref)
    assert len(events) == 1
    assert events[0].event_date is not None
    assert events[0].event_date.date().isoformat() == "2023-05-07"


def test_fake_profile_prefixes_speaker() -> None:
    fx = FakeExtractors(banner=False)
    facts = fx.extract_profile("I love hiking.", speaker="Alice")
    assert facts == [ExtractedFact(text="Alice: I love hiking.", confidence=1.0)]


def test_fake_scripted_queues_and_call_counters() -> None:
    scripted = CognifyResult(entities=[KGEntity(entity_type="person", name="alice")])
    fx = FakeExtractors(
        cognify_results=[scripted],
        verdicts=[ContradictionVerdict(is_contradiction=True, description="moved")],
        banner=False,
    )
    assert fx.cognify("anything") == scripted
    assert fx.cognify("anything") == CognifyResult()  # queue exhausted -> default
    verdict = fx.check_contradiction("new", "old")
    assert verdict.is_contradiction and verdict.description == "moved"
    assert fx.calls["cognify"] == 2
    assert fx.calls["check_contradiction"] == 1


def test_fake_banner_emitted_once_on_stderr(capsys) -> None:
    FakeExtractors()
    captured = capsys.readouterr()
    assert "FAKE EXTRACTORS" in captured.err


def test_null_extractors_extract_nothing() -> None:
    nx = NullExtractors()
    assert nx.extract_facts("Alice lives in Boston.") == []
    assert nx.extract_events("yesterday", datetime(2023, 5, 8)) == []
    assert nx.extract_profile("text", speaker="Alice") == []
    assert nx.cognify("text") == CognifyResult()
    assert nx.check_contradiction("a", "b").is_contradiction is False


def test_make_extractors_backend_switch() -> None:
    cfg = load_config()
    fake = make_extractors(cfg.with_overrides({"write.extraction_backend": "fake"}))
    assert isinstance(fake, FakeExtractors)
    null = make_extractors(cfg.with_overrides({"write.extraction_backend": "none"}))
    assert isinstance(null, NullExtractors)
