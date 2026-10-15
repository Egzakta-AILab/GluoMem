"""CachedExtractors record, replay, readthrough, and concurrency tests."""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

import pytest

from gluomem.llm import (
    CachedExtractors,
    CognifyResult,
    ContradictionVerdict,
    ExtractedEvent,
    ExtractedFact,
    ExtractionCacheMiss,
    ExtractionError,
    Extractors,
    FakeExtractors,
    KGEntity,
)

_TEXT = "Alice moved to Denver. She likes tea."


class _Boom:
    usage = None

    def extract_facts(self, text: str) -> list[ExtractedFact]:
        raise AssertionError("inner extractor called on a cache hit")

    def extract_events(self, source_text: str, reference_date: datetime) -> list[ExtractedEvent]:
        raise AssertionError("inner extractor called on a cache hit")

    def extract_profile(self, text: str, *, speaker: str) -> list[ExtractedFact]:
        raise AssertionError("inner extractor called on a cache hit")

    def cognify(self, text: str) -> CognifyResult:
        raise AssertionError("inner extractor called on a cache hit")

    def check_contradiction(
        self, new_statement: str, stored_statement: str
    ) -> ContradictionVerdict:
        raise AssertionError("inner extractor called on a cache hit")


def _cache(inner: Extractors, tmp_path: Path, mode: str) -> CachedExtractors:
    return CachedExtractors(inner, path=str(tmp_path / "cache.jsonl"), mode=mode)


def test_cached_extractors_record_replay_readthrough_and_concurrency(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="mode"):
        _cache(FakeExtractors(banner=False), tmp_path, "bogus")
    with pytest.raises(ValueError, match="path"):
        CachedExtractors(FakeExtractors(banner=False), path="", mode="record")

    inner = FakeExtractors(
        cognify_results=[CognifyResult(entities=[KGEntity(entity_type="person", name="A")])],
        banner=False,
    )
    rec = _cache(inner, tmp_path, "record")
    facts = rec.extract_facts(_TEXT)
    assert rec.extract_facts("") == []
    events = rec.extract_events("She moved yesterday.", datetime(2023, 5, 8))
    profile = rec.extract_profile("I love hiking.", speaker="Bob")
    cog = rec.cognify(_TEXT)
    verdict = rec.check_contradiction("Bob in Denver", "Bob in Boston")

    rep = _cache(_Boom(), tmp_path, "replay")
    assert [f.model_dump() for f in rep.extract_facts(_TEXT)] == [f.model_dump() for f in facts]
    assert rep.extract_facts("") == []
    assert [
        e.model_dump() for e in rep.extract_events("She moved yesterday.", datetime(2023, 5, 8))
    ] == [e.model_dump() for e in events]
    assert [p.model_dump() for p in rep.extract_profile("I love hiking.", speaker="Bob")] == [
        p.model_dump() for p in profile
    ]
    assert rep.cognify(_TEXT).model_dump() == cog.model_dump()
    assert (
        rep.check_contradiction("Bob in Denver", "Bob in Boston").model_dump()
        == verdict.model_dump()
    )
    assert rep.cache_misses == 0

    with pytest.raises(ExtractionCacheMiss):
        rep.extract_facts("never recorded")
    assert not issubclass(ExtractionCacheMiss, ExtractionError)

    rt_inner = FakeExtractors(banner=False)
    rt = _cache(rt_inner, tmp_path, "readthrough")
    rt.extract_facts(_TEXT)
    assert rt_inner.calls["extract_facts"] == 0 and rt.cache_misses == 0
    rt.extract_facts("brand new text")
    assert rt_inner.calls["extract_facts"] == 1 and rt.cache_misses == 1

    # Order-independent cache SHA and thread-safe writes.
    a = _cache(FakeExtractors(banner=False), tmp_path / "a", "record")
    b = _cache(FakeExtractors(banner=False), tmp_path / "b", "record")
    a.extract_facts("one")
    a.extract_facts("two")
    b.extract_facts("two")
    b.extract_facts("one")
    assert a.cache_sha() == b.cache_sha()

    t_rec = _cache(FakeExtractors(banner=False), tmp_path / "t", "record")
    threads = [threading.Thread(target=t_rec.extract_facts, args=(f"s {i}.",)) for i in range(16)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    t_rep = _cache(_Boom(), tmp_path / "t", "replay")
    for i in range(16):
        t_rep.extract_facts(f"s {i}.")
    assert t_rep.cache_misses == 0
