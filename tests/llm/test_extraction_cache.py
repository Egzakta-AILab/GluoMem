"""CachedExtractors — freeze one extraction, replay it across ablation arms.

Uses FakeExtractors (deterministic, call-counting, dspy-free) as the inner so
the cache behaviour is testable offline. See ``gluomem/llm/cache.py``.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

import pytest

from gluomem.llm import (
    CachedExtractors,
    ExtractionCacheMiss,
    ExtractionError,
    Extractors,
    FakeExtractors,
)
from gluomem.llm.outputs import (
    CognifyResult,
    ContradictionVerdict,
    ExtractedEvent,
    ExtractedFact,
    KGEntity,
)

_TEXT = "Alice moved to Denver. She likes tea."


class _Boom:
    """Inner Extractors that fails if any method is called (replay-hit guard)."""

    usage = None

    def extract_facts(self, text: str) -> list[ExtractedFact]:
        raise AssertionError("inner extractor called on a cache hit")

    def extract_events(
        self, source_text: str, reference_date: datetime
    ) -> list[ExtractedEvent]:
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


# --- construction / validation ----------------------------------------------


def test_bad_mode_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mode"):
        _cache(FakeExtractors(banner=False), tmp_path, "bogus")


def test_non_off_requires_path() -> None:
    with pytest.raises(ValueError, match="path"):
        CachedExtractors(FakeExtractors(banner=False), path="", mode="record")


# --- round-trip: record then replay -----------------------------------------


def test_record_then_replay_identical_all_methods(tmp_path: Path) -> None:
    inner = FakeExtractors(banner=False)
    rec = _cache(inner, tmp_path, "record")
    facts = rec.extract_facts(_TEXT)
    events = rec.extract_events("She moved yesterday.", datetime(2023, 5, 8))
    profile = rec.extract_profile("I love hiking.", speaker="Bob")
    cog = rec.cognify(_TEXT)
    verdict = rec.check_contradiction("Bob in Denver", "Bob in Boston")

    # Replay against an inner that raises if touched → every call must hit.
    rep = _cache(_Boom(), tmp_path, "replay")
    assert [f.model_dump() for f in rep.extract_facts(_TEXT)] == [
        f.model_dump() for f in facts
    ]
    assert [e.model_dump() for e in rep.extract_events(
        "She moved yesterday.", datetime(2023, 5, 8)
    )] == [e.model_dump() for e in events]
    assert [p.model_dump() for p in rep.extract_profile(
        "I love hiking.", speaker="Bob"
    )] == [p.model_dump() for p in profile]
    assert rep.cognify(_TEXT).model_dump() == cog.model_dump()
    assert rep.check_contradiction(
        "Bob in Denver", "Bob in Boston"
    ).model_dump() == verdict.model_dump()
    assert rep.cache_misses == 0


def test_replay_never_calls_inner(tmp_path: Path) -> None:
    inner = FakeExtractors(banner=False)
    _cache(inner, tmp_path, "record").extract_facts(_TEXT)
    before = FakeExtractors(banner=False)  # fresh counters
    rep = _cache(before, tmp_path, "replay")
    rep.extract_facts(_TEXT)
    assert before.calls["extract_facts"] == 0  # served from cache, inner untouched


def test_cached_empty_list_is_served_not_miss(tmp_path: Path) -> None:
    # extract_facts on empty text yields [] — a cached empty list must still be
    # a HIT (None means store-miss only).
    inner = FakeExtractors(banner=False)
    rec = _cache(inner, tmp_path, "record")
    assert rec.extract_facts("") == []
    rep = _cache(_Boom(), tmp_path, "replay")
    assert rep.extract_facts("") == []  # would raise via _Boom on a miss


# --- strict replay misses ----------------------------------------------------


def test_replay_miss_raises_cache_miss_not_extraction_error(tmp_path: Path) -> None:
    _cache(FakeExtractors(banner=False), tmp_path, "record").extract_facts(_TEXT)
    rep = _cache(_Boom(), tmp_path, "replay")
    with pytest.raises(ExtractionCacheMiss):
        rep.extract_facts("some text never recorded")
    # MUST NOT be an ExtractionError, else ingest's degrade-to-raw swallows it.
    assert not issubclass(ExtractionCacheMiss, ExtractionError)


# --- readthrough -------------------------------------------------------------


def test_readthrough_hit_skips_inner_miss_records_and_counts(tmp_path: Path) -> None:
    _cache(FakeExtractors(banner=False), tmp_path, "record").extract_facts(_TEXT)
    inner = FakeExtractors(banner=False)
    rt = _cache(inner, tmp_path, "readthrough")
    rt.extract_facts(_TEXT)  # hit
    assert inner.calls["extract_facts"] == 0 and rt.cache_misses == 0
    rt.extract_facts("brand new text")  # miss → live + record + count
    assert inner.calls["extract_facts"] == 1 and rt.cache_misses == 1
    # The miss was persisted → a later strict replay now hits it.
    rep = _cache(_Boom(), tmp_path, "replay")
    rep.extract_facts("brand new text")


# --- keying ------------------------------------------------------------------


def test_facts_and_profile_same_text_distinct_keys(tmp_path: Path) -> None:
    inner = FakeExtractors(
        facts=[[ExtractedFact(text="FACT")], [ExtractedFact(text="PROF")]],
        banner=False,
    )
    rec = _cache(inner, tmp_path, "record")
    f = rec.extract_facts("same text")
    p = rec.extract_profile("same text", speaker="Bob")
    # Different method (and profile adds speaker) → different cache slots.
    assert f[0].text == "FACT" and p[0].text == "PROF"


def test_key_stable_across_unrelated_state(tmp_path: Path) -> None:
    inner = FakeExtractors(
        cognify_results=[
            CognifyResult(entities=[KGEntity(entity_type="person", name="A")])
        ],
        banner=False,
    )
    rec = _cache(inner, tmp_path, "record")
    rec.cognify(_TEXT)
    # A brand-new wrapper (no shared memory) replaying the SAME input hits — the
    # key depends only on method+inputs, not on any per-process state.
    rep = _cache(_Boom(), tmp_path, "replay")
    assert rep.cognify(_TEXT).entities[0].name == "A"


# --- concurrency -------------------------------------------------------------


def test_thread_safe_record(tmp_path: Path) -> None:
    inner = FakeExtractors(banner=False)
    rec = _cache(inner, tmp_path, "record")
    texts = [f"sentence number {i}." for i in range(64)]

    def work(t: str) -> None:
        rec.extract_facts(t)

    threads = [threading.Thread(target=work, args=(t,)) for t in texts]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    # Every distinct input persisted exactly once; replay hits all of them.
    rep = _cache(_Boom(), tmp_path, "replay")
    for t in texts:
        rep.extract_facts(t)
    assert rep.cache_misses == 0


def test_cache_sha_order_independent(tmp_path: Path) -> None:
    a = _cache(FakeExtractors(banner=False), tmp_path / "a", "record")
    a.extract_facts("one")
    a.extract_facts("two")
    b = _cache(FakeExtractors(banner=False), tmp_path / "b", "record")
    b.extract_facts("two")  # reverse insertion order
    b.extract_facts("one")
    assert a.cache_sha() == b.cache_sha()
