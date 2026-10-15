"""Shared bench test fixtures: fixture adapter, per-conversation ingested DBs, and fake backends."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import pytest

from gluomem.bench.adapters.locomo import LoCoMoAdapter
from gluomem.config.app_config import AppConfig, load_config
from gluomem.llm.outputs import AnswerResult

FIXTURE = Path(__file__).parent.parent / "fixtures" / "locomo_tiny.json"
FIXTURE_SHA = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()

BENCH_OVERRIDES: dict[str, object] = {
    "benchmark.datasets_dir": str(FIXTURE.parent),
    "benchmark.datasets.locomo.file": FIXTURE.name,
    "benchmark.datasets.locomo.sha256": FIXTURE_SHA,
    "embedding.model": "mock",
    "components.allow_fake": True,
    "search.rerank.enabled": False,
}

FAKE_BACKENDS: dict[str, object] = {
    "benchmark.answer.backend": "fake",
    "benchmark.judge.backend": "fake",
    "benchmark.sample.per_category": 1,
    "benchmark.sample.manifest": "does-not-exist.json",
}


@pytest.fixture
def bench_cfg() -> AppConfig:
    return load_config().with_overrides(dict(BENCH_OVERRIDES))


@pytest.fixture
def fixture_adapter(bench_cfg: AppConfig) -> LoCoMoAdapter:
    return LoCoMoAdapter(bench_cfg, path=FIXTURE)


class RecordingAnswerFn:
    """Deterministic AnswerFn double: echoes a scripted answer and records calls."""

    def __init__(self, answer: str = "scripted answer") -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def __call__(self, question: str, context: str) -> AnswerResult:
        with self._lock:
            self.calls.append((question, context))
        return AnswerResult(text=self.answer, prompt_tokens=10, completion_tokens=5)


@pytest.fixture
def answer_fn() -> RecordingAnswerFn:
    return RecordingAnswerFn()


@pytest.fixture
def recording_answer_fn_cls() -> type[RecordingAnswerFn]:
    return RecordingAnswerFn


@pytest.fixture
def ingested_db_dir(bench_cfg: AppConfig, tmp_path: Path) -> Path:
    from gluomem.llm import NullExtractors
    from gluomem.write.ingest import ingest_path

    cfg = bench_cfg.with_overrides({"write.extraction_backend": "none"})
    db_dir = tmp_path / "dbs"
    db_dir.mkdir()
    ingest_path(FIXTURE, cfg, facade=NullExtractors(), db_path=db_dir / "conv-tiny-1.mind")
    return db_dir
