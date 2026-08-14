"""Mem0 integration — offline suite (mem0ai NOT installed; module mocked)."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from gluomem.bench.mem0 import (
    LiteLLMUsageTap,
    Mem0Retriever,
    build_mem0_config,
    ingest_mem0_per_conversation,
)
from gluomem.bench.types import Sample
from gluomem.config.app_config import AppConfig

SLUG = "openrouter/deepseek/deepseek-v4-flash"


class FakeMem0Memory:
    """Records from_config/add calls; scripted search results (class-level)."""

    instances: list[FakeMem0Memory] = []
    search_results: list[dict] = []

    def __init__(self, config: dict) -> None:
        self.config = config
        self.add_calls: list[dict] = []
        type(self).instances.append(self)

    @classmethod
    def from_config(cls, config: dict) -> FakeMem0Memory:
        return cls(config)

    def add(self, messages, *, user_id=None, metadata=None, infer=True):
        self.add_calls.append(
            {
                "messages": messages,
                "user_id": user_id,
                "metadata": metadata,
                "infer": infer,
            }
        )
        return {"results": [{"memory": "m"}] * 2}  # 2 memories per add

    def search(self, query, *, filters=None, top_k=20):
        self.search_calls = getattr(self, "search_calls", [])
        self.search_calls.append(
            {"query": query, "filters": filters, "top_k": top_k}
        )
        return {"results": list(type(self).search_results)}


@pytest.fixture
def fake_mem0(monkeypatch: pytest.MonkeyPatch) -> type[FakeMem0Memory]:
    FakeMem0Memory.instances = []
    FakeMem0Memory.search_results = []
    monkeypatch.setitem(
        sys.modules, "mem0", types.SimpleNamespace(Memory=FakeMem0Memory)
    )
    return FakeMem0Memory


def _sample(question: str = "Where does Bob live?") -> Sample:
    return Sample(
        id="conv-tiny-1:qa:0",
        conv_id="conv-tiny-1",
        question=question,
        gold="Seattle",
        category=4,
        category_name="single-hop",
    )


# --- import discipline + missing extra ----------------------------------------


def test_import_mem0_module_is_dependency_free() -> None:
    """Importing gluomem.bench.mem0 pulls neither mem0 nor litellm nor dspy."""
    code = (
        "import sys; import gluomem.bench.mem0; "
        "bad = {m for m in ('mem0', 'litellm', 'dspy') if m in sys.modules}; "
        "sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).parents[2]),
    )
    assert proc.returncode == 0, proc.stderr


def test_missing_extra_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch, bench_cfg: AppConfig, tmp_path: Path, answer_fn
) -> None:
    monkeypatch.setitem(sys.modules, "mem0", None)  # forces ImportError
    with pytest.raises(RuntimeError, match="uv sync --extra mem0"):
        Mem0Retriever(bench_cfg, db_dir=tmp_path, answer_fn=answer_fn)


# --- config builder --------------------------------------------------------------


def test_build_mem0_config_maps_registry(bench_cfg: AppConfig, tmp_path: Path) -> None:
    config = build_mem0_config(bench_cfg, conv_dir=tmp_path / "conv-1")
    assert config["llm"] == {
        "provider": "litellm",
        "config": {"model": SLUG},
    }
    assert config["embedder"]["provider"] == "huggingface"
    assert (
        config["embedder"]["config"]["model"]
        == "sentence-transformers/all-MiniLM-L6-v2"
    )
    vs = config["vector_store"]
    # qdrant local mode = mem0's default store (faiss silently disables
    # their hybrid BM25 scoring).
    assert vs["provider"] == "qdrant"
    assert vs["config"]["embedding_model_dims"] == 384
    assert vs["config"]["path"] == str(tmp_path / "conv-1" / "qdrant")
    assert vs["config"]["on_disk"] is True
    assert config["history_db_path"] == str(tmp_path / "conv-1" / "history.db")


# --- ingest -----------------------------------------------------------------------


def test_ingest_adds_per_session_with_dates_and_metadata(
    fake_mem0: type[FakeMem0Memory], bench_cfg: AppConfig, tmp_path: Path
) -> None:
    reports = ingest_mem0_per_conversation(
        bench_cfg, dataset="locomo", db_dir=tmp_path / "mem0"
    )
    (report,) = reports
    (instance,) = fake_mem0.instances
    # One add() per session (3 in the fixture).
    assert len(instance.add_calls) == 3
    first = instance.add_calls[0]
    assert first["user_id"] == "conv-tiny-1"
    assert first["infer"] is True
    assert first["metadata"] == {
        "session_index": 1,
        "session_date": "2023-05-08T13:56:00",
    }
    # Speaker- and date-prefixed content; all turns role=user.
    assert all(m["role"] == "user" for m in first["messages"])
    assert first["messages"][0]["content"].startswith(
        "[2023-05-08T13:56:00] Alice:"
    )
    assert report.sessions == 3 and report.turns == 9
    assert report.memories == 6  # fake yields 2 per add
    # Sentinel written with telemetry.
    meta = json.loads(
        (tmp_path / "mem0" / "conv-tiny-1" / "ingest_meta.json").read_text()
    )
    assert meta["memories"] == 6
    assert meta["usage_measured"] is False  # no litellm traffic in the fake
    assert meta["cost_usd"] is None
    assert "mem0_config" in meta and "versions" in meta


def test_ingest_infer_flag_honored(
    fake_mem0: type[FakeMem0Memory], bench_cfg: AppConfig, tmp_path: Path
) -> None:
    cfg = bench_cfg.with_overrides({"benchmark.mem0.infer": False})
    ingest_mem0_per_conversation(cfg, dataset="locomo", db_dir=tmp_path / "m")
    assert fake_mem0.instances[0].add_calls[0]["infer"] is False


def test_ingest_skip_existing_and_partial_dir(
    fake_mem0: type[FakeMem0Memory], bench_cfg: AppConfig, tmp_path: Path
) -> None:
    db_dir = tmp_path / "mem0"
    ingest_mem0_per_conversation(bench_cfg, dataset="locomo", db_dir=db_dir)
    # Completed dir without --skip-existing -> refuses (add() not idempotent).
    with pytest.raises(RuntimeError, match="skip-existing"):
        ingest_mem0_per_conversation(bench_cfg, dataset="locomo", db_dir=db_dir)
    # With the flag: skipped, no new store built.
    before = len(fake_mem0.instances)
    again = ingest_mem0_per_conversation(
        bench_cfg, dataset="locomo", db_dir=db_dir, skip_existing=True
    )
    assert again == [] and len(fake_mem0.instances) == before
    # Partial dir (sentinel missing) fails loud even with the flag.
    (db_dir / "conv-tiny-1" / "ingest_meta.json").unlink()
    with pytest.raises(RuntimeError, match="partial ingest"):
        ingest_mem0_per_conversation(
            bench_cfg, dataset="locomo", db_dir=db_dir, skip_existing=True
        )


# --- usage tap --------------------------------------------------------------------


def test_reasoning_off_shim_injects_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gluomem.bench.mem0 import LiteLLMReasoningOff

    calls: list[dict] = []

    def fake_completion(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace()

    fake_litellm = types.SimpleNamespace(completion=fake_completion)
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    with LiteLLMReasoningOff(slugs=frozenset({SLUG})):
        fake_litellm.completion(model=SLUG, messages=[])
        fake_litellm.completion(model="openrouter/x/other", messages=[])
        # An explicit caller-provided reasoning setting is never overridden.
        fake_litellm.completion(
            model=SLUG, messages=[], extra_body={"reasoning": {"enabled": True}}
        )
    assert fake_litellm.completion is fake_completion  # restored on exit
    assert calls[0]["extra_body"] == {"reasoning": {"enabled": False}}
    assert "extra_body" not in calls[1]  # unrelated slug untouched
    assert calls[2]["extra_body"] == {"reasoning": {"enabled": True}}


def test_usage_tap_accumulates_and_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_litellm = types.SimpleNamespace(success_callback=[])
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    with LiteLLMUsageTap(slugs=frozenset({SLUG})) as acc:
        assert len(fake_litellm.success_callback) == 1
        cb = fake_litellm.success_callback[0]
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20)
        cb({"model": SLUG}, SimpleNamespace(usage=usage), None, None)
        # litellm sometimes reports the bare model name (provider stripped).
        cb(
            {"model": "deepseek/deepseek-v4-flash"},
            SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3)),
            None,
            None,
        )
        # Unrelated slug filtered out.
        cb({"model": "openrouter/x/other"}, SimpleNamespace(usage=usage), None, None)
    assert fake_litellm.success_callback == []  # removed on exit
    snap = acc.snapshot()
    assert snap == {SLUG: {"prompt_tokens": 107, "completion_tokens": 23}}


# --- retriever -------------------------------------------------------------------


def _ingested(
    fake_mem0: type[FakeMem0Memory], cfg: AppConfig, tmp_path: Path
) -> Path:
    db_dir = tmp_path / "mem0"
    ingest_mem0_per_conversation(cfg, dataset="locomo", db_dir=db_dir)
    return db_dir


def test_retriever_maps_results_and_prefixes_dates(
    fake_mem0: type[FakeMem0Memory],
    bench_cfg: AppConfig,
    tmp_path: Path,
    answer_fn,
) -> None:
    db_dir = _ingested(fake_mem0, bench_cfg, tmp_path)
    fake_mem0.search_results = [
        {
            "memory": "Bob moved to Seattle",
            "score": 0.9,
            "metadata": {"session_date": "2023-08-02T19:30:00"},
        },
        {"memory": "Bob lived in Boston", "score": 0.5, "created_at": "2026-07-05"},
    ]
    r = Mem0Retriever(bench_cfg, db_dir=db_dir, answer_fn=answer_fn)
    p = r.answer(_sample())
    assert p.answer == "scripted answer"
    assert [d.text for d in p.retrieved] == [
        "Bob moved to Seattle",
        "Bob lived in Boston",
    ]
    assert all(d.dia_id is None for d in p.retrieved)  # distilled memories
    assert p.retrieved[0].score == 0.9
    (_, context), = answer_fn.calls
    lines = context.splitlines()
    assert lines[0].startswith("[2023-08-02T19:30:00]")  # metadata date wins
    assert lines[1].startswith("[2026-07-05]")  # created_at fallback
    # top_k defaults to answer.context_k.
    instance = fake_mem0.instances[-1]
    assert instance.search_calls[0]["top_k"] == bench_cfg.benchmark.answer.context_k
    assert instance.search_calls[0]["filters"] == {"user_id": "conv-tiny-1"}


def test_retriever_top_k_override_wins(
    fake_mem0: type[FakeMem0Memory], bench_cfg: AppConfig, tmp_path: Path, answer_fn
) -> None:
    db_dir = _ingested(fake_mem0, bench_cfg, tmp_path)
    cfg = bench_cfg.with_overrides({"benchmark.mem0.top_k": 3})
    fake_mem0.search_results = [{"memory": "x", "score": 0.1}]
    r = Mem0Retriever(cfg, db_dir=db_dir, answer_fn=answer_fn)
    r.answer(_sample())
    assert fake_mem0.instances[-1].search_calls[0]["top_k"] == 3


def test_retriever_empty_results_abstains_without_llm_call(
    fake_mem0: type[FakeMem0Memory], bench_cfg: AppConfig, tmp_path: Path, answer_fn
) -> None:
    db_dir = _ingested(fake_mem0, bench_cfg, tmp_path)
    fake_mem0.search_results = []
    r = Mem0Retriever(bench_cfg, db_dir=db_dir, answer_fn=answer_fn)
    p = r.answer(_sample())
    assert p.abstained is True
    assert p.answer == bench_cfg.benchmark.answer.abstain_text
    assert answer_fn.calls == []


def test_retriever_missing_ingest_names_cli(
    fake_mem0: type[FakeMem0Memory], bench_cfg: AppConfig, tmp_path: Path, answer_fn
) -> None:
    r = Mem0Retriever(bench_cfg, db_dir=tmp_path / "empty", answer_fn=answer_fn)
    with pytest.raises(FileNotFoundError, match="--system mem0"):
        r.answer(_sample())


def test_manifest_extra_records_versions_and_config(
    fake_mem0: type[FakeMem0Memory], bench_cfg: AppConfig, tmp_path: Path, answer_fn
) -> None:
    db_dir = _ingested(fake_mem0, bench_cfg, tmp_path)
    r = Mem0Retriever(bench_cfg, db_dir=db_dir, answer_fn=answer_fn)
    extra = r.manifest_extra()
    assert extra["system"] == "mem0-oss"
    assert "mem0ai" in extra["versions"]
    assert extra["mem0_config"]["llm"]["config"]["model"] == SLUG
    assert extra["search_params"]["threshold"] == "mem0-default"
    assert "conv-tiny-1" in extra["ingest_meta"]


# --- runner e2e --------------------------------------------------------------------


def test_bench_run_mem0_offline_e2e(
    fake_mem0: type[FakeMem0Memory], bench_cfg: AppConfig, tmp_path: Path
) -> None:
    from gluomem.bench.runner import run_benchmark
    from tests.bench.test_runner import FAKE_BACKENDS

    db_dir = _ingested(fake_mem0, bench_cfg, tmp_path)
    fake_mem0.search_results = [
        {"memory": "Alice adopted a cat", "score": 0.9,
         "metadata": {"session_date": "2023-05-08"}},
    ]
    cfg = bench_cfg.with_overrides(dict(FAKE_BACKENDS))
    paths = run_benchmark(
        cfg,
        dataset="locomo",
        retriever="mem0",
        db_dir=db_dir,
        out_root=tmp_path / "runs",
    )
    manifest = json.loads(paths.manifest.read_text())
    assert manifest["retriever"] == "mem0"
    assert manifest["retriever_info"]["system"] == "mem0-oss"
    entries = json.loads(paths.scores.read_text())
    trio = [
        e for e in entries
        if e["scope"] == "overall" and e["tag"]["judge_family"] == "trio_strict"
    ]
    assert len(trio) == 1


def test_run_mem0_without_db_dir_fails(
    fake_mem0: type[FakeMem0Memory], bench_cfg: AppConfig, tmp_path: Path
) -> None:
    from gluomem.bench.runner import run_benchmark
    from tests.bench.test_runner import FAKE_BACKENDS

    cfg = bench_cfg.with_overrides(dict(FAKE_BACKENDS))
    with pytest.raises(ValueError, match="db-dir"):
        run_benchmark(
            cfg, dataset="locomo", retriever="mem0", out_root=tmp_path
        )


def test_baseline_manifest_retriever_info_null(
    bench_cfg: AppConfig, tmp_path: Path
) -> None:
    from gluomem.bench.runner import run_benchmark
    from tests.bench.test_runner import FAKE_BACKENDS

    cfg = bench_cfg.with_overrides(dict(FAKE_BACKENDS))
    paths = run_benchmark(
        cfg, dataset="locomo", retriever="bm25", out_root=tmp_path / "runs"
    )
    manifest = json.loads(paths.manifest.read_text())
    assert manifest["retriever_info"] is None
