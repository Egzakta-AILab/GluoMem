"""Mem0 integration — config builder, ingest, retriever, e2e runner, and live smoke."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import dotenv
import pytest

from gluomem.bench.mem0 import (
    Mem0Retriever,
    build_mem0_config,
    ingest_mem0_per_conversation,
)
from gluomem.bench.runner import run_benchmark
from gluomem.bench.types import Sample
from gluomem.config.app_config import AppConfig, load_config
from tests.bench.conftest import FAKE_BACKENDS, FIXTURE

dotenv.load_dotenv(Path(__file__).parents[2] / ".env", override=False)

MODEL_ID = "deepseek/deepseek-v4-flash"


class FakeMem0Memory:
    """Records from_config/add/search calls with scripted search results."""

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
        return {"results": [{"memory": "m"}] * 2}

    def search(self, query, *, filters=None, top_k=20):
        self.search_calls = getattr(self, "search_calls", [])
        self.search_calls.append({"query": query, "filters": filters, "top_k": top_k})
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


def test_import_mem0_module_and_missing_extra(
    monkeypatch: pytest.MonkeyPatch, bench_cfg: AppConfig, tmp_path: Path, answer_fn
) -> None:
    code = (
        "import sys; import gluomem.bench.mem0; "
        "bad = {m for m in ('mem0', 'dspy') if m in sys.modules}; "
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

    monkeypatch.setitem(sys.modules, "mem0", None)
    with pytest.raises(RuntimeError, match="uv sync --extra mem0"):
        Mem0Retriever(bench_cfg, db_dir=tmp_path, answer_fn=answer_fn)


def test_build_mem0_config_maps_registry_to_openai_provider(
    bench_cfg: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
    config = build_mem0_config(bench_cfg, conv_dir=tmp_path / "conv-1")
    assert config["llm"] == {
        "provider": "openai",
        "config": {
            "model": MODEL_ID,
            "openrouter_base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-test-123",
        },
    }
    assert config["embedder"]["provider"] == "huggingface"
    assert config["embedder"]["config"]["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    vs = config["vector_store"]
    assert vs["provider"] == "qdrant"
    assert vs["config"]["embedding_model_dims"] == 384
    assert vs["config"]["path"] == str(tmp_path / "conv-1" / "qdrant")
    assert vs["config"]["on_disk"] is True
    assert config["history_db_path"] == str(tmp_path / "conv-1" / "history.db")


def test_mem0_ingest_sessions_infer_flag_and_skip_existing(
    fake_mem0: type[FakeMem0Memory], bench_cfg: AppConfig, tmp_path: Path
) -> None:
    db_dir = tmp_path / "mem0"
    (report,) = ingest_mem0_per_conversation(bench_cfg, dataset="locomo", db_dir=db_dir)
    (instance,) = fake_mem0.instances
    assert len(instance.add_calls) == 3
    first = instance.add_calls[0]
    assert first["user_id"] == "conv-tiny-1" and first["infer"] is True
    assert first["metadata"] == {"session_index": 1, "session_date": "2023-05-08T13:56:00"}
    assert all(m["role"] == "user" for m in first["messages"])
    assert first["messages"][0]["content"].startswith("[2023-05-08T13:56:00] Alice:")
    assert report.sessions == 3 and report.turns == 9 and report.memories == 6

    meta = json.loads((db_dir / "conv-tiny-1" / "ingest_meta.json").read_text())
    assert meta["memories"] == 6 and meta["usage_measured"] is False

    with pytest.raises(RuntimeError, match="skip-existing"):
        ingest_mem0_per_conversation(bench_cfg, dataset="locomo", db_dir=db_dir)
    assert (
        ingest_mem0_per_conversation(
            bench_cfg, dataset="locomo", db_dir=db_dir, skip_existing=True
        )
        == []
    )

    (db_dir / "conv-tiny-1" / "ingest_meta.json").unlink()
    with pytest.raises(RuntimeError, match="partial ingest"):
        ingest_mem0_per_conversation(
            bench_cfg, dataset="locomo", db_dir=db_dir, skip_existing=True
        )

    cfg_no_infer = bench_cfg.with_overrides({"benchmark.mem0.infer": False})
    ingest_mem0_per_conversation(cfg_no_infer, dataset="locomo", db_dir=tmp_path / "m2")
    assert fake_mem0.instances[-1].add_calls[0]["infer"] is False


def test_mem0_retriever_and_bench_run_e2e(
    fake_mem0: type[FakeMem0Memory],
    bench_cfg: AppConfig,
    tmp_path: Path,
    answer_fn,
) -> None:
    db_dir = tmp_path / "mem0"
    ingest_mem0_per_conversation(bench_cfg, dataset="locomo", db_dir=db_dir)
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
    assert [d.text for d in p.retrieved] == ["Bob moved to Seattle", "Bob lived in Boston"]
    (_, context), = answer_fn.calls
    assert context.splitlines()[0].startswith("[2023-08-02T19:30:00]")
    assert context.splitlines()[1].startswith("[2026-07-05]")

    extra = r.manifest_extra()
    assert extra["system"] == "mem0-oss"
    assert extra["mem0_config"]["llm"]["config"]["model"] == MODEL_ID

    r_topk = Mem0Retriever(
        bench_cfg.with_overrides({"benchmark.mem0.top_k": 3}),
        db_dir=db_dir,
        answer_fn=answer_fn,
    )
    r_topk.answer(_sample())
    assert fake_mem0.instances[-1].search_calls[0]["top_k"] == 3

    fake_mem0.search_results = []
    answer_fn.calls.clear()
    p_empty = r.answer(_sample())
    assert p_empty.abstained is True and answer_fn.calls == []

    r_missing = Mem0Retriever(bench_cfg, db_dir=tmp_path / "empty", answer_fn=answer_fn)
    with pytest.raises(FileNotFoundError, match="--system mem0"):
        r_missing.answer(_sample())

    fake_mem0.search_results = [
        {"memory": "Alice adopted a cat", "score": 0.9, "metadata": {"session_date": "2023-05-08"}}
    ]
    cfg = bench_cfg.with_overrides(dict(FAKE_BACKENDS))
    paths = run_benchmark(
        cfg, dataset="locomo", retriever="mem0", db_dir=db_dir, out_root=tmp_path / "runs"
    )
    manifest = json.loads(paths.manifest.read_text())
    assert manifest["retriever"] == "mem0"
    assert manifest["retriever_info"]["system"] == "mem0-oss"

    with pytest.raises(ValueError, match="db-dir"):
        run_benchmark(cfg, dataset="locomo", retriever="mem0", out_root=tmp_path)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)
def test_live_mem0_ingest_search_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("mem0", reason="uv sync --extra mem0 not installed")
    from gluomem.llm.answer import Answerer

    cfg = load_config().with_overrides(
        {
            "benchmark.datasets_dir": str(FIXTURE.parent),
            "benchmark.datasets.locomo.file": FIXTURE.name,
            "benchmark.datasets.locomo.sha256": hashlib.sha256(
                FIXTURE.read_bytes()
            ).hexdigest(),
        }
    )
    db_dir = tmp_path / "mem0_stores"
    (report,) = ingest_mem0_per_conversation(cfg, dataset="locomo", db_dir=db_dir)
    assert report.memories > 0

    retriever = Mem0Retriever(cfg, db_dir=db_dir, answer_fn=Answerer(cfg))
    prediction = retriever.answer(_sample("Where does Bob live now?"))
    assert not prediction.abstained and prediction.retrieved
