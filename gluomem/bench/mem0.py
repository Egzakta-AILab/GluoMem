"""mem0 OSS head-to-head baseline (ingest + Retriever)."""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gluomem.bench.adapters import make_adapter
from gluomem.bench.types import Prediction, RetrievedDoc, Sample

if TYPE_CHECKING:
    from gluomem.bench.types import AnswerFn
    from gluomem.config.app_config import AppConfig
    from gluomem.write.ingest import Session

_SENTINEL = "ingest_meta.json"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _import_memory_cls() -> Any:
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    try:
        from mem0 import Memory
    except ImportError as exc:
        raise RuntimeError(
            "the 'mem0' retriever/system needs the optional dependency "
            "group — run `uv sync --extra mem0`"
        ) from exc
    return Memory


def _installed_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in ("mem0ai", "qdrant-client"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "not-installed"
    return out


def _provider_model_id(slug: str) -> str:
    if slug.startswith("openrouter/"):
        return slug.removeprefix("openrouter/")
    return slug


def build_mem0_config(cfg: AppConfig, *, conv_dir: Path) -> dict[str, Any]:
    """Build the ``Memory.from_config`` dict for one conversation store."""
    llm_entry = cfg.models[cfg.benchmark.mem0.model]
    embed_entry = cfg.models[cfg.benchmark.mem0.embedder_model]
    llm_cfg: dict[str, Any] = {
        "model": _provider_model_id(llm_entry.provider_model),
        "openrouter_base_url": llm_entry.api_base or _OPENROUTER_BASE_URL,
    }
    if llm_entry.api_key_env:
        api_key = os.environ.get(llm_entry.api_key_env)
        if api_key:
            llm_cfg["api_key"] = api_key
    return {
        "llm": {
            "provider": "openai",
            "config": llm_cfg,
        },
        "embedder": {
            "provider": "huggingface",
            "config": {"model": embed_entry.provider_model},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "mem0",
                "path": str(conv_dir / "qdrant"),
                "on_disk": True,
                "embedding_model_dims": embed_entry.dim,
            },
        },
        "history_db_path": str(conv_dir / "history.db"),
    }


def _session_messages(session: Session) -> list[dict[str, str]]:
    iso = session.date_time.isoformat()
    return [
        {"role": "user", "content": f"[{iso}] {turn.speaker}: {turn.text}"}
        for turn in session.turns
    ]


@dataclass(frozen=True)
class Mem0IngestReport:
    conv_id: str
    sessions: int
    turns: int
    memories: int
    seconds: float
    usage: dict[str, dict[str, int]]
    cost_usd: float | None


def _memories_added(add_result: Any) -> int:
    if isinstance(add_result, dict):
        results = add_result.get("results")
        if isinstance(results, list):
            return len(results)
    return 0


def ingest_mem0_per_conversation(
    cfg: AppConfig,
    *,
    dataset: str,
    db_dir: Path,
    skip_existing: bool = False,
    conv_ids: set[str] | None = None,
) -> list[Mem0IngestReport]:
    """Ingest one mem0 store per conversation under ``<db_dir>/<conv_id>/``."""
    memory_cls = _import_memory_cls()
    adapter = make_adapter(dataset, cfg)
    adapter.verify()
    db_dir.mkdir(parents=True, exist_ok=True)
    batch_started = time.perf_counter()

    convs = adapter.conversations()
    if conv_ids is not None:
        convs = [c for c in convs if c.conv_id in conv_ids]

    todo: list[Any] = []
    for conv in convs:
        conv_dir = db_dir / conv.conv_id
        sentinel = conv_dir / _SENTINEL
        if conv_dir.exists():
            if sentinel.is_file():
                if skip_existing:
                    print(f"skipped {conv.conv_id}: {sentinel} exists")
                    continue
                raise RuntimeError(
                    f"{conv_dir} already ingested; pass --skip-existing to "
                    f"resume around it or delete the directory to redo"
                )
            raise RuntimeError(
                f"{conv_dir} exists without {_SENTINEL} — partial ingest; "
                f"delete the directory and rerun"
            )
        todo.append(conv)

    workers = min(cfg.write.conv_parallelism, max(1, len(todo)))
    parallel = workers > 1 and len(todo) > 1

    def _ingest_one(conv: Any) -> Mem0IngestReport:
        conv_dir = db_dir / conv.conv_id
        conv_dir.mkdir(parents=True)
        memory = memory_cls.from_config(build_mem0_config(cfg, conv_dir=conv_dir))
        started = time.perf_counter()
        memories = 0
        turns = 0
        for session in conv.sessions:
            messages = _session_messages(session)
            turns += len(messages)
            result = memory.add(
                messages,
                user_id=conv.conv_id,
                metadata={
                    "session_index": session.index,
                    "session_date": session.date_time.isoformat(),
                },
                infer=cfg.benchmark.mem0.infer,
            )
            memories += _memories_added(result)
        seconds = time.perf_counter() - started
        report = Mem0IngestReport(
            conv_id=conv.conv_id,
            sessions=len(conv.sessions),
            turns=turns,
            memories=memories,
            seconds=seconds,
            usage={},
            cost_usd=None,
        )
        meta = {
            "conv_id": conv.conv_id,
            "sessions": report.sessions,
            "turns": report.turns,
            "memories": report.memories,
            "seconds": round(seconds, 3),
            "usage": {},
            "usage_measured": False,
            "cost_usd": None,
            "versions": _installed_versions(),
            "mem0_config": build_mem0_config(cfg, conv_dir=conv_dir),
        }
        sentinel = conv_dir / _SENTINEL
        tmp = sentinel.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        tmp.replace(sentinel)
        print(
            f"ingested {conv.conv_id} (mem0): {report.memories} memories from "
            f"{report.turns} turns / {report.sessions} sessions | {seconds:.0f}s",
            flush=True,
        )
        return report

    reports_by_id: dict[str, Mem0IngestReport] = {}
    failures: list[str] = []
    if parallel:
        from concurrent.futures import ThreadPoolExecutor

        def _run(conv: Any) -> None:
            try:
                reports_by_id[conv.conv_id] = _ingest_one(conv)
            except Exception as exc:  # noqa: BLE001
                failures.append(conv.conv_id)
                print(
                    f"mem0 ingest failed for {conv.conv_id}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_run, todo))
    else:
        for conv in todo:
            reports_by_id[conv.conv_id] = _ingest_one(conv)

    if failures:
        print(f"mem0 ingest failures: {sorted(failures)}", file=sys.stderr)
    reports = [reports_by_id[c.conv_id] for c in todo if c.conv_id in reports_by_id]
    total = sum(r.memories for r in reports)
    print(
        f"mem0 ingest total: {len(reports)} conversation(s), {total} memories "
        f"in {time.perf_counter() - batch_started:.0f}s"
    )
    return reports


class Mem0Retriever:
    """``Retriever`` protocol implementation over per-conversation mem0 stores."""

    name = "mem0"

    def __init__(self, cfg: AppConfig, *, db_dir: Path, answer_fn: AnswerFn) -> None:
        self._memory_cls = _import_memory_cls()
        self._cfg = cfg
        self._db_dir = Path(db_dir)
        self._answer_fn = answer_fn
        self._memories: dict[str, Any] = {}
        self._lock = threading.Lock()

    @property
    def _top_k(self) -> int:
        return (
            self._cfg.benchmark.mem0.top_k
            if self._cfg.benchmark.mem0.top_k is not None
            else self._cfg.benchmark.answer.context_k
        )

    def _memory_for(self, conv_id: str) -> Any:
        memory = self._memories.get(conv_id)
        if memory is None:
            conv_dir = self._db_dir / conv_id
            if not (conv_dir / _SENTINEL).is_file():
                raise FileNotFoundError(
                    f"no completed mem0 store for conversation '{conv_id}' at "
                    f"{conv_dir} — run `gluomem bench ingest --dataset locomo "
                    f"--system mem0 --db-dir {self._db_dir}` first"
                )
            memory = self._memory_cls.from_config(build_mem0_config(self._cfg, conv_dir=conv_dir))
            self._memories[conv_id] = memory
        return memory

    def answer(self, s: Sample) -> Prediction:
        start = time.perf_counter()
        with self._lock:
            memory = self._memory_for(s.conv_id)
            response = memory.search(
                s.question,
                filters={"user_id": s.conv_id},
                top_k=self._top_k,
            )
        results = response.get("results", []) if isinstance(response, dict) else []
        if not results:
            return Prediction(
                sample_id=s.id,
                answer=self._cfg.benchmark.answer.abstain_text,
                abstained=True,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                abstention={"reason": "no_hits", "candidates": 0},
            )
        docs: list[RetrievedDoc] = []
        lines: list[str] = []
        for rank, row in enumerate(results):
            text = str(row.get("memory", ""))
            metadata = row.get("metadata") or {}
            date = metadata.get("session_date") or row.get("created_at") or ""
            docs.append(
                RetrievedDoc(
                    rank=rank,
                    text=text,
                    score=float(row.get("score") or 0.0),
                )
            )
            lines.append(f"[{date}] {text}")
        out = self._answer_fn(s.question, "\n".join(lines))
        return Prediction(
            sample_id=s.id,
            answer=out.text,
            retrieved=tuple(docs),
            prompt_tokens=out.prompt_tokens,
            completion_tokens=out.completion_tokens,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )

    def manifest_extra(self) -> dict[str, Any]:
        """Provenance for the run manifest."""
        ingest_meta: dict[str, Any] = {}
        if self._db_dir.is_dir():
            for sentinel in sorted(self._db_dir.glob(f"*/{_SENTINEL}")):
                meta = json.loads(sentinel.read_text(encoding="utf-8"))
                ingest_meta[meta.get("conv_id", sentinel.parent.name)] = meta
        return {
            "system": "mem0-oss",
            "versions": _installed_versions(),
            "mem0_config": build_mem0_config(self._cfg, conv_dir=self._db_dir / "<conv_id>"),
            "llm_policy": "openai-compatible OpenRouter endpoint",
            "search_params": {
                "top_k": self._top_k,
                "threshold": "mem0-default",
                "rerank": "mem0-default",
            },
            "identity_mapping": (
                "user_id=conv_id; all turns role=user; content "
                "'[session_iso] Speaker: text'; session date also in metadata"
            ),
            "evidence_metrics": "not-applicable (distilled memories, no dia_ids)",
            "ingest_meta": ingest_meta,
        }


__all__ = [
    "Mem0IngestReport",
    "Mem0Retriever",
    "build_mem0_config",
    "ingest_mem0_per_conversation",
]
