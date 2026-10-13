"""gluomem.bench.mem0 — Mem0 OSS as a head-to-head Retriever.

Fairness contract (recorded in every run manifest):
- Extraction backbone = the SAME registry model as gluomem's
  (``benchmark.mem0.model``, default deepseek-flash) via mem0's litellm
  provider; answerer, judges, sample and dataset are identical to every
  other retriever in the harness — the memory architecture is the only
  variable.
- Everything else runs mem0 DEFAULTS: their extraction/update pipeline
  (``infer``), their search threshold/rerank (not passed). One forced
  deviation: embedder = local MiniLM via the huggingface provider — mem0's
  OpenAI-embeddings default needs an OpenAI key this project does not use,
  and MiniLM also matches gluomem's own embedder.
- Versions pinned (``uv sync --extra mem0``: mem0ai + faiss-cpu exact pins
  in pyproject); versions + effective config land in ``manifest_extra``.

Identity mapping for two-party LoCoMo conversations: one mem0 store per
conversation (``user_id = conv_id``); every turn ``role="user"`` with
speaker-prefixed, session-date-prefixed content (``[iso] Speaker: text``) —
OSS ``add()`` has no timestamp parameter, so dates in content are the only
way mem0's extractor can resolve temporal facts (this favors mem0: 80/320
sampled questions are temporal).

Known limitation: mem0 returns distilled memories, not turns — retrieved
docs carry no dia_id, so evidence metrics are structurally zero for mem0
runs. qa_accuracy is the headline; NEVER apply the CI gate to a mem0 run.

Import discipline: importing this module imports neither ``mem0`` nor
``litellm`` nor ``dspy`` (subprocess-asserted); the offline gate passes
without the optional extra installed.
"""

from __future__ import annotations

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
from gluomem.llm.usage import UsageAccumulator, total_tokens, usage_delta

if TYPE_CHECKING:
    from gluomem.bench.types import AnswerFn
    from gluomem.config.app_config import AppConfig
    from gluomem.write.ingest import Session

MEM0_EXTRA_HINT = "run `uv sync --extra mem0` (pins mem0ai==2.0.11)"
_SENTINEL = "ingest_meta.json"


def _import_memory_cls() -> Any:
    """Lazy mem0 import with an actionable error; disables their telemetry."""
    # posthog telemetry is ON by default — a hidden network call.
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    try:
        from mem0 import Memory
    except ImportError as exc:
        raise RuntimeError(
            f"the 'mem0' retriever/system needs the optional dependency "
            f"group — {MEM0_EXTRA_HINT}"
        ) from exc
    return Memory


def _installed_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for pkg in ("mem0ai", "faiss-cpu"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def build_mem0_config(cfg: AppConfig, *, conv_dir: Path) -> dict[str, Any]:
    """The mem0 ``Memory.from_config`` dict for one conversation store.

    Pure (no mem0 import) so the mapping is offline-testable; recorded
    verbatim (path templated) in the run manifest.

    Vector store = qdrant in serverless local-path mode — mem0's DEFAULT
    store, shipped inside mem0ai's core deps. Not faiss: mem0 logs faiss as
    "hybrid (BM25) scoring will be disabled", which would silently run the
    competitor with its hybrid search off — a fairness problem.
    """
    llm_entry = cfg.models[cfg.benchmark.mem0.model]
    embed_entry = cfg.models[cfg.benchmark.mem0.embedder_model]
    return {
        "llm": {
            "provider": "litellm",
            "config": {"model": llm_entry.provider_model},
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
    """All turns role="user", speaker- and session-date-prefixed."""
    iso = session.date_time.isoformat()
    return [
        {"role": "user", "content": f"[{iso}] {turn.speaker}: {turn.text}"}
        for turn in session.turns
    ]


class LiteLLMUsageTap:
    """Temporary litellm success-callback capturing token usage.

    mem0 exposes no usage API, but its litellm provider routes every
    extraction/update call through in-process ``litellm.completion`` —
    a success callback sees each response's ``usage``. Filtered to the
    configured slugs so unrelated in-process litellm traffic never
    contaminates the numbers. Best-effort by design: on failure the
    accumulator simply stays empty and cost reports as null.
    """

    def __init__(self, *, slugs: frozenset[str]) -> None:
        self._slugs = slugs
        self.usage = UsageAccumulator()
        self._callback: Any = None

    def __enter__(self) -> UsageAccumulator:
        import litellm

        def _capture(
            kwargs: dict[str, Any],
            completion_response: Any,
            start_time: Any,  # noqa: ARG001
            end_time: Any,  # noqa: ARG001
        ) -> None:
            model = str(kwargs.get("model") or "")
            matched = next(
                (s for s in self._slugs if s == model or s.endswith(model)),
                None,
            )
            if matched is None:
                return
            usage = getattr(completion_response, "usage", None)
            if usage is None:
                return
            self.usage.add(
                {
                    matched: {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(
                            usage, "completion_tokens", 0
                        ),
                    }
                }
            )

        self._callback = _capture
        litellm.success_callback.append(_capture)
        return self.usage

    def __exit__(self, *exc: object) -> None:
        import litellm

        try:
            litellm.success_callback.remove(self._callback)
        except ValueError:
            pass


class LiteLLMReasoningOff:
    """Disable OpenRouter hidden reasoning for mem0's litellm calls.

    gluomem's router sends ``extra_body={"reasoning": {"enabled": false}}``
    on every call — deepseek-v4-flash otherwise burns 5-12x hidden reasoning
    tokens and, under mem0's max_tokens=2000 default, returns content=None
    ("'NoneType' object has no attribute 'strip'" in mem0's extraction).
    mem0's litellm provider passes no extra_body, so the SAME policy is
    applied by wrapping ``litellm.completion`` for the configured slugs;
    restored on exit. This is backbone parity, not a mem0-pipeline change.
    """

    def __init__(self, *, slugs: frozenset[str]) -> None:
        self._slugs = slugs
        self._original: Any = None

    def __enter__(self) -> None:
        import litellm

        original: Any = litellm.completion

        def patched(*args: Any, **kwargs: Any) -> Any:
            model = str(kwargs.get("model") or "")
            if any(s == model or s.endswith(model) for s in self._slugs):
                extra = dict(kwargs.get("extra_body") or {})
                extra.setdefault("reasoning", {"enabled": False})
                kwargs["extra_body"] = extra
            return original(*args, **kwargs)

        self._original = original
        litellm.completion = patched

    def __exit__(self, *exc: object) -> None:
        import litellm

        if self._original is not None:
            litellm.completion = self._original


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
    """One mem0 store per conversation: ``<db_dir>/<conv_id>/``.

    The ``ingest_meta.json`` sentinel is written only after the last
    session lands — it is the resume marker (a dir WITHOUT it is a partial
    ingest and must be deleted) and the per-conversation telemetry record.
    """
    from gluomem.bench.runner import usage_cost

    memory_cls = _import_memory_cls()
    adapter = make_adapter(dataset, cfg)
    adapter.verify()
    db_dir.mkdir(parents=True, exist_ok=True)
    slug = cfg.models[cfg.benchmark.mem0.model].provider_model
    slugs = frozenset({slug})
    batch_started = time.perf_counter()

    convs = adapter.conversations()
    if conv_ids is not None:
        convs = [c for c in convs if c.conv_id in conv_ids]

    # Sentinel screening happens up front; the pool below only ever sees
    # convs that need ingesting.
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

    # Cross-conversation parallelism (write.conv_parallelism, same dial as the
    # gluomem ingest): each conversation is an independent mem0 store, so convs
    # ingest concurrently. The litellm callbacks are process-global, so under
    # parallelism ONE reasoning-off + usage tap wraps the whole pool and token
    # usage is batch-attributed (per-conv usage_measured=false); the sequential
    # path (conv_parallelism=1) keeps exact per-conv telemetry as before.
    workers = min(cfg.write.conv_parallelism, max(1, len(todo)))
    parallel = workers > 1 and len(todo) > 1

    def _ingest_one(
        conv: Any, usage_acc: Any | None
    ) -> Mem0IngestReport:
        conv_dir = db_dir / conv.conv_id
        conv_dir.mkdir(parents=True)
        memory = memory_cls.from_config(build_mem0_config(cfg, conv_dir=conv_dir))
        usage_before = usage_acc.snapshot() if usage_acc is not None else {}
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
        usage = (
            usage_delta(usage_acc.snapshot(), usage_before)
            if usage_acc is not None
            else {}
        )
        cost = usage_cost(cfg, usage) if usage else None
        report = Mem0IngestReport(
            conv_id=conv.conv_id,
            sessions=len(conv.sessions),
            turns=turns,
            memories=memories,
            seconds=seconds,
            usage=usage,
            cost_usd=cost,
        )
        meta = {
            "conv_id": conv.conv_id,
            "sessions": report.sessions,
            "turns": report.turns,
            "memories": report.memories,
            "seconds": round(seconds, 3),
            "usage": usage,
            "usage_measured": bool(usage),
            "cost_usd": cost,
            "versions": _installed_versions(),
            "mem0_config": build_mem0_config(cfg, conv_dir=conv_dir),
        }
        sentinel = conv_dir / _SENTINEL
        tmp = sentinel.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        tmp.replace(sentinel)
        prompt, completion = total_tokens(usage)
        cost_str = f"${cost:.3f}" if cost is not None else "n/a"
        print(
            f"ingested {conv.conv_id} (mem0): {report.memories} memories from "
            f"{report.turns} turns / {report.sessions} sessions | "
            f"{seconds:.0f}s, tokens={prompt}+{completion}, cost~{cost_str}",
            flush=True,
        )
        return report

    reports_by_id: dict[str, Mem0IngestReport] = {}
    failures: list[str] = []
    with LiteLLMReasoningOff(slugs=slugs), LiteLLMUsageTap(slugs=slugs) as batch_usage:
        if parallel:
            from concurrent.futures import ThreadPoolExecutor

            def _run(conv: Any) -> None:
                try:
                    # Batch-level tap only: per-conv attribution is impossible
                    # with process-global callbacks and concurrent convs.
                    reports_by_id[conv.conv_id] = _ingest_one(conv, None)
                except Exception as exc:  # noqa: BLE001 - recorded, batch continues
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
                reports_by_id[conv.conv_id] = _ingest_one(conv, batch_usage)

    if failures:
        print(f"mem0 ingest failures: {sorted(failures)}", file=sys.stderr)
    reports = [reports_by_id[c.conv_id] for c in todo if c.conv_id in reports_by_id]
    total = sum(r.memories for r in reports)
    batch_snapshot = batch_usage.snapshot()
    prompt, completion = total_tokens(batch_snapshot)
    print(
        f"mem0 ingest total: {len(reports)} conversation(s), {total} memories "
        f"in {time.perf_counter() - batch_started:.0f}s, "
        f"batch tokens={prompt}+{completion}"
    )
    return reports


class Mem0Retriever:
    """``Retriever`` protocol impl over per-conversation mem0 stores."""

    name = "mem0"

    def __init__(self, cfg: AppConfig, *, db_dir: Path, answer_fn: AnswerFn) -> None:
        # Fail fast (before any paid phase) if the extra is missing.
        self._memory_cls = _import_memory_cls()
        self._cfg = cfg
        self._db_dir = Path(db_dir)
        self._answer_fn = answer_fn
        self._memories: dict[str, Any] = {}
        # Runner fans answer() out to threads; mem0's embedder/faiss
        # thread-safety is unverified — serialize everything but the LLM call.
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
            memory = self._memory_cls.from_config(
                build_mem0_config(self._cfg, conv_dir=conv_dir)
            )
            self._memories[conv_id] = memory
        return memory

    def answer(self, s: Sample) -> Prediction:
        start = time.perf_counter()
        with self._lock:
            memory = self._memory_for(s.conv_id)
            # threshold/rerank deliberately not passed: mem0 defaults
            # (fairness contract); recorded in manifest_extra.
            response = memory.search(
                s.question,
                filters={"user_id": s.conv_id},
                top_k=self._top_k,
            )
        results = (
            response.get("results", []) if isinstance(response, dict) else []
        )
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
        """Provenance for the run manifest (the fairness contract's record)."""
        ingest_meta: dict[str, Any] = {}
        if self._db_dir.is_dir():
            for sentinel in sorted(self._db_dir.glob(f"*/{_SENTINEL}")):
                meta = json.loads(sentinel.read_text(encoding="utf-8"))
                ingest_meta[meta.get("conv_id", sentinel.parent.name)] = meta
        return {
            "system": "mem0-oss",
            "versions": _installed_versions(),
            "mem0_config": build_mem0_config(
                self._cfg, conv_dir=self._db_dir / "<conv_id>"
            ),
            "llm_policy": (
                "hidden reasoning disabled via litellm extra_body during "
                "ingest — backbone parity with gluomem's "
                "dspy.reasoning_enabled=false"
            ),
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
    "LiteLLMReasoningOff",
    "LiteLLMUsageTap",
    "Mem0IngestReport",
    "Mem0Retriever",
    "build_mem0_config",
    "ingest_mem0_per_conversation",
]
