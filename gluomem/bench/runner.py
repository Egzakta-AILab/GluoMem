"""gluomem.bench.runner — ``bench run`` orchestration.

The ONLY bench module that reaches dspy — and only through function-local
imports on the ``"dspy"`` backend path (the ``make_extractors`` pattern), so
``import gluomem.bench.runner`` stays dspy-free and the fake-backend CLI e2e
runs fully offline.

A run directory is self-contained for offline rescore: ``predictions.jsonl``
embeds every sample field the metrics need (category, evidence, retrieved
dia_ids) and ``judgments.jsonl`` holds the per-backbone JudgeRecords, so
``bench rescore`` needs neither the dataset file nor any model.
``denominators.json`` records every count behind the scores (failures,
abstentions, missing verdicts, retrieval not applicable, pool exclusions), and
the manifest carries the registry entry the run was scored under.

Artifact determinism: prediction and judgment rows are written in sample
order regardless of ``benchmark.parallelism`` (thunks return their rows;
ordering happens after ``run_ordered``), and ``scores.json`` entries are
sorted — a rerun over the same judgments is byte-identical.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from gluomem.bench.adapters import make_adapter
from gluomem.bench.adapters.locomo import (
    load_sample_manifest,
    stratified_sample,
)
from gluomem.bench.baselines import (
    Bm25Retriever,
    FullContextRetriever,
    VectorRagRetriever,
)
from gluomem.bench.judge import SingleJudge, fake_judge_fn
from gluomem.bench.metrics import (
    MetricRegistry,
    by_category,
    compute_denominators,
    qa_scores_from_records,
    retrieval_not_applicable,
    retrieval_scores,
)
from gluomem.bench.registry import (
    get_entry,
    manifest_record,
    protocol_id,
    require_runnable,
)
from gluomem.bench.report import render_report
from gluomem.bench.retriever import GluoMemRetriever
from gluomem.bench.types import (
    UNREGISTERED,
    JudgeRecord,
    Prediction,
    RunPaths,
    Sample,
    Score,
)
from gluomem.components import require_fake_allowed
from gluomem.llm.outputs import AnswerResult
from gluomem.write.lanes import run_ordered

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from gluomem.bench.types import AnswerFn, DatasetAdapter, JudgeFn, Retriever
    from gluomem.config.app_config import AppConfig
    from gluomem.embedding.base import Embedder
    from gluomem.embedding.reranker import Reranker
    from gluomem.llm.outputs import Extractors
    from gluomem.write.ingest import IngestReport

_log = logging.getLogger(__name__)

_FAKE_ANSWER_BANNER = (
    "!!! GLUOMEM BENCH IS USING THE FAKE ANSWERER (DETERMINISTIC, NO LLM) — "
    "PREDICTIONS ECHO THE RETRIEVED CONTEXT, NOT MODEL ANSWERS !!!"
)
_fake_answer_banner_emitted = False

RETRIEVER_NAMES = ("gluomem", "full_context", "vector_rag", "bm25", "mem0")


def _fake_answer_fn(question: str, context: str) -> AnswerResult:  # noqa: ARG001
    """Offline AnswerFn: the prediction IS the context, so the fake judge's
    substring match measures whether retrieval surfaced the gold text."""
    global _fake_answer_banner_emitted
    if not _fake_answer_banner_emitted:
        print(_FAKE_ANSWER_BANNER, file=sys.stderr, flush=True)
        _log.warning(_FAKE_ANSWER_BANNER)
        _fake_answer_banner_emitted = True
    return AnswerResult(text=context)


def _make_answer_fn(cfg: AppConfig) -> AnswerFn:
    if cfg.benchmark.answer.backend == "fake":
        require_fake_allowed(cfg, "benchmark.answer.backend='fake'")
        return _fake_answer_fn
    from gluomem.extras import require

    require("llm", "benchmark.answer.backend=dspy")
    from gluomem.llm.answer import Answerer
    from gluomem.llm.router import LMRouter

    # Live path: preflight the key so a missing/empty env var fails here with
    # a named error instead of a provider 401 after ingest already ran.
    router = LMRouter(cfg)
    router.assert_api_key(cfg.dspy.tasks[cfg.benchmark.answer.task].model)
    return Answerer(cfg, router=router)


def _make_judge_fns(
    cfg: AppConfig,
) -> tuple[dict[str, JudgeFn], Callable[[], dict[str, dict[str, int]]]]:
    """(backbone -> JudgeFn, harvest) — harvest returns merged judge token
    usage per model slug (empty on the fake path)."""
    backbones = cfg.benchmark.judge.backbones
    if cfg.benchmark.judge.backend == "fake":
        require_fake_allowed(cfg, "benchmark.judge.backend='fake'")
        return {b: fake_judge_fn for b in backbones}, dict
    from gluomem.extras import require

    require("llm", "benchmark.judge.backend=dspy")
    from gluomem.llm.judge import QAJudge
    from gluomem.llm.router import LMRouter
    from gluomem.llm.usage import UsageAccumulator

    router = LMRouter(cfg)
    for b in backbones:
        router.assert_api_key(b)
    judges = {b: QAJudge(router, b) for b in backbones}

    def harvest() -> dict[str, dict[str, int]]:
        merged = UsageAccumulator()
        for judge in judges.values():
            merged.add(judge.usage.snapshot())
        return merged.snapshot()

    return {b: j.judge for b, j in judges.items()}, harvest


def usage_cost(
    cfg: AppConfig, usage: dict[str, dict[str, int]]
) -> float | None:
    """USD estimate from registry per-1k rates; None when no slug has rates."""
    by_slug = {
        entry.provider_model: entry
        for entry in cfg.models.values()
        if entry.cost_per_1k_in is not None and entry.cost_per_1k_out is not None
    }
    cost, matched = 0.0, False
    for slug, stats in usage.items():
        entry = by_slug.get(slug)
        if entry is None:
            continue
        matched = True
        cost += stats.get("prompt_tokens", 0) / 1000.0 * (entry.cost_per_1k_in or 0.0)
        cost += stats.get("completion_tokens", 0) / 1000.0 * (
            entry.cost_per_1k_out or 0.0
        )
    return cost if matched else None


def _judge_prompt_sha(cfg: AppConfig, dataset: str = "locomo") -> str | None:
    """sha256 of the live judge instruction — prompt drift shows in artifacts.

    LoCoMo hashes the generic JudgeQA docstring. LongMemEval runs hash
    JudgeQA + JudgeQATyped + the sorted per-type rubric map, so a rubric
    edit is tamper-evident in the manifest. Only computed on the dspy path
    (importing signatures pulls dspy)."""
    if cfg.benchmark.judge.backend != "dspy":
        return None
    from gluomem.llm.signatures import JudgeQA

    material = JudgeQA.__doc__ or ""
    if dataset.startswith("longmemeval"):
        from gluomem.bench.adapters.longmemeval import JUDGE_RUBRICS
        from gluomem.llm.signatures import JudgeQATyped

        material += JudgeQATyped.__doc__ or ""
        material += json.dumps(JUDGE_RUBRICS, sort_keys=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _git_sha() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except OSError:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _load_samples(
    cfg: AppConfig, adapter: DatasetAdapter, limit: int | None
) -> tuple[list[Sample], str]:
    manifest_path = Path(cfg.benchmark.sample.manifest)
    if manifest_path.is_file():
        samples = load_sample_manifest(manifest_path, adapter=adapter)
        source = f"manifest:{manifest_path}"
    else:
        print(
            f"gluomem bench: no committed sample manifest at {manifest_path} — "
            f"sampling in-memory (ids recorded in the run manifest)",
            file=sys.stderr,
        )
        samples = stratified_sample(
            adapter.load(),
            per_category=cfg.benchmark.sample.per_category,
            categories=cfg.benchmark.sample.categories,
            seed=cfg.benchmark.sample.seed,
        )
        source = "in-memory"
    if limit is not None:
        samples = samples[:limit]
    return samples, source


def _make_retriever(
    name: str,
    cfg: AppConfig,
    adapter: DatasetAdapter,
    answer_fn: AnswerFn,
    *,
    db_dir: str | Path | None,
    embedder: Embedder | None,
    reranker: Reranker | None,
) -> Retriever:
    if name == "gluomem":
        if db_dir is None:
            raise ValueError(
                "retriever 'gluomem' needs --db-dir (per-conversation .mind "
                "files from `gluomem bench ingest`)"
            )
        return GluoMemRetriever(
            cfg,
            db_dir=Path(db_dir),
            answer_fn=answer_fn,
            content_to_dias=adapter.content_to_dias(),
            embedder=embedder,
            reranker=reranker,
        )
    if name == "mem0":
        if db_dir is None:
            raise ValueError(
                "retriever 'mem0' needs --db-dir (per-conversation mem0 "
                "stores from `gluomem bench ingest --system mem0`)"
            )
        from gluomem.bench.mem0 import Mem0Retriever

        return Mem0Retriever(cfg, db_dir=Path(db_dir), answer_fn=answer_fn)
    if name == "full_context":
        return FullContextRetriever(cfg, adapter.conversations(), answer_fn)
    if name == "bm25":
        return Bm25Retriever(cfg, adapter.conversations(), answer_fn)
    if name == "vector_rag":
        return VectorRagRetriever(
            cfg, adapter.conversations(), answer_fn, embedder=embedder
        )
    raise ValueError(
        f"unknown retriever '{name}'; known: {', '.join(RETRIEVER_NAMES)}"
    )


def _run_dir(
    cfg: AppConfig, out_root: str | Path | None, dataset: str, retriever: str
) -> Path:
    root = Path(out_root) if out_root is not None else Path(cfg.benchmark.run_dir)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    base = root / f"{stamp}_{dataset}_{retriever}"
    candidate, i = base, 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{i}")
        i += 1
    candidate.mkdir(parents=True)
    return candidate


def _score_entry(scope: str, score: Score) -> dict[str, Any]:
    return {
        "scope": scope,
        "value": score.value,
        "n": score.n,
        "tag": dataclasses.asdict(score.tag),
    }


def compute_score_entries(
    predictions: Mapping[str, Prediction],
    records: list[JudgeRecord],
    samples: Mapping[str, Sample],
    *,
    backbones: list[str],
    judge_family: str,
    ks: list[int],
    dataset_sha: str,
    protocol: str = UNREGISTERED,
    retrieves: bool = True,
) -> list[dict[str, Any]]:
    """All scores.json entries — the SINGLE scoring path shared by runner and
    rescore (identical inputs -> byte-identical output).

    Every prediction is in the QA denominator. Retrieval families cover every
    sample where retrieval applies (``retrieval_not_applicable``), so an
    answerable sample with nothing retrieved contributes zero.
    """
    registry = MetricRegistry()
    entries: list[dict[str, Any]] = []

    trio, singles = qa_scores_from_records(
        records,
        backbones=backbones,
        dataset_sha=dataset_sha,
        sample_ids=predictions.keys(),
        protocol=protocol,
    )
    if trio and judge_family == "trio_strict":
        entries.append(_score_entry("overall", registry.aggregate(trio.values())))
        for cat, score in by_category(trio, samples).items():
            entries.append(_score_entry(f"category:{cat}", score))
    for per_sample in singles.values():
        if per_sample:
            entries.append(
                _score_entry("overall", registry.aggregate(per_sample.values()))
            )

    # Retrieval diagnostics: group per-sample scores by metric family.
    families: dict[str, dict[str, Score]] = {}
    for sample_id, p in predictions.items():
        s = samples[sample_id]
        if retrieval_not_applicable(s, retrieves=retrieves) is not None:
            continue
        for score in retrieval_scores(
            p, s, ks=ks, dataset_sha=dataset_sha, protocol=protocol
        ):
            families.setdefault(score.tag.metric_family, {})[sample_id] = score
    for per_sample in families.values():
        entries.append(
            _score_entry("overall", registry.aggregate(per_sample.values()))
        )
        for cat, score in by_category(per_sample, samples).items():
            entries.append(_score_entry(f"category:{cat}", score))

    entries.sort(
        key=lambda e: (
            e["scope"],
            e["tag"]["metric_family"],
            e["tag"]["judge_family"],
            e["tag"]["backbone"],
        )
    )
    return entries


def _prediction_row(s: Sample, p: Prediction) -> dict[str, Any]:
    return {
        "sample_id": s.id,
        "conv_id": s.conv_id,
        "category": s.category,
        "category_name": s.category_name,
        "question": s.question,
        "gold": s.gold,
        "evidence": list(s.evidence),
        "answer": p.answer,
        "abstained": p.abstained,
        "retrieved": [dataclasses.asdict(d) for d in p.retrieved],
        "prompt_tokens": p.prompt_tokens,
        "completion_tokens": p.completion_tokens,
        "latency_ms": p.latency_ms,
        "abstention": p.abstention,
        "failure": p.failure,
    }


def run_benchmark(
    cfg: AppConfig,
    *,
    dataset: str,
    retriever: str,
    db_dir: str | Path | None = None,
    out_root: str | Path | None = None,
    limit: int | None = None,
    answer_fn: AnswerFn | None = None,
    judge_fns: Mapping[str, JudgeFn] | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> RunPaths:
    # Bench methodology: LoCoMo's usable categories carry no abstention gold, so
    # abstaining is always scored wrong — never beneficial — and no threshold
    # has been calibrated on frozen development data yet. Disable the gate here
    # (scoped to gluomem) so the manifest's recorded config matches what ran.
    # Questions with no eligible evidence still produce an abstained prediction
    # with ``abstention.reason == "no_hits"``.
    if retriever == "gluomem":
        cfg = cfg.with_overrides({"search.abstain.enabled": False})
    entry = get_entry(dataset)
    require_runnable(entry)
    protocol = protocol_id(entry)
    adapter = make_adapter(dataset, cfg)
    adapter.verify()
    dataset_sha = adapter.sha
    excluded_fn = getattr(adapter, "excluded_counts", None)
    excluded = cast("dict[str, int]", excluded_fn()) if callable(excluded_fn) else {}
    samples, sample_source = _load_samples(cfg, adapter, limit)

    answer_fn = answer_fn if answer_fn is not None else _make_answer_fn(cfg)
    if judge_fns is not None:
        fns: Mapping[str, JudgeFn] = judge_fns
        harvest_judge_usage: Callable[[], dict[str, dict[str, int]]] = dict
    else:
        fns, harvest_judge_usage = _make_judge_fns(cfg)
    impl = _make_retriever(
        retriever,
        cfg,
        adapter,
        answer_fn,
        db_dir=db_dir,
        embedder=embedder,
        reranker=reranker,
    )

    run_dir = _run_dir(cfg, out_root, dataset, retriever)
    paths = RunPaths(
        root=run_dir,
        manifest=run_dir / "manifest.json",
        predictions=run_dir / "predictions.jsonl",
        judgments=run_dir / "judgments.jsonl",
        scores=run_dir / "scores.json",
        report=run_dir / "report.md",
        denominators=run_dir / "denominators.json",
    )
    retrieves = bool(getattr(impl, "retrieves", True))

    # --- predict phase (parallel across samples, rows written in order) -----
    # A single sample's LLM/parse failure is DATA, not a run abort: the
    # prediction degrades to an empty answer (judged incorrect, denominator
    # unchanged — the same protocol rule as a missing judge verdict).
    answer_failures: list[str] = []  # list.append is atomic under the GIL

    def _predict(s: Sample) -> Prediction:
        try:
            return impl.answer(s)
        except Exception as exc:  # noqa: BLE001 - recorded, run continues
            answer_failures.append(s.id)
            print(f"answer failed for {s.id}: {exc}", file=sys.stderr)
            kind = "timeout" if isinstance(exc, TimeoutError) else "error"
            return Prediction(sample_id=s.id, answer="", failure=kind)

    parallelism = cfg.benchmark.parallelism
    predict_started = time.perf_counter()
    predict_thunks = [(s.id, lambda s=s: _predict(s)) for s in samples]
    predictions: dict[str, Prediction] = dict(
        run_ordered(predict_thunks, max_workers=parallelism)
    )
    predict_seconds = time.perf_counter() - predict_started
    samples_by_id = {s.id: s for s in samples}
    with paths.predictions.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(
                json.dumps(_prediction_row(s, predictions[s.id]), sort_keys=True)
                + "\n"
            )

    # --- judge phase (parallel across samples, sequential within a sample) --
    backbones = sorted(fns)
    slugs = {
        b: (cfg.models[b].provider_model if b in cfg.models else b)
        for b in backbones
    }

    judge_family = cfg.benchmark.judge.family
    judge_failures: list[str] = []

    # LongMemEval grades per question type (temporal off-by-one, knowledge-
    # update accepts old+new, abstention correct iff declined) — select the
    # rubric per sample; LoCoMo keeps the generic JudgeQA path (rubric None).
    rubric_for: Callable[[Sample], str | None] | None = None
    if dataset.startswith("longmemeval"):
        from gluomem.bench.adapters.longmemeval import JUDGE_RUBRICS

        def rubric_for(s: Sample) -> str | None:
            return JUDGE_RUBRICS.get(s.category_name)

    def _judge_sample(s: Sample) -> list[JudgeRecord]:
        # Every backbone records independently; the trio decision is derived
        # from the records in metrics (qa_scores_from_records), so a failed
        # verdict simply leaves no record — scored not-correct, denominator
        # unchanged (the documented parse-failure protocol).
        rows: list[JudgeRecord] = []
        for b in backbones:
            member = SingleJudge(
                b,
                fns[b],
                model_slug=slugs[b],
                dataset_sha=dataset_sha,
                sink=rows.append,
                rubric_for=rubric_for,
            )
            try:
                member.score(s, predictions[s.id])
            except Exception as exc:  # noqa: BLE001 - recorded, run continues
                judge_failures.append(f"{s.id}:{b}")
                print(f"judge {b} failed for {s.id}: {exc}", file=sys.stderr)
        return rows

    judge_started = time.perf_counter()
    judge_thunks = [(s.id, lambda s=s: _judge_sample(s)) for s in samples]
    records: list[JudgeRecord] = []
    for _sid, rows in run_ordered(judge_thunks, max_workers=parallelism):
        records.extend(rows)
    judge_seconds = time.perf_counter() - judge_started
    with paths.judgments.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(dataclasses.asdict(rec), sort_keys=True) + "\n")

    # --- scores ---------------------------------------------------------------
    entries = compute_score_entries(
        predictions,
        records,
        samples_by_id,
        backbones=backbones,
        judge_family=judge_family,
        ks=list(cfg.benchmark.retrieval.ks),
        dataset_sha=dataset_sha,
        protocol=protocol,
        retrieves=retrieves,
    )
    paths.scores.write_text(
        json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    denominators = compute_denominators(
        predictions,
        records,
        samples_by_id,
        backbones=backbones,
        retrieves=retrieves,
        excluded=excluded,
    )
    paths.denominators.write_text(
        json.dumps(denominators, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # --- manifest ---------------------------------------------------------------
    answer_task = cfg.benchmark.answer.task
    answer_model = cfg.dspy.tasks[answer_task].model
    answer_entry = cfg.models[answer_model]
    prompt_tokens = sum(p.prompt_tokens or 0 for p in predictions.values())
    completion_tokens = sum(p.completion_tokens or 0 for p in predictions.values())
    cost_usd: float | None = None
    if answer_entry.cost_per_1k_in is not None and answer_entry.cost_per_1k_out is not None:
        cost_usd = (
            prompt_tokens / 1000.0 * answer_entry.cost_per_1k_in
            + completion_tokens / 1000.0 * answer_entry.cost_per_1k_out
        )
    # Competitor retrievers (mem0, ...) attach their own provenance
    # (versions, effective config, per-conv ingest telemetry).
    extra_fn = getattr(impl, "manifest_extra", None)
    retriever_info = extra_fn() if callable(extra_fn) else None
    judge_usage = harvest_judge_usage()
    judge_cost = usage_cost(cfg, judge_usage)
    total_cost: float | None = None
    if cost_usd is not None or judge_cost is not None:
        total_cost = (cost_usd or 0.0) + (judge_cost or 0.0)
    manifest = {
        "run_id": run_dir.name,
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "dataset": dataset,
        "dataset_sha": dataset_sha,
        "benchmark": manifest_record(entry),
        "protocol": protocol,
        "excluded_from_pool": excluded,
        "retrieves": retrieves,
        "sample_source": sample_source,
        "sample_ids": [s.id for s in samples],
        "sample_seed": cfg.benchmark.sample.seed,
        "limit": limit,
        "retriever": retriever,
        "answer": {
            "task": answer_task,
            "model": answer_model,
            "model_slug": answer_entry.provider_model,
            "backend": cfg.benchmark.answer.backend,
            "style": cfg.benchmark.answer.style,
        },
        "judge": {
            "family": cfg.benchmark.judge.family,
            "backbones": backbones,
            "model_slugs": slugs,
            "backend": cfg.benchmark.judge.backend,
            "prompt_sha": _judge_prompt_sha(cfg, dataset),
        },
        "retrieval_ks": list(cfg.benchmark.retrieval.ks),
        "retriever_info": retriever_info,
        "timing": {
            "predict_seconds": round(predict_seconds, 3),
            "judge_seconds": round(judge_seconds, 3),
            "parallelism": parallelism,
        },
        "usage": {
            "answer_prompt_tokens": prompt_tokens,
            "answer_completion_tokens": completion_tokens,
            "answer_cost_usd_estimate": cost_usd,
            "judge_calls": len(records),
            "judge_usage": judge_usage,
            "judge_cost_usd_estimate": judge_cost,
            "total_cost_usd_estimate": total_cost,
        },
        "failures": {
            "answer": sorted(answer_failures),
            "judge": sorted(judge_failures),
        },
        "config": cfg.model_dump(mode="json"),
    }
    paths.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths.report.write_text(
        render_report(entries, manifest, denominators), encoding="utf-8"
    )

    headline = _headline(entries)
    if headline is not None:
        cost_str = f"${total_cost:.2f}" if total_cost is not None else "n/a"
        print(
            f"[{dataset} sha={dataset_sha[:12]}] {retriever}: "
            f"qa_accuracy={headline['value']:.4f} n={headline['n']} "
            f"judge={headline['tag']['judge_family']}"
            f"({headline['tag']['backbone']}) "
            f"predict={predict_seconds:.0f}s judge={judge_seconds:.0f}s "
            f"cost~{cost_str}"
        )
    return paths


def _headline(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    for e in entries:
        if (
            e["scope"] == "overall"
            and e["tag"]["metric_family"] == "qa_accuracy"
            and e["tag"]["judge_family"] == "trio_strict"
        ):
            return e
    return None


# --- per-conversation ingest -----------------------------------------------------


def _ingest_sentinel(db_dir: Path, conv_id: str) -> Path:
    """Completion sentinel beside the ``.mind`` (mirrors mem0's ingest_meta.json).

    Written ATOMICALLY only after the conversation's last session is admitted;
    a ``.mind`` without it is a partial ingest and must never be silently
    scored — the file's existence alone proves nothing about completeness.
    """
    return db_dir / f"{conv_id}.ingest.json"


def ingest_per_conversation(
    cfg: AppConfig,
    *,
    dataset: str,
    db_dir: Path,
    facade: Extractors,
    embedder: Embedder | None = None,
    skip_existing: bool = False,
    conv_ids: set[str] | None = None,
) -> list[IngestReport]:
    """One ``.mind`` per conversation — the layout GluoMemRetriever expects.

    Resume semantics (fail-loud): a conversation is "done" only when its
    completion sentinel (``<conv_id>.ingest.json``) exists — the sentinel is
    written atomically AFTER the last session lands. ``skip_existing`` skips
    sentinel-complete conversations; a ``.mind`` WITHOUT a sentinel is a
    partial (or pre-sentinel) ingest and raises either way — delete it to
    re-ingest, or bless a verified-complete legacy dir via
    ``gluomem bench verify --write-sentinels``. Prints per-conversation wall
    time, token usage and cost (when the facade tracks usage) plus a total.
    """
    from gluomem.embedding.providers import make_embedder
    from gluomem.frames.store import FrameStore
    from gluomem.kg.graph import KnowledgeGraph
    from gluomem.llm.usage import total_tokens, usage_delta
    from gluomem.storage.db import MindDB
    from gluomem.write.gate import InjectionScanner
    from gluomem.write.ingest import ingest_conversation

    adapter = make_adapter(dataset, cfg)
    adapter.verify()
    db_dir.mkdir(parents=True, exist_ok=True)
    scanner = InjectionScanner(cfg.safety)
    # One embedder for the whole batch (loading MiniLM per conversation
    # wastes ~20s and churns memory).
    if embedder is None:
        embedder = make_embedder(cfg)

    def snapshot() -> dict[str, dict[str, int]]:
        acc = getattr(facade, "usage", None)
        return acc.snapshot() if acc is not None else {}

    # Conversations are independent .mind DBs, so they ingest concurrently.
    # conv_parallelism bounds how many at once; the process-wide LM gate
    # (DspyExtractors) is the real API throttle. reports come back in input
    # order (thread pool completes out of order); a single conversation's crash
    # is recorded and the batch continues (mirrors run_benchmark's tolerance).
    convs = list(adapter.conversations())
    if conv_ids is not None:
        convs = [c for c in convs if c.conv_id in conv_ids]
        missing = set(conv_ids) - {c.conv_id for c in convs}
        if missing:
            raise ValueError(
                f"conv_ids not present in dataset: {sorted(missing)[:5]}..."
                if len(missing) > 5
                else f"conv_ids not present in dataset: {sorted(missing)}"
            )
    # Sentinel screening happens up front (before any spend); the pool below
    # only ever sees conversations that genuinely need ingesting.
    todo: list[tuple[int, Any]] = []
    for i, conv in enumerate(convs):
        db_path = db_dir / f"{conv.conv_id}.mind"
        sentinel = _ingest_sentinel(db_dir, conv.conv_id)
        if db_path.is_file():
            if not sentinel.is_file():
                raise RuntimeError(
                    f"{db_path} exists without {sentinel.name} — partial or "
                    f"pre-sentinel ingest; delete the .mind to re-ingest, or "
                    f"bless a complete legacy dir with `gluomem bench verify "
                    f"--dataset {dataset} --db-dir {db_dir} --write-sentinels`"
                )
            if skip_existing:
                print(f"skipped {conv.conv_id}: {sentinel.name} exists")
                continue
            raise RuntimeError(
                f"{db_path} already ingested; pass --skip-existing to resume "
                f"around it or delete it (and {sentinel.name}) to redo"
            )
        sentinel.unlink(missing_ok=True)  # stale sentinel without a .mind
        todo.append((i, conv))

    workers = min(cfg.write.conv_parallelism, max(1, len(todo)))
    parallel = workers > 1 and len(todo) > 1
    reports_by_index: dict[int, IngestReport] = {}
    failures: list[str] = []
    batch_started = time.perf_counter()
    usage_start = snapshot()

    def _ingest_one(index: int, conv: Any) -> None:
        db_path = db_dir / f"{conv.conv_id}.mind"
        conv_started = time.perf_counter()
        # Per-conv token/cost attribution needs an isolated usage window; under
        # concurrency the shared accumulator sees other convs' tokens, so we only
        # snapshot per-conv in sequential mode and report tokens/cost/time at the
        # (exact) batch total otherwise.
        usage_before = None if parallel else snapshot()
        db = MindDB(db_path, dim=cfg.storage.vector_dim)
        try:
            store = FrameStore(db, embedder=embedder, chunking=cfg.chunking)
            kg = KnowledgeGraph(db)
            report = ingest_conversation(
                conv, facade=facade, scanner=scanner, store=store, kg=kg, cfg=cfg
            )
        except Exception as exc:  # noqa: BLE001 - recorded, batch continues
            failures.append(conv.conv_id)
            print(f"ingest failed for {conv.conv_id}: {exc}", file=sys.stderr)
            return
        finally:
            db.close()
        # Completion sentinel — atomic write AFTER the whole conversation
        # landed, so an interruption anywhere above leaves no sentinel and the
        # resume path refuses the partial .mind instead of skipping it.
        meta = {
            "conv_id": conv.conv_id,
            "sessions": len(conv.sessions),
            "i_frames": report.i_frames,
            "p_frames": report.p_frames,
            "b_frames": report.b_frames,
            "deduped": report.deduped,
            "semantic_deduped": report.semantic_deduped,
            "quarantined": report.quarantined,
            "rejected": report.rejected,
            "entities": report.entities,
            "relations": report.relations,
            "retired_edges": report.retired_edges,
            "demoted": report.demoted,
            "extraction_errors": len(report.extraction_errors),
            "seconds": round(time.perf_counter() - conv_started, 3),
        }
        sentinel = _ingest_sentinel(db_dir, conv.conv_id)
        tmp = sentinel.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        tmp.replace(sentinel)
        reports_by_index[index] = report  # distinct keys → GIL-atomic, no lock
        counts = (
            f"ingested {conv.conv_id}: {report.i_frames} I / "
            f"{report.p_frames} P / {report.b_frames} B, "
            f"deduped={report.deduped}, sem_deduped={report.semantic_deduped}, "
            f"quarantined={report.quarantined}"
        )
        if usage_before is None:
            print(f"{counts} | (tokens/cost: batch total)")
        else:
            conv_usage = usage_delta(snapshot(), usage_before)
            prompt, completion = total_tokens(conv_usage)
            cost = usage_cost(cfg, conv_usage)
            cost_str = f"${cost:.3f}" if cost is not None else "n/a"
            print(
                f"{counts} | {time.perf_counter() - conv_started:.0f}s, "
                f"tokens={prompt}+{completion}, cost~{cost_str}"
            )

    if parallel:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for fut in [pool.submit(_ingest_one, i, conv) for i, conv in todo]:
                fut.result()  # ingest errors already caught; surface unexpected ones
    else:
        for i, conv in todo:
            _ingest_one(i, conv)

    reports = [reports_by_index[i] for i in sorted(reports_by_index)]
    if failures:
        print(f"ingest failures: {sorted(failures)}", file=sys.stderr)
    batch_usage = usage_delta(snapshot(), usage_start)
    prompt, completion = total_tokens(batch_usage)
    cost = usage_cost(cfg, batch_usage)
    cost_str = f"${cost:.3f}" if cost is not None else "n/a"
    print(
        f"ingest total: {len(reports)} conversation(s) in "
        f"{time.perf_counter() - batch_started:.0f}s, "
        f"tokens={prompt}+{completion}, cost~{cost_str}"
    )
    # Frozen-extraction provenance: when the facade is a CachedExtractors, print
    # the cache sha (portable, order-independent) + live-miss count so an ablation
    # arm can prove it replayed the recorded extraction (record: all live; replay/
    # readthrough hits: misses≈0). Written to a sidecar for durable provenance.
    if getattr(cfg.write.extraction_cache, "mode", "off") != "off" and hasattr(
        facade, "cache_sha"
    ):
        sha = facade.cache_sha()
        misses = getattr(facade, "cache_misses", 0)
        print(
            f"extraction cache [{cfg.write.extraction_cache.mode}]: "
            f"sha={sha[:12]} live_misses={misses}"
        )
        try:
            (db_dir / "extraction_cache.sha256").write_text(sha + "\n")
        except OSError:
            pass
    return reports


def verify_ingest_completeness(
    cfg: AppConfig,
    *,
    dataset: str,
    db_dir: Path,
    write_sentinels: bool = False,
) -> list[dict[str, Any]]:
    """Audit a per-conversation ``.mind`` dir for silently-partial ingests.

    For every dataset conversation whose ``.mind`` exists, compares the
    distinct raw-turn session GOPs (``{conv_id}:session:N``) against the true
    session count. Relies on the raw-turns ingest layout
    (``write.raw_turns.enabled``, the default): a conversation with fewer
    session GOPs than sessions reads as ``incomplete``, including the
    raw-turns-disabled case — fail loud, never silently pass. Conversations
    without a ``.mind`` are reported ``absent`` and do NOT fail the audit
    (subset ingests are legitimate). ``write_sentinels`` blesses complete
    conversations from a pre-sentinel ingest with the completion sentinel so
    ``--skip-existing`` accepts them.
    """
    import sqlite3

    adapter = make_adapter(dataset, cfg)
    adapter.verify()
    rows: list[dict[str, Any]] = []
    for conv in adapter.conversations():
        db_path = db_dir / f"{conv.conv_id}.mind"
        expected = len(conv.sessions)
        if not db_path.is_file():
            rows.append(
                {
                    "conv_id": conv.conv_id,
                    "expected_sessions": expected,
                    "found_sessions": None,
                    "status": "absent",
                }
            )
            continue
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            found = int(
                con.execute(
                    "SELECT COUNT(DISTINCT gop_id) FROM memory_frames "
                    "WHERE gop_id LIKE ?",
                    (f"{conv.conv_id}:session:%",),
                ).fetchone()[0]
            )
        finally:
            con.close()
        complete = found == expected
        rows.append(
            {
                "conv_id": conv.conv_id,
                "expected_sessions": expected,
                "found_sessions": found,
                "status": "complete" if complete else "incomplete",
            }
        )
        if complete and write_sentinels:
            sentinel = _ingest_sentinel(db_dir, conv.conv_id)
            if not sentinel.is_file():
                tmp = sentinel.with_suffix(".tmp")
                tmp.write_text(
                    json.dumps(
                        {
                            "conv_id": conv.conv_id,
                            "sessions": expected,
                            "verified_by": "bench verify --write-sentinels",
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                tmp.replace(sentinel)
    return rows


__all__ = [
    "RETRIEVER_NAMES",
    "compute_score_entries",
    "ingest_per_conversation",
    "verify_ingest_completeness",
    "run_benchmark",
]
