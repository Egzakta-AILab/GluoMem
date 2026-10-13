"""gluomem.bench.registry — versioned benchmark registry (dspy-free).

One ``BenchmarkEntry`` per protocol the harness knows, runnable or not. An
entry records what a result means before any number exists: source ancestry,
native or adapted protocol, scorer families, logical clock, capability tags,
how splits are grouped, which model roles a run needs, rights, readiness and
budget. Runs copy the entry (and its digest) into their manifest, and every
score carries ``protocol_id(entry)`` in its ``ProtocolTag``, so a native and an
adapted protocol never aggregate into one number.

Overlap is tracked through ``source_ancestry``. Two entries that share an
ancestor (MemoryAgentBench reformulates LongMemEval; the LongMemEval oracle
and S variants ask the same questions) are not independent evidence and
cannot serve as each other's held-out set.

Entries may be ``blocked``: listed so their ancestry participates in overlap
checks, but refused by the runner until a revision and hash are pinned and
rights are verified.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Literal

REGISTRY_VERSION = 1

Mode = Literal["native", "adapted"]
Clock = Literal["session_dates", "event_order", "none"]
SplitUnit = Literal["question", "history", "subtask_family"]
Readiness = Literal["ready", "fixture", "blocked"]
RunStatus = Literal["unrun", "fixture_only", "historical_unrescored"]

WITHIN_HISTORY_LABEL = (
    "within-history question sample (NOT independent-history generalization)"
)
GROUPED_LABEL = "grouped by {unit}; independent only across source-disjoint entries"


class RegistryError(ValueError):
    """Unknown, blocked or mis-specified registry entry."""


class SourceOverlapError(RegistryError):
    """Two splits that must be independent share source ancestry."""


@dataclass(frozen=True)
class BenchmarkEntry:
    """One registered protocol. ``sha256`` is the pinned file hash when the
    registry owns the file (fixtures); config-pinned datasets name their
    ``benchmark.datasets`` key in ``dataset_key`` instead."""

    id: str
    version: int
    title: str
    revision: str
    source_ancestry: tuple[str, ...]
    mode: Mode
    scorers: tuple[str, ...]
    scorer_note: str
    logical_clock: Clock
    capabilities: tuple[str, ...]
    split_unit: SplitUnit
    model_roles: tuple[str, ...]
    rights: str
    readiness: Readiness
    readiness_note: str
    run_status: RunStatus
    official: bool
    budget_usd: float | None
    dataset_key: str | None = None
    sha256: str | None = None


def _entries() -> tuple[BenchmarkEntry, ...]:
    return (
        BenchmarkEntry(
            id="locomo",
            version=1,
            title="LoCoMo QA (categories 1-4)",
            revision="snap-research/locomo locomo10.json (sha pinned in config)",
            source_ancestry=("locomo",),
            mode="adapted",
            scorers=("qa_accuracy", "precision@k", "evidence_recall@k", "evidence_hit@k"),
            scorer_note=(
                "LLM-judge QA accuracy (trio-strict or single) instead of the "
                "paper's token F1; retrieval diagnostics against evidence dia_ids"
            ),
            logical_clock="session_dates",
            capabilities=("single_hop", "multi_hop", "temporal", "open_domain"),
            split_unit="question",
            model_roles=("reader", "judge"),
            rights="CC BY-NC 4.0 (dataset not redistributed)",
            readiness="ready",
            readiness_note="category 5 and rows without gold are excluded",
            run_status="historical_unrescored",
            official=False,
            budget_usd=None,
            dataset_key="locomo",
        ),
        BenchmarkEntry(
            id="longmemeval_oracle",
            version=1,
            title="LongMemEval oracle haystacks",
            revision="xiaowu0162/longmemeval-cleaned (sha pinned in config)",
            source_ancestry=("longmemeval",),
            mode="adapted",
            scorers=("qa_accuracy", "precision@k", "evidence_recall@k", "evidence_hit@k"),
            scorer_note="per-type judge rubrics adapted from the paper's GPT-4o prompts",
            logical_clock="session_dates",
            capabilities=(
                "information_extraction",
                "multi_session",
                "temporal",
                "knowledge_update",
                "abstention",
            ),
            split_unit="question",
            model_roles=("reader", "judge"),
            rights="unverified: confirm dataset terms before redistribution",
            readiness="ready",
            readiness_note="every row is eligible",
            run_status="historical_unrescored",
            official=False,
            budget_usd=None,
            dataset_key="longmemeval_oracle",
        ),
        BenchmarkEntry(
            id="longmemeval_s",
            version=1,
            title="LongMemEval S haystacks",
            revision="xiaowu0162/longmemeval-cleaned (sha pinned in config)",
            source_ancestry=("longmemeval",),
            mode="adapted",
            scorers=("qa_accuracy", "precision@k", "evidence_recall@k", "evidence_hit@k"),
            scorer_note="per-type judge rubrics adapted from the paper's GPT-4o prompts",
            logical_clock="session_dates",
            capabilities=(
                "information_extraction",
                "multi_session",
                "temporal",
                "knowledge_update",
                "abstention",
            ),
            split_unit="question",
            model_roles=("reader", "judge"),
            rights="unverified: confirm dataset terms before redistribution",
            readiness="ready",
            readiness_note="same questions as longmemeval_oracle with longer haystacks",
            run_status="historical_unrescored",
            official=False,
            budget_usd=None,
            dataset_key="longmemeval_s",
        ),
        BenchmarkEntry(
            id="mab_fixture",
            version=1,
            title="Generated MemoryAgentBench-shaped replay fixture",
            revision="gluomem/bench/fixtures/mab_tiny.json",
            source_ancestry=("gluomem-generated:mab_tiny",),
            mode="adapted",
            scorers=("evidence_hit@k", "context_substring_match@k", "current_over_stale@k"),
            scorer_note=(
                "retrieval-only: no reader model; substring match of accepted "
                "answers against retrieved context, not MAB's reader-output scorers"
            ),
            logical_clock="event_order",
            capabilities=("accurate_retrieval", "conflict_resolution"),
            split_unit="history",
            model_roles=("embedder",),
            rights="Apache-2.0, authored for GluoMem; contains no MemoryAgentBench text",
            readiness="fixture",
            readiness_note="contract fixture; not a benchmark result",
            run_status="fixture_only",
            official=False,
            budget_usd=0.0,
            sha256="1192a402561bcf606dca2b32d67c29f4fc641c1d6b7dc35e9ce81aaffea4c183",
        ),
        BenchmarkEntry(
            id="memoryagentbench",
            version=1,
            title="MemoryAgentBench (arXiv:2507.05257)",
            revision="unpinned",
            source_ancestry=("memoryagentbench", "longmemeval"),
            mode="native",
            scorers=("substring_exact_match", "recommendation_recall", "llm_judge"),
            scorer_note="per-subtask native scorers; never averaged across families",
            logical_clock="event_order",
            capabilities=(
                "accurate_retrieval",
                "test_time_learning",
                "long_range_understanding",
                "conflict_resolution",
            ),
            split_unit="subtask_family",
            model_roles=("reader", "judge"),
            rights="unverified",
            readiness="blocked",
            readiness_note=(
                "revision, hash and rights not pinned; ancestry lists only the "
                "LongMemEval-derived component confirmed by the paper"
            ),
            run_status="unrun",
            official=False,
            budget_usd=None,
        ),
        BenchmarkEntry(
            id="beam",
            version=1,
            title="BEAM long-history tiers",
            revision="unpinned",
            source_ancestry=("beam",),
            mode="native",
            scorers=("llm_judge",),
            scorer_note="native rubric scoring per tier",
            logical_clock="event_order",
            capabilities=("contradiction", "preference", "temporal", "scale"),
            split_unit="history",
            model_roles=("reader", "judge"),
            rights="unverified",
            readiness="blocked",
            readiness_note="small development slice not yet registered",
            run_status="unrun",
            official=False,
            budget_usd=None,
        ),
        BenchmarkEntry(
            id="memoryarena",
            version=1,
            title="MemoryArena interdependent subtasks",
            revision="unpinned (upstream marks itself preview)",
            source_ancestry=("memoryarena",),
            mode="native",
            scorers=("task_success",),
            scorer_note="trajectory outcomes; needs the trajectory runner seam",
            logical_clock="event_order",
            capabilities=("experience_reuse",),
            split_unit="history",
            model_roles=("agent", "environment"),
            rights="unverified",
            readiness="blocked",
            readiness_note="trajectory runner not implemented",
            run_status="unrun",
            official=False,
            budget_usd=None,
        ),
    )


REGISTRY: dict[str, BenchmarkEntry] = {e.id: e for e in _entries()}


def get_entry(benchmark_id: str) -> BenchmarkEntry:
    entry = REGISTRY.get(benchmark_id)
    if entry is None:
        raise RegistryError(
            f"benchmark '{benchmark_id}' is not registered; known: "
            f"{', '.join(sorted(REGISTRY))}"
        )
    return entry


def require_runnable(entry: BenchmarkEntry) -> None:
    if entry.readiness == "blocked":
        raise RegistryError(
            f"benchmark '{entry.id}' is blocked: {entry.readiness_note}"
        )


def protocol_id(entry: BenchmarkEntry) -> str:
    """Stamp carried by every ProtocolTag, e.g. ``adapted:locomo@v1``."""
    return f"{entry.mode}:{entry.id}@v{entry.version}"


def entry_digest(entry: BenchmarkEntry) -> str:
    material = json.dumps(asdict(entry), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def registry_digest() -> str:
    material = json.dumps(
        {"version": REGISTRY_VERSION, "entries": {k: entry_digest(v) for k, v in REGISTRY.items()}},
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def generalization_label(entry: BenchmarkEntry) -> str:
    """How a result from this entry may be described. Question-level samples
    drawn from shared histories are never independent-history evidence."""
    if entry.split_unit == "question":
        return WITHIN_HISTORY_LABEL
    return GROUPED_LABEL.format(unit=entry.split_unit)


def shared_ancestry(a: BenchmarkEntry, b: BenchmarkEntry) -> frozenset[str]:
    return frozenset(a.source_ancestry) & frozenset(b.source_ancestry)


def overlap_report(ids: Iterable[str]) -> list[dict[str, object]]:
    """Every pair among ``ids`` that shares a source ancestor."""
    entries = sorted({get_entry(i).id: get_entry(i) for i in ids}.values(), key=lambda e: e.id)
    out: list[dict[str, object]] = []
    for i, a in enumerate(entries):
        for b in entries[i + 1 :]:
            shared = shared_ancestry(a, b)
            if shared:
                out.append({"a": a.id, "b": b.id, "shared": sorted(shared)})
    return out


def assert_disjoint(development: Iterable[str], held_out: Iterable[str]) -> None:
    """Refuse a held-out set that shares source ancestry with development."""
    dev = [get_entry(i) for i in development]
    test = [get_entry(i) for i in held_out]
    clashes = [
        f"{d.id}~{t.id}: {sorted(shared)}"
        for d in dev
        for t in test
        if (shared := shared_ancestry(d, t))
    ]
    if clashes:
        raise SourceOverlapError(
            "held-out entries share source ancestry with development: " + "; ".join(clashes)
        )


def manifest_record(entry: BenchmarkEntry) -> dict[str, object]:
    """What a run manifest or receipt stores about its protocol."""
    return {
        **asdict(entry),
        "source_ancestry": list(entry.source_ancestry),
        "scorers": list(entry.scorers),
        "capabilities": list(entry.capabilities),
        "model_roles": list(entry.model_roles),
        "protocol": protocol_id(entry),
        "digest": entry_digest(entry),
        "registry_version": REGISTRY_VERSION,
        "generalization": generalization_label(entry),
    }


__all__ = [
    "REGISTRY",
    "REGISTRY_VERSION",
    "WITHIN_HISTORY_LABEL",
    "BenchmarkEntry",
    "RegistryError",
    "SourceOverlapError",
    "assert_disjoint",
    "entry_digest",
    "generalization_label",
    "get_entry",
    "manifest_record",
    "overlap_report",
    "protocol_id",
    "registry_digest",
    "require_runnable",
    "shared_ancestry",
]
