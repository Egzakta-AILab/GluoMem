"""Deterministic offline retrieval and relevance signal ablation runner (`DESIGN.md` §5, §9).

Evaluates signal-zeroing ablations (`no_temporal`, `no_popularity`,
`no_contextual`, `no_importance`) and lane ablations (`fts_only`, `vec_only`,
`hybrid_rrf`) over a :class:`CandidateCache` with zero model calls.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gluomem.tune.candidate_cache import CandidateCache
from gluomem.tune.rescore import ParamVector, evaluate_cache

if TYPE_CHECKING:
    from gluomem.config.app_config import AppConfig

__all__ = [
    "AblationReport",
    "AblationRow",
    "AblationVariant",
    "default_ablation_variants",
    "format_ablation_table",
    "run_ablation",
]


@dataclass(frozen=True)
class AblationVariant:
    """One named parameter configuration to evaluate in an ablation sweep."""

    name: str
    description: str
    params: ParamVector


@dataclass(frozen=True)
class AblationRow:
    """Metrics for one :class:`AblationVariant`."""

    name: str
    description: str
    ndcg: float
    recall: float
    hit: float
    mrr: float
    delta_ndcg: float
    delta_recall: float
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AblationReport:
    """Full ablation sweep report across a :class:`CandidateCache`."""

    profile: str
    k: int
    query_count: int
    cache_manifest_digest: str
    incumbent_ndcg: float
    incumbent_recall: float
    rows: list[AblationRow]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "k": self.k,
            "query_count": self.query_count,
            "cache_manifest_digest": self.cache_manifest_digest,
            "incumbent_ndcg": self.incumbent_ndcg,
            "incumbent_recall": self.incumbent_recall,
            "rows": [r.to_dict() for r in self.rows],
        }


def default_ablation_variants(base: ParamVector) -> list[AblationVariant]:
    """Build the standard signal and lane ablation suite from ``base``."""
    variants: tuple[tuple[str, str, dict[str, Any]], ...] = (
        ("incumbent", "Full hybrid RRF × relevance", {}),
        ("no_temporal", "Zero temporal decay", {"w_t": 0.0}),
        ("no_popularity", "Zero popularity boost", {"w_p": 0.0}),
        ("no_contextual", "Zero KG contextual boost", {"w_c": 0.0}),
        ("no_importance", "Zero importance multiplier", {"w_i": 0.0}),
        ("hybrid_rrf", "Pure RRF fusion", {"use_relevance": False}),
        ("fts_only", "FTS lane only", {"use_vec": False}),
        ("vec_only", "Vector lane only", {"use_fts": False}),
    )
    return [
        AblationVariant(name, description, replace(base, **changes))
        for name, description, changes in variants
    ]


def run_ablation(
    cache: CandidateCache,
    cfg: AppConfig,
    *,
    profile: str | None = None,
    k: int = 10,
    variants: Sequence[AblationVariant] | None = None,
    out_dir: Path | str | None = None,
) -> AblationReport:
    """Evaluate all ablation variants on `cache` and optionally write `ablation_report.json`."""
    prof_name = profile or cfg.search.default_profile
    incumbent = ParamVector.from_config(cfg, profile=prof_name)
    incumbent_metrics = evaluate_cache(cache, incumbent, k=k)
    base_ndcg = float(incumbent_metrics["ndcg"])
    base_recall = float(incumbent_metrics["recall"])

    resolved_variants = (
        list(variants) if variants is not None else default_ablation_variants(incumbent)
    )
    rows: list[AblationRow] = []
    for variant in resolved_variants:
        m = evaluate_cache(cache, variant.params, k=k)
        rows.append(
            AblationRow(
                name=variant.name,
                description=variant.description,
                ndcg=float(m["ndcg"]),
                recall=float(m["recall"]),
                hit=float(m["hit"]),
                mrr=float(m["mrr"]),
                delta_ndcg=float(m["ndcg"]) - base_ndcg,
                delta_recall=float(m["recall"]) - base_recall,
                params=variant.params.to_dict(),
            )
        )

    qids = [qid for qid in cache.query_ids() if cache.qrel(qid)]
    report = AblationReport(
        profile=prof_name,
        k=int(k),
        query_count=len(qids),
        cache_manifest_digest=cache.manifest.digest(),
        incumbent_ndcg=base_ndcg,
        incumbent_recall=base_recall,
        rows=rows,
    )
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "ablation_report.json").write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def format_ablation_table(report: AblationReport) -> str:
    """Render a compact human-readable table for CLI output."""
    header = (
        f"Ablation Report (profile={report.profile}, k={report.k}, "
        f"queries={report.query_count})\n"
        f"{'Variant':<16} {'nDCG@k':>8} {'ΔnDCG':>8} {'Recall@k':>9} "
        f"{'Hit@k':>7} {'MRR@k':>7}  Description"
    )
    sep = "-" * len(header.splitlines()[-1])
    lines = [header, sep]
    for r in report.rows:
        lines.append(
            f"{r.name:<16} {r.ndcg:8.4f} {r.delta_ndcg:+8.4f} {r.recall:9.4f} "
            f"{r.hit:7.4f} {r.mrr:7.4f}  {r.description}"
        )
    return "\n".join(lines)
