"""gluomem.bench.rescore — offline headline reproduction. NEVER imports dspy.

Re-derives ``scores.json``, ``denominators.json`` and ``report.md`` from a run directory's
committed ``manifest.json`` + ``predictions.jsonl`` + ``judgments.jsonl``
through the SAME ``compute_score_entries`` path the runner used — zero
network, zero model calls, no dataset file needed (prediction rows are
self-contained).

``check=True`` compares instead of rewriting and raises
``RescoreMismatchError`` on any drift — the tamper-evidence half of the
acceptance criterion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gluomem.bench.metrics import compute_denominators
from gluomem.bench.report import render_report
from gluomem.bench.runner import compute_score_entries
from gluomem.bench.types import (
    UNREGISTERED,
    JudgeRecord,
    Prediction,
    RetrievedDoc,
    Sample,
)


class RescoreMismatchError(RuntimeError):
    """Committed scores/report do not match what the judgments re-derive."""


def _sample_from_row(row: dict[str, Any]) -> Sample:
    return Sample(
        id=row["sample_id"],
        conv_id=row["conv_id"],
        question=row["question"],
        gold=row["gold"],
        category=row["category"],
        category_name=row["category_name"],
        evidence=tuple(row["evidence"]),
    )


def _prediction_from_row(row: dict[str, Any]) -> Prediction:
    return Prediction(
        sample_id=row["sample_id"],
        answer=row["answer"],
        retrieved=tuple(RetrievedDoc(**d) for d in row["retrieved"]),
        abstained=row["abstained"],
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        latency_ms=row["latency_ms"],
        abstention=row.get("abstention"),
        failure=row.get("failure"),
    )


def rescore_run(run_dir: str | Path, *, check: bool = False) -> list[dict[str, Any]]:
    """Recompute score entries; rewrite scores/report, or verify with check."""
    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    samples: dict[str, Sample] = {}
    predictions: dict[str, Prediction] = {}
    predictions_path = root / "predictions.jsonl"
    for line_no, line in enumerate(
        predictions_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            row = json.loads(line)
            samples[row["sample_id"]] = _sample_from_row(row)
            predictions[row["sample_id"]] = _prediction_from_row(row)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RescoreMismatchError(
                f"{predictions_path} line {line_no}: malformed prediction row "
                f"({exc!r}) — the committed artifact is corrupt or was "
                f"written by different code"
            ) from exc

    judgments_path = root / "judgments.jsonl"
    records: list[JudgeRecord] = []
    for line_no, line in enumerate(
        judgments_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            records.append(JudgeRecord(**json.loads(line)))
        except (json.JSONDecodeError, TypeError) as exc:
            raise RescoreMismatchError(
                f"{judgments_path} line {line_no}: malformed judgment row "
                f"({exc!r}) — the committed artifact is corrupt or was "
                f"written by different code"
            ) from exc

    backbones = list(manifest["judge"]["backbones"])
    # Pre-registry manifests carry neither key; their artifacts rescore as
    # unregistered, with retrieval presumed applicable except full context.
    protocol = manifest.get("protocol", UNREGISTERED)
    retrieves = manifest.get("retrieves", manifest["retriever"] != "full_context")
    entries = compute_score_entries(
        predictions,
        records,
        samples,
        backbones=backbones,
        judge_family=manifest["judge"]["family"],
        ks=list(manifest["retrieval_ks"]),
        dataset_sha=manifest["dataset_sha"],
        protocol=protocol,
        retrieves=retrieves,
    )
    denominators = compute_denominators(
        predictions,
        records,
        samples,
        backbones=backbones,
        retrieves=retrieves,
        excluded=manifest.get("excluded_from_pool"),
    )
    scores_text = json.dumps(entries, indent=2, sort_keys=True) + "\n"
    denominators_text = json.dumps(denominators, indent=2, sort_keys=True) + "\n"
    report_text = render_report(entries, manifest, denominators)

    scores_path = root / "scores.json"
    denominators_path = root / "denominators.json"
    report_path = root / "report.md"
    if check:
        committed_scores = scores_path.read_text(encoding="utf-8")
        if committed_scores != scores_text:
            raise RescoreMismatchError(
                f"{scores_path}: committed scores diverge from what the "
                f"committed judgments re-derive — artifacts were tampered "
                f"with or written by different code"
            )
        if not denominators_path.is_file() or (
            denominators_path.read_text(encoding="utf-8") != denominators_text
        ):
            raise RescoreMismatchError(
                f"{denominators_path}: committed denominators are missing or "
                f"diverge from what the committed predictions re-derive"
            )
        committed_report = report_path.read_text(encoding="utf-8")
        if committed_report != report_text:
            raise RescoreMismatchError(f"{report_path}: committed report diverges from re-render")
    else:
        scores_path.write_text(scores_text, encoding="utf-8")
        denominators_path.write_text(denominators_text, encoding="utf-8")
        report_path.write_text(report_text, encoding="utf-8")
    return entries


__all__ = ["RescoreMismatchError", "rescore_run"]
