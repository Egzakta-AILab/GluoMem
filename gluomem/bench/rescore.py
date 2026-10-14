"""gluomem.bench.rescore — offline headline reproduction. NEVER imports dspy.

Re-derives ``scores.json`` and ``report.md`` from a run directory's
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

from gluomem.bench.report import render_report
from gluomem.bench.runner import compute_score_entries
from gluomem.bench.types import JudgeRecord, Prediction, RetrievedDoc, Sample


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

    entries = compute_score_entries(
        predictions,
        records,
        samples,
        backbones=list(manifest["judge"]["backbones"]),
        judge_family=manifest["judge"]["family"],
        ks=list(manifest["retrieval_ks"]),
        dataset_sha=manifest["dataset_sha"],
    )
    scores_text = json.dumps(entries, indent=2, sort_keys=True) + "\n"
    report_text = render_report(entries, manifest)

    scores_path = root / "scores.json"
    report_path = root / "report.md"
    if check:
        committed_scores = scores_path.read_text(encoding="utf-8")
        if committed_scores != scores_text:
            raise RescoreMismatchError(
                f"{scores_path}: committed scores diverge from what the "
                f"committed judgments re-derive — artifacts were tampered "
                f"with or written by different code"
            )
        committed_report = report_path.read_text(encoding="utf-8")
        if committed_report != report_text:
            raise RescoreMismatchError(
                f"{report_path}: committed report diverges from re-render"
            )
    else:
        scores_path.write_text(scores_text, encoding="utf-8")
        report_path.write_text(report_text, encoding="utf-8")
    return entries


__all__ = ["RescoreMismatchError", "rescore_run"]
