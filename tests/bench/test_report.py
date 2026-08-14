"""Report — one protocol group per table, citations as prose, CI gate."""

from __future__ import annotations

import re
from typing import Any

import pytest

from gluomem.bench.report import check_gate, render_report
from gluomem.config.app_config import load_config

SHA = "d" * 64


def _entry(
    scope: str,
    value: float,
    metric_family: str,
    judge_family: str = "none",
    backbone: str = "none",
    n: int = 4,
) -> dict[str, Any]:
    return {
        "scope": scope,
        "value": value,
        "n": n,
        "tag": {
            "metric_family": metric_family,
            "judge_family": judge_family,
            "judge_count": 3 if judge_family == "trio_strict" else 0,
            "backbone": backbone,
            "dataset_sha": SHA,
        },
    }


def _manifest() -> dict[str, Any]:
    return {
        "run_id": "20260705T000000_locomo_bm25",
        "dataset": "locomo",
        "dataset_sha": SHA,
        "retriever": "bm25",
        "answer": {"model": "deepseek-flash", "model_slug": "x/y", "backend": "fake"},
        "judge": {
            "family": "trio_strict",
            "backbones": ["gpt5-mini", "haiku", "qwen3"],
            "backend": "fake",
        },
        "sample_ids": ["a", "b", "c", "d"],
        "sample_source": "in-memory",
        "sample_seed": 42,
        "created_at": "2026-07-05T00:00:00+00:00",
        "git_sha": "abc",
    }


ENTRIES = [
    _entry(
        "overall", 0.75, "qa_accuracy", "trio_strict", "gpt5-mini+haiku+qwen3"
    ),
    _entry("category:temporal", 0.5, "qa_accuracy", "trio_strict", "gpt5-mini+haiku+qwen3"),
    _entry("overall", 1.0, "qa_accuracy", "single", "haiku"),
    _entry("overall", 0.4, "precision@5"),
    _entry("overall", 0.9, "evidence_hit@5"),
]


def test_one_protocol_group_per_table() -> None:
    report = render_report(ENTRIES, _manifest())
    # Split into sections; every table (## ... until next ##) mentions exactly
    # one metric family in its header and its rows carry no other family.
    sections = re.split(r"^## ", report, flags=re.MULTILINE)[1:]
    tables = [s for s in sections if "| scope |" in s]
    assert len(tables) == 4  # trio, single, precision@5, evidence_hit@5
    for section in tables:
        header = section.splitlines()[0]
        families = {
            f for f in ("qa_accuracy", "precision@5", "evidence_hit@5")
            if f in header
        }
        assert len(families) == 1, header


def test_qa_and_objective_never_share_a_table() -> None:
    report = render_report(ENTRIES, _manifest())
    for line in report.splitlines():
        if line.startswith("|") and "qa_accuracy" in line:
            raise AssertionError(
                "metric family leaked into a table row (families live in "
                "section headers only)"
            )


def test_citations_present_but_never_in_tables() -> None:
    report = render_report(ENTRIES, _manifest())
    assert "Mem0" in report and "hive-mind" in report
    citation_zone = report.split("Different-protocol citations")[1]
    assert "| " not in citation_zone  # prose, not rows


def test_check_gate_pass_and_fail() -> None:
    cfg = load_config()
    ok, entry = check_gate(ENTRIES, cfg.benchmark.ci_gate)
    assert ok is True and entry["value"] == 0.9  # evidence_hit@5 >= 0.7
    strict = cfg.with_overrides({"benchmark.ci_gate.min": 0.95})
    ok2, _ = check_gate(ENTRIES, strict.benchmark.ci_gate)
    assert ok2 is False


def test_check_gate_qa_accuracy_uses_trio() -> None:
    cfg = load_config().with_overrides({"benchmark.ci_gate.metric": "qa_accuracy"})
    ok, entry = check_gate(ENTRIES, cfg.benchmark.ci_gate)
    assert entry["tag"]["judge_family"] == "trio_strict"
    assert ok is True  # 0.75 >= 0.7


def test_check_gate_missing_metric_fails_loud() -> None:
    cfg = load_config().with_overrides(
        {"benchmark.ci_gate.metric": "evidence_recall@10"}
    )
    with pytest.raises(ValueError, match="never computed"):
        check_gate(ENTRIES, cfg.benchmark.ci_gate)
