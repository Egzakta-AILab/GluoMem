"""gluomem.bench.report — family-separated report rendering + the CI gate.

"Reports never co-locate F1 with LLM-judge accuracy" is enforced in code:
every table is built from exactly one (metric_family, judge_family,
backbone) group and an internal check raises if a mixed group ever reaches
a table. Literature numbers (Mem0, the sibling hive-mind harness) appear
only in a prose citation block with their protocols spelled out — never as
table rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gluomem.config.app_config import BenchmarkCiGateConfig

# Different-protocol anchors: cited, tagged, never co-tabulated.
_CITATIONS = """\
## Different-protocol citations (NOT comparable rows)

Numbers below were produced under other harnesses/protocols; they are
context, not columns:

- Mem0 paper (arXiv:2504.19413): LoCoMo J=66.88 (Mem0) / 68.44 (Mem0-graph)
  — single GPT-4o-mini judge, ~1,540 questions, generous-grading rubric.
- hive-mind (sibling TS harness): LoCoMo trio-strict 67.8% / self-judge
  73.1% (+5.3pp inflation) — N=320 stratified seed 42, Opus 4.7 subject.
- LoCoMo full-context reference (literature): ~72.9% — beats memory systems
  at this scale; the RQ1 hypothesis is that this flips as history grows.
"""


def _group_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    tag = entry["tag"]
    return (tag["metric_family"], tag["judge_family"], tag["backbone"])


def render_report(
    entries: Sequence[dict[str, Any]], manifest: dict[str, Any]
) -> str:
    """Markdown report: one table per protocol group, citations as prose."""
    lines: list[str] = [
        f"# bench report — {manifest['run_id']}",
        "",
        f"- dataset: {manifest['dataset']} (sha `{manifest['dataset_sha']}`)",
        f"- retriever: **{manifest['retriever']}**",
        (
            f"- answer: {manifest['answer']['model']} "
            f"(`{manifest['answer']['model_slug']}`, "
            f"backend={manifest['answer']['backend']})"
        ),
        (
            f"- judges: {', '.join(manifest['judge']['backbones'])} "
            f"(family={manifest['judge']['family']}, "
            f"backend={manifest['judge']['backend']})"
        ),
        f"- samples: {len(manifest['sample_ids'])} "
        f"(source={manifest['sample_source']}, seed={manifest['sample_seed']})",
        f"- created: {manifest['created_at']} @ git {manifest['git_sha']}",
        "",
    ]

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        groups.setdefault(_group_key(entry), []).append(entry)

    for (metric_family, judge_family, backbone), group in sorted(groups.items()):
        # The acceptance invariant as code: a table renders exactly one
        # protocol group; anything else is a bug, not a formatting choice.
        mixed = {_group_key(e) for e in group}
        if mixed != {(metric_family, judge_family, backbone)}:
            raise AssertionError(f"mixed protocol groups in one table: {mixed}")
        judge_desc = (
            "objective (no LLM judge)"
            if judge_family == "none"
            else f"{judge_family} ({backbone})"
        )
        lines.append(f"## {metric_family} — {judge_desc}")
        lines.append("")
        lines.append("| scope | value | n |")
        lines.append("|---|---|---|")
        for entry in sorted(group, key=lambda e: e["scope"]):
            lines.append(
                f"| {entry['scope']} | {entry['value']:.4f} | {entry['n']} |"
            )
        lines.append("")

    lines.append(_CITATIONS)
    return "\n".join(lines)


def check_gate(
    entries: Sequence[dict[str, Any]], gate: BenchmarkCiGateConfig
) -> tuple[bool, dict[str, Any]]:
    """Locate the gate metric's overall score; missing metric = config error,
    never a silent pass."""
    # qa_accuracy gates on the trio-strict headline; @k metrics are objective.
    wanted_judge = "trio_strict" if gate.metric == "qa_accuracy" else "none"
    for entry in entries:
        if (
            entry["scope"] == "overall"
            and entry["tag"]["metric_family"] == gate.metric
            and entry["tag"]["judge_family"] == wanted_judge
        ):
            return entry["value"] >= gate.min, entry
    raise ValueError(
        f"ci_gate metric '{gate.metric}' (judge_family={wanted_judge}) not "
        f"present in scores — the run never computed it"
    )


__all__ = ["check_gate", "render_report"]
