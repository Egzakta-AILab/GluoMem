"""gluomem.llm.optimize — OFFLINE prompt optimization → pinned artifacts.

``make optimize-prompts`` (→ ``python -m gluomem.llm.optimize``) compiles the
task modules with ``MIPROv2(auto=cfg.dspy.optimizer.auto)`` against a JSONL
trainset and persists the result through ``bootstrap.save_artifact`` (sha256
manifest). Runtime NEVER compiles — it only loads these artifacts read-only
(``bootstrap.load_or_build``).

This entrypoint spends real OpenRouter tokens: it fail-louds without
``OPENROUTER_API_KEY``. Tests inject a fake ``teleprompter_factory`` and
never touch MIPROv2.

Trainset format: one JSON object per line whose keys are the task
signature's INPUT fields plus expected OUTPUT fields, e.g. for
``cognify``::

    {"text": "...", "entities": [...], "relations": [...]}

Fact and event extraction are scored by the frozen source-backed evaluator
(``gluomem.llm.extraction_eval``), so their rows must also carry the
``case_id`` of an annotated, non-sealed fixture case; ``fixture_examples``
builds them directly. Unannotated extraction rows are refused, because
nothing could check their facts against the source.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import dspy

from gluomem.config.app_config import AppConfig, load_config
from gluomem.llm.bootstrap import save_artifact
from gluomem.llm.extraction_eval import (
    SEALED_SPLIT,
    Fixture,
    SealedSplitError,
    load_fixture,
    score_case,
)
from gluomem.llm.modules import build_module
from gluomem.llm.router import LMRouter
from gluomem.llm.signatures import TASK_SIGNATURES

# Anything with .compile(student, trainset=...) — tests inject fakes.
TeleprompterFactory = Callable[[Callable], Any]

# Task -> the output field the source-backed evaluator scores.
EXTRACTION_TASKS = {"fact_extraction": "facts", "event_extraction": "events"}


def load_trainset(path: str | Path, task: str) -> list[dspy.Example]:
    """Load a JSONL trainset, marking the task signature's inputs as inputs."""
    signature = TASK_SIGNATURES[task]
    input_fields = list(signature.input_fields)
    examples: list[dspy.Example] = []
    with Path(path).expanduser().open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            missing = [f for f in input_fields if f not in record]
            if missing:
                raise ValueError(
                    f"{path}:{lineno}: missing input fields {missing} for task "
                    f"'{task}' (expected {input_fields})"
                )
            examples.append(dspy.Example(**record).with_inputs(*input_fields))
    if not examples:
        raise ValueError(f"{path}: empty trainset for task '{task}'")
    return examples


def fixture_examples(
    task: str, splits: Sequence[str] = ("train",), fixture: Fixture | None = None
) -> list[dspy.Example]:
    """Optimizer examples from the frozen extraction fixture.

    An example holds only the signature inputs and the ``case_id`` the metric
    resolves; claims and support spans stay in the evaluator. Sealed-test
    cases are refused.
    """
    if task not in EXTRACTION_TASKS:
        raise ValueError(f"no extraction fixture for task '{task}'")
    if SEALED_SPLIT in splits:
        raise SealedSplitError(f"{SEALED_SPLIT} cases never become optimizer examples")
    fixture = fixture or load_fixture()
    input_fields = list(TASK_SIGNATURES[task].input_fields)
    return [
        dspy.Example(**case.inputs(), case_id=case.case_id).with_inputs(*input_fields)
        for case in fixture.select(task=task, splits=splits)
    ]


def check_trainset(task: str, examples: Sequence[dspy.Example]) -> None:
    """Refuse optimizer examples the frozen evaluator cannot score or must not see.

    Extraction examples must reference a non-sealed fixture case by
    ``case_id``: without annotated source support there is nothing to check
    a fact against, and a sealed case would leak test gold into search.
    """
    fixture = load_fixture()
    sealed = {c.case_id for c in fixture.select(splits=(SEALED_SPLIT,))}
    for i, ex in enumerate(examples):
        case_id = ex.get("case_id")
        if case_id in sealed:
            raise SealedSplitError(f"example {i}: case {case_id!r} is in {SEALED_SPLIT}")
        if task not in EXTRACTION_TASKS:
            continue
        if case_id is None:
            raise ValueError(
                f"example {i}: '{task}' examples need a case_id from the annotated "
                f"extraction fixture; unannotated gold cannot score source support"
            )
        case = fixture.case(case_id)
        if case.task != task:
            raise ValueError(f"example {i}: case {case_id!r} is a '{case.task}' case")


def metric_for(task: str, fixture: Fixture | None = None) -> Callable:
    """Optimizer metric for ``task``.

    Fact and event extraction use the frozen source-backed evaluator
    (``extraction_eval``). Called MIPROv2-style ``(gold, pred, trace)`` it
    returns the scalar; called GEPA-style with ``pred_name`` it returns
    ``dspy.Prediction(score=..., feedback=...)``. Other tasks keep the
    structural parse-or-exact-match metric.
    """
    if task in EXTRACTION_TASKS:
        return _evidence_metric(task, fixture)
    return _structural_metric(task)


def _evidence_metric(task: str, fixture: Fixture | None) -> Callable:
    output_field = EXTRACTION_TASKS[task]

    def metric(
        gold: dspy.Example,
        pred: dspy.Prediction,
        trace: Any = None,  # noqa: ARG001
        pred_name: str | None = None,
        pred_trace: Any = None,  # noqa: ARG001
    ) -> float | dspy.Prediction:
        case_id = gold.get("case_id")
        if case_id is None:
            raise ValueError(f"'{task}' metric needs an example with a fixture case_id")
        case = (fixture or load_fixture()).case(case_id)
        result = score_case(case, getattr(pred, output_field, None), purpose="optimize")
        if pred_name is None:
            return result.score
        return dspy.Prediction(score=result.score, feedback=result.feedback)

    return metric


def _structural_metric(task: str) -> Callable:
    """Parse validity scores 0.5; exact agreement with gold scores 1.0."""
    signature = TASK_SIGNATURES[task]
    output_fields = list(signature.output_fields)

    def metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:  # noqa: ARG001
        score = 0.5  # typed parse succeeded (we got a Prediction at all)
        for field in output_fields:
            gold = getattr(example, field, None)
            got = getattr(pred, field, None)
            if gold is None or got is None:
                return score
            got_dump = [
                v.model_dump() if hasattr(v, "model_dump") else v
                for v in (got if isinstance(got, list) else [got])
            ]
            gold_norm = gold if isinstance(gold, list) else [gold]
            if got_dump != gold_norm:
                return score
        return 1.0

    return metric


def _default_teleprompter_factory(cfg: AppConfig) -> TeleprompterFactory:
    auto = cfg.dspy.optimizer.auto
    if auto not in ("light", "medium", "heavy"):
        raise ValueError(
            f"dspy.optimizer.auto '{auto}' invalid; expected light|medium|heavy"
        )

    def factory(metric: Callable) -> Any:
        return dspy.MIPROv2(
            metric=metric, auto=cast(Literal["light", "medium", "heavy"], auto)
        )

    return factory


def optimize_task(
    task: str,
    cfg: AppConfig,
    trainset_path: str | Path | Sequence[dspy.Example],
    *,
    teleprompter_factory: TeleprompterFactory | None = None,
) -> Path:
    """Compile one task offline and persist the artifact; returns its dir.

    ``trainset_path`` is a JSONL path or already-built examples (for example
    ``fixture_examples``). The trainset is checked before any model call.
    """
    trainset = (
        list(trainset_path)
        if not isinstance(trainset_path, str | Path)
        else load_trainset(trainset_path, task)
    )
    check_trainset(task, trainset)
    if teleprompter_factory is None:
        entry = cfg.models[cfg.dspy.tasks[task].model]
        key_env = entry.api_key_env or "OPENROUTER_API_KEY"
        if not os.environ.get(key_env):
            raise RuntimeError(
                f"optimization for task '{task}' calls the real model "
                f"'{cfg.dspy.tasks[task].model}' and needs {key_env} set; "
                f"export it (or pass a teleprompter_factory for dry runs)"
            )
        teleprompter_factory = _default_teleprompter_factory(cfg)

    student = build_module(task, cfg)
    teleprompter = teleprompter_factory(metric_for(task))

    router = LMRouter(cfg)
    with router.bind(task):
        compiled = teleprompter.compile(student, trainset=trainset)

    return save_artifact(
        compiled,
        cfg.dspy.artifacts.dir,
        task,
        model_ref=cfg.dspy.tasks[task].model,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m gluomem.llm.optimize",
        description="Offline DSPy prompt optimization -> artifacts/ (costs tokens).",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=[*sorted(TASK_SIGNATURES), "all"],
        help="task to optimize, or 'all'",
    )
    parser.add_argument(
        "--trainset",
        required=True,
        help="JSONL trainset path (for 'all': a directory holding <task>.jsonl)",
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    tasks = sorted(TASK_SIGNATURES) if args.task == "all" else [args.task]
    for task in tasks:
        trainset = (
            Path(args.trainset) / f"{task}.jsonl"
            if args.task == "all"
            else Path(args.trainset)
        )
        target = optimize_task(task, cfg, trainset)
        print(f"optimized '{task}' -> {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
