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
``fact_extraction``::

    {"text": "...", "facts": [{"text": "...", "confidence": 1.0}]}
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import dspy

from gluomem.config.app_config import AppConfig, load_config
from gluomem.llm.bootstrap import save_artifact
from gluomem.llm.modules import build_module
from gluomem.llm.router import LMRouter
from gluomem.llm.signatures import TASK_SIGNATURES

# Anything with .compile(student, trainset=...) — tests inject fakes.
TeleprompterFactory = Callable[[Callable], Any]


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


def metric_for(task: str) -> Callable:
    """Validity + gold-agreement metric for ``task`` (used by the optimizer).

    Baseline metric: the prediction parsed into the typed output (validity)
    scores 0.5; exact agreement with the gold output field scores 1.0.
    Refine-style validity is the floor so optimization never rewards
    unparseable outputs.
    """
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
    trainset_path: str | Path,
    *,
    teleprompter_factory: TeleprompterFactory | None = None,
) -> Path:
    """Compile one task offline and persist the artifact; returns its dir."""
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

    trainset = load_trainset(trainset_path, task)
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
