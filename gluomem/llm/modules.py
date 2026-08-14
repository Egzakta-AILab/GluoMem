"""gluomem.llm.modules — config→module mapping + the dspy-backed ``Extractors``.

THE ONLY package whose import pulls dspy transitively is this module tree
(router/signatures/modules/judge/bootstrap/optimize). Never import it at
module top level outside ``gluomem/llm/``; go through
``gluomem.llm.make_extractors`` instead.

Module mapping (config ``dspy.tasks.<task>.module``): ``predict`` →
``dspy.Predict``, ``cot`` → ``dspy.ChainOfThought``; a task with
``refine: {enabled: true, N: k}`` is wrapped in ``dspy.Refine`` for
output-validity retries.

Refine notes (dspy 3.2.1 behaviour):

- the reward fn runs only AFTER the adapter successfully parsed the typed
  output — returning 1.0 with ``threshold=1.0`` therefore means "first valid
  parse wins" and the OfferFeedback LLM path never triggers;
- a parse failure raises inside the attempt and Refine retries with a new
  rollout id at ``temperature=1.0`` (the task temperature is bypassed by
  design during retries);
- ``Refine`` MUTATES ``self.fail_count`` across calls and returns ``None``
  when every attempt failed — so the wrapper is constructed per invocation
  and a ``None`` prediction becomes ``ExtractionError``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dspy

from gluomem.llm.bootstrap import load_or_build
from gluomem.llm.outputs import (
    CognifyResult,
    ContradictionVerdict,
    ExtractedEvent,
    ExtractedFact,
    ExtractionError,
)
from gluomem.llm.router import LMRouter
from gluomem.llm.signatures import TASK_SIGNATURES

if TYPE_CHECKING:
    from datetime import datetime

    from gluomem.config.app_config import AppConfig, DspyTaskConfig


def _parse_validity_reward(args: dict, pred: dspy.Prediction) -> float:  # noqa: ARG001
    """Reaching this fn means the adapter parsed the typed output — valid."""
    return 1.0


def build_module(task: str, cfg: AppConfig) -> dspy.Module:
    """Build the (unwrapped) dspy module for ``task`` from config."""
    try:
        signature = TASK_SIGNATURES[task]
    except KeyError as exc:
        known = ", ".join(sorted(TASK_SIGNATURES))
        raise KeyError(f"unknown dspy task '{task}'; known: {known}") from exc
    if task == "cognify" and cfg.kg.entity_type_prompt_enum:
        from gluomem.llm.signatures import make_cognify_signature

        signature = make_cognify_signature(cfg.kg.entity_type_allowlist)
    tc = cfg.dspy.tasks[task]
    if tc.module == "predict":
        return dspy.Predict(signature)
    if tc.module == "cot":
        return dspy.ChainOfThought(signature)
    raise ValueError(
        f"dspy.tasks.{task}.module '{tc.module}' invalid; expected 'predict' or 'cot'"
    )


def _refine_wrap(module: dspy.Module, tc: DspyTaskConfig) -> dspy.Module:
    """Per-invocation Refine wrapper (Refine mutates fail_count across calls)."""
    refine = tc.refine or {}
    if not refine.get("enabled", False):
        return module
    n = int(refine.get("N", 1))
    return dspy.Refine(
        module=module,
        N=n,
        reward_fn=_parse_validity_reward,
        threshold=1.0,
        fail_count=n,
    )


class DspyExtractors:
    """dspy-backed ``Extractors`` (fact/event/profile/cognify/contradiction).

    Profile extraction deliberately reuses the ``fact_extraction`` task over
    per-speaker text (the canonical config surface has four tasks; profile is
    a lane, not a separate signature).
    """

    def __init__(self, cfg: AppConfig, *, router: LMRouter | None = None) -> None:
        from gluomem.llm.concurrency import ConcurrencyGate
        from gluomem.llm.usage import UsageAccumulator

        self._cfg = cfg
        self._router = router if router is not None else LMRouter(cfg)
        self._modules: dict[str, dspy.Module] = {}
        # Token accounting across every extraction call (thread-safe; the
        # write path fans out). Consumers read .usage.snapshot().
        self.usage = UsageAccumulator()
        # Process-wide throttle: however many threads the batch/session pools
        # spawn, at most max_concurrent_llm extraction calls hit the API at once.
        self._llm_gate = ConcurrencyGate(cfg.write.max_concurrent_llm)

    def _module(self, task: str) -> dspy.Module:
        if task not in self._modules:
            self._modules[task] = load_or_build(task, self._cfg)
        return _refine_wrap(self._modules[task], self._cfg.dspy.tasks[task])

    def _run(self, task: str, **inputs: Any) -> dspy.Prediction:
        module = self._module(task)
        try:
            with self._llm_gate, self._router.bind(task):
                pred = module(**inputs)
        except Exception as exc:
            raise ExtractionError(f"dspy task '{task}' failed: {exc}") from exc
        if pred is None:
            raise ExtractionError(
                f"dspy task '{task}' produced no valid output after retries"
            )
        if self._cfg.dspy.track_usage:
            self.usage.add(pred.get_lm_usage())
        return pred

    # --- Extractors protocol -------------------------------------------------

    def extract_facts(self, text: str) -> list[ExtractedFact]:
        pred = self._run("fact_extraction", text=text)
        return list(pred.facts)

    def extract_events(
        self, source_text: str, reference_date: datetime
    ) -> list[ExtractedEvent]:
        pred = self._run(
            "event_extraction",
            source_text=source_text,
            reference_date=reference_date,
        )
        return list(pred.events)

    def extract_profile(self, text: str, *, speaker: str) -> list[ExtractedFact]:
        pred = self._run(
            "fact_extraction",
            text=f"Speaker {speaker} said:\n{text}",
        )
        return list(pred.facts)

    def cognify(self, text: str) -> CognifyResult:
        pred = self._run("cognify", text=text)
        return CognifyResult(
            entities=list(pred.entities), relations=list(pred.relations)
        )

    def check_contradiction(
        self, new_statement: str, stored_statement: str
    ) -> ContradictionVerdict:
        pred = self._run(
            "contradiction",
            new_statement=new_statement,
            stored_statement=stored_statement,
        )
        return pred.verdict
