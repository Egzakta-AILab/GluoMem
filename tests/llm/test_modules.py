"""Module-mapping + DspyExtractors end-to-end under DummyLM (offline).

Covers: config->module dispatch, Refine wrapping (retry-on-invalid, all-fail
-> ExtractionError), and every facade method returning validated Pydantic.
"""

from __future__ import annotations

from datetime import datetime

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from gluomem.config.app_config import load_config
from gluomem.llm.modules import DspyExtractors, _refine_wrap, build_module
from gluomem.llm.outputs import ExtractionError
from gluomem.llm.router import LMRouter


def _cfg(**over):
    cfg = load_config()
    return cfg.with_overrides(over) if over else cfg


class _CtxRouter(LMRouter):
    """LMRouter double whose bind() injects a DummyLM instead of OpenRouter."""

    def __init__(self, cfg, lm) -> None:
        super().__init__(cfg)
        self._lm = lm

    def lm_for(self, task: str) -> dspy.LM:  # noqa: ARG002
        return self._lm


def _extractors(lm, **over) -> DspyExtractors:
    cfg = _cfg(**over)
    return DspyExtractors(cfg, router=_CtxRouter(cfg, lm))


def test_build_module_mapping() -> None:
    cfg = _cfg()
    # Experiment baseline: every task is plain Predict.
    assert isinstance(build_module("fact_extraction", cfg), dspy.Predict)
    assert isinstance(build_module("cognify", cfg), dspy.Predict)
    contradiction = build_module("contradiction", cfg)
    assert isinstance(contradiction, dspy.Predict)
    assert not isinstance(contradiction, dspy.ChainOfThought)
    # cot stays one config flip away (ablation).
    cot_cfg = _cfg(**{"dspy.tasks.contradiction.module": "cot"})
    assert isinstance(build_module("contradiction", cot_cfg), dspy.ChainOfThought)
    with pytest.raises(KeyError, match="unknown dspy task"):
        build_module("nope", cfg)
    bad = _cfg(**{"dspy.tasks.cognify.module": "reflect"})
    with pytest.raises(ValueError, match="invalid"):
        build_module("cognify", bad)


def test_refine_wrap_only_when_enabled() -> None:
    # Default is OFF (dspy.Refine forces temperature=1.0 per attempt); the
    # machinery stays available as a pure config flip.
    cfg = _cfg()
    base = build_module("event_extraction", cfg)
    assert _refine_wrap(base, cfg.dspy.tasks["event_extraction"]) is base
    enabled = _cfg(**{"dspy.tasks.event_extraction.refine": {"enabled": True, "N": 2}})
    wrapped = _refine_wrap(base, enabled.dspy.tasks["event_extraction"])
    assert isinstance(wrapped, dspy.Refine)
    plain = _refine_wrap(base, cfg.dspy.tasks["fact_extraction"])
    assert plain is base


def test_extract_facts_end_to_end() -> None:
    lm = DummyLM([{"facts": [{"text": "Alice lives in Boston.", "confidence": 0.9}]}])
    fx = _extractors(lm)
    facts = fx.extract_facts("Alice lives in Boston!")
    assert len(facts) == 1
    assert facts[0].text == "Alice lives in Boston."
    assert facts[0].confidence == 0.9


def test_extract_events_end_to_end_with_datetime() -> None:
    lm = DummyLM(
        [
            {
                "events": [
                    {
                        "text": "moved to Seattle",
                        "event_date": "2023-05-07T00:00:00",
                        "confidence": 1.0,
                    }
                ]
            }
        ]
    )
    fx = _extractors(lm)
    events = fx.extract_events("She moved yesterday.", datetime(2023, 5, 8))
    assert events[0].event_date == datetime(2023, 5, 7)


def test_extract_profile_routes_through_fact_task() -> None:
    lm = DummyLM([{"facts": [{"text": "Alice loves hiking.", "confidence": 1.0}]}])
    fx = _extractors(lm)
    facts = fx.extract_profile("I love hiking.", speaker="Alice")
    assert facts[0].text == "Alice loves hiking."


def test_cognify_end_to_end() -> None:
    lm = DummyLM(
        [
            {
                "entities": [{"entity_type": "person", "name": "alice", "properties": {}}],
                "relations": [
                    {
                        "source": "alice",
                        "target": "boston",
                        "relation_type": "LIVES_IN",
                        "confidence": 1.0,
                    }
                ],
            }
        ]
    )
    fx = _extractors(lm)
    result = fx.cognify("Alice lives in Boston.")
    assert result.entities[0].name == "alice"
    assert result.relations[0].relation_type == "LIVES_IN"


def test_check_contradiction_end_to_end_predict() -> None:
    lm = DummyLM(
        [
            {
                "verdict": {
                    "is_contradiction": True,
                    "description": "moved cities",
                    "confidence": 0.8,
                },
            }
        ]
    )
    fx = _extractors(lm)
    verdict = fx.check_contradiction("lives in Seattle", "lives in Boston")
    assert verdict.is_contradiction and verdict.description == "moved cities"


def test_check_contradiction_cot_flip() -> None:
    lm = DummyLM(
        [
            {
                "reasoning": "Boston and Seattle conflict.",
                "verdict": {
                    "is_contradiction": True,
                    "description": "moved cities",
                    "confidence": 0.8,
                },
            }
        ]
    )
    fx = _extractors(lm, **{"dspy.tasks.contradiction.module": "cot"})
    verdict = fx.check_contradiction("lives in Seattle", "lives in Boston")
    assert verdict.is_contradiction


class _StatefulDummyLM(DummyLM):
    """DummyLM whose answer position survives Refine's per-attempt deepcopy.

    ``Refine`` runs each attempt on ``lm.copy(rollout_id=..., temperature=1.0)``
    which DEEP-COPIES the LM — with a plain DummyLM every attempt replays the
    same first answer. Sharing the instance across copies makes multi-attempt
    scripts deterministic.
    """

    def __deepcopy__(self, memo):  # noqa: ARG002
        return self


def test_refine_retries_invalid_then_valid() -> None:
    """First attempt unparseable for the typed field -> Refine retries.

    A failed ChatAdapter parse internally falls back to JSONAdapter (a second
    LM call) before the attempt fails, so attempt 1 consumes TWO bad answers;
    attempt 2 gets the valid third answer.
    """
    bad = {"events": "not-a-list"}
    good = {"events": [{"text": "ok", "event_date": None, "confidence": 1.0}]}
    lm = _StatefulDummyLM([bad, bad, good])
    fx = _extractors(
        lm, **{"dspy.tasks.event_extraction.refine": {"enabled": True, "N": 2}}
    )
    events = fx.extract_events("whatever", datetime(2023, 5, 8))
    assert len(events) == 1 and events[0].text == "ok"


def test_all_attempts_fail_raises_extraction_error() -> None:
    lm = DummyLM([{"events": "junk"}, {"events": "junk"}, {"events": "junk"}])
    fx = _extractors(lm)
    with pytest.raises(ExtractionError):
        fx.extract_events("whatever", datetime(2023, 5, 8))


def test_make_extractors_dspy_backend_builds_real_extractors() -> None:
    from gluomem.llm import make_extractors

    fx = make_extractors(_cfg())
    assert isinstance(fx, DspyExtractors)
