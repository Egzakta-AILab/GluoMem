"""DspyExtractors, FakeExtractors, NullExtractors, and module builder tests."""

from __future__ import annotations

from datetime import datetime

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from gluomem.config.app_config import load_config
from gluomem.llm import (
    CognifyResult,
    ContradictionVerdict,
    ExtractedFact,
    ExtractionError,
    FakeExtractors,
    KGEntity,
    NullExtractors,
    make_extractors,
)
from gluomem.llm.modules import DspyExtractors, _refine_wrap, build_module
from gluomem.llm.router import LMRouter


def _cfg(**over):
    cfg = load_config()
    return cfg.with_overrides(over) if over else cfg


class _CtxRouter(LMRouter):
    def __init__(self, cfg, lm) -> None:
        super().__init__(cfg)
        self._lm = lm

    def assert_api_key(self, model_name: str) -> None:  # noqa: ARG002
        return

    def lm_for(self, task: str) -> dspy.LM:  # noqa: ARG002
        return self._lm


class _StatefulDummyLM(DummyLM):
    def __deepcopy__(self, memo):  # noqa: ARG002
        return self


def _extractors(lm, **over) -> DspyExtractors:
    cfg = _cfg(**over)
    return DspyExtractors(cfg, router=_CtxRouter(cfg, lm))


def test_build_module_and_refine_wrap() -> None:
    cfg = _cfg()
    for task in ("fact_extraction", "cognify", "contradiction"):
        mod = build_module(task, cfg)
        assert isinstance(mod, dspy.Predict) and not isinstance(mod, dspy.ChainOfThought)

    cot_cfg = _cfg(**{"dspy.tasks.contradiction.module": "cot"})
    assert isinstance(build_module("contradiction", cot_cfg), dspy.ChainOfThought)

    with pytest.raises(KeyError, match="unknown dspy task"):
        build_module("nope", cfg)
    with pytest.raises(ValueError, match="invalid"):
        build_module("cognify", _cfg(**{"dspy.tasks.cognify.module": "reflect"}))

    base = build_module("event_extraction", cfg)
    assert _refine_wrap(base, cfg.dspy.tasks["event_extraction"]) is base
    enabled = _cfg(**{"dspy.tasks.event_extraction.refine": {"enabled": True, "N": 2}})
    assert isinstance(_refine_wrap(base, enabled.dspy.tasks["event_extraction"]), dspy.Refine)


def test_dspy_extractors_all_methods_under_dummylm() -> None:
    fx_fact = _extractors(
        DummyLM([{"facts": [{"text": "Alice lives in Boston.", "confidence": 0.9}]}])
    )
    facts = fx_fact.extract_facts("Alice lives in Boston!")
    assert len(facts) == 1 and facts[0].text == "Alice lives in Boston."

    fx_prof = _extractors(
        DummyLM([{"facts": [{"text": "Alice loves hiking.", "confidence": 1.0}]}])
    )
    prof_out = fx_prof.extract_profile("I love hiking.", speaker="Alice")
    assert prof_out[0].text == "Alice loves hiking."

    fx_ev = _extractors(
        DummyLM(
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
    )
    ev_out = fx_ev.extract_events("She moved yesterday.", datetime(2023, 5, 8))
    assert ev_out[0].event_date == datetime(2023, 5, 7)

    fx_cog = _extractors(
        DummyLM(
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
    )
    cog = fx_cog.cognify("Alice lives in Boston.")
    assert cog.entities[0].name == "alice" and cog.relations[0].relation_type == "LIVES_IN"

    fx_con = _extractors(
        DummyLM(
            [
                {
                    "reasoning": "conflict",
                    "verdict": {
                        "is_contradiction": True,
                        "description": "moved cities",
                        "confidence": 0.8,
                    },
                }
            ]
        ),
        **{"dspy.tasks.contradiction.module": "cot"},
    )
    v = fx_con.check_contradiction("lives in Seattle", "lives in Boston")
    assert v.is_contradiction and v.description == "moved cities"


def test_refine_retries_and_exhaustion_raises_extraction_error() -> None:
    bad = {"events": "not-a-list"}
    good = {"events": [{"text": "ok", "event_date": None, "confidence": 1.0}]}
    lm = _StatefulDummyLM([bad, bad, good])
    fx = _extractors(
        lm, **{"dspy.tasks.event_extraction.refine": {"enabled": True, "N": 2}}
    )
    events = fx.extract_events("whatever", datetime(2023, 5, 8))
    assert len(events) == 1 and events[0].text == "ok"

    fx_fail = _extractors(DummyLM([bad, bad, bad]))
    with pytest.raises(ExtractionError):
        fx_fail.extract_events("whatever", datetime(2023, 5, 8))


def test_fake_and_null_extractors_and_factory(capsys) -> None:
    FakeExtractors()
    assert "FAKE EXTRACTORS" in capsys.readouterr().err

    fx = FakeExtractors(
        cognify_results=[CognifyResult(entities=[KGEntity(entity_type="person", name="alice")])],
        verdicts=[ContradictionVerdict(is_contradiction=True, description="moved")],
        banner=False,
    )
    assert [f.text for f in fx.extract_facts("Alice lives in Boston. She works at Acme.")] == [
        "Alice lives in Boston.",
        "She works at Acme.",
    ]
    ev = fx.extract_events("We met yesterday. The sky is blue.", datetime(2023, 5, 8))
    assert len(ev) == 1 and ev[0].event_date.date().isoformat() == "2023-05-07"  # type: ignore[union-attr]
    assert fx.extract_profile("I love hiking.", speaker="Alice") == [
        ExtractedFact(text="Alice: I love hiking.", confidence=1.0)
    ]
    assert fx.cognify("anything").entities[0].name == "alice"
    assert fx.cognify("anything") == CognifyResult()
    assert fx.check_contradiction("new", "old").is_contradiction

    nx = NullExtractors()
    assert nx.extract_facts("x") == [] and nx.cognify("x") == CognifyResult()
    assert not nx.check_contradiction("a", "b").is_contradiction

    cfg = _cfg()
    assert isinstance(make_extractors(cfg), DspyExtractors)
    assert isinstance(
        make_extractors(
            cfg.with_overrides(
                {"write.extraction_backend": "fake", "components.allow_fake": True}
            )
        ),
        FakeExtractors,
    )
    assert isinstance(
        make_extractors(cfg.with_overrides({"write.extraction_backend": "none"})),
        NullExtractors,
    )
