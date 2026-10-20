"""Bounded transient retries at the extraction boundary, not semantic repair."""

import asyncio
from contextlib import nullcontext
from typing import cast

import dspy
import pytest
from dspy.utils.exceptions import LMServerError

from gluomem.config.app_config import AppConfig
from gluomem.llm.modules import DspyExtractors
from gluomem.llm.router import LMRouter


def test_transient_failure_retries_and_records_every_attempt(caplog):
    class Router:
        def bind(self, task):
            return nullcontext()

    cfg = AppConfig(
        dspy={
            "model": "mock",
            "track_usage": False,
            "retry": {"max_attempts": 3, "initial_delay_seconds": 0, "max_delay_seconds": 0},
        }
    )
    extractor = DspyExtractors(cfg, router=cast(LMRouter, Router()))
    calls = []

    def module(**inputs):
        calls.append(inputs)
        if len(calls) < 3:
            raise LMServerError("PRIVATE_SERVER_TEXT", status=502)
        return dspy.Prediction(claims=[])

    extractor._modules["event_claims"] = cast(dspy.Module, module)
    with caplog.at_level("INFO", logger="gluomem.llm.retry"):
        assert extractor.extract_event_claims("PRIVATE_INPUT") == []
    assert len(calls) == 3
    records = [r for r in caplog.records if r.name == "gluomem.llm.retry"]
    assert [r.attempt for r in records] == [1, 2, 3]
    assert [r.outcome for r in records] == ["retrying", "retrying", "succeeded"]
    assert [r.http_status for r in records] == [502, 502, None]
    assert len({r.operation_id for r in records}) == 1
    assert all(r.max_attempts == 3 and r.task == "event_claims" for r in records)
    assert "PRIVATE_SERVER_TEXT" not in caplog.text and "PRIVATE_INPUT" not in caplog.text
    assert all(r.exc_info is None for r in records)


def test_exponential_backoff_is_capped_and_stops_at_max_attempts(monkeypatch, caplog):
    from gluomem.config.app_config import DspyRetryConfig
    from gluomem.llm import retry

    delays = []
    monkeypatch.setattr(retry.time, "sleep", delays.append)
    original = LMServerError("PRIVATE_FAILURE", status=503)
    calls = []

    def fail():
        calls.append(None)
        raise original

    with caplog.at_level("INFO", logger="gluomem.llm.retry"):
        with pytest.raises(LMServerError) as caught:
            retry.run_with_retry(
                fail,
                task="consolidate",
                policy=DspyRetryConfig(
                    max_attempts=5,
                    initial_delay_seconds=1,
                    max_delay_seconds=3,
                ),
            )
    assert caught.value is original
    assert len(calls) == 5 and delays == [1, 2, 3, 3]
    assert [r.outcome for r in caplog.records] == ["retrying"] * 4 + ["failed"]


def test_router_disables_sdk_retry_multiplication():
    cfg = AppConfig()
    cfg.models[cfg.dspy.model].extra["num_retries"] = 7
    assert LMRouter(cfg).lm_for("consolidate").num_retries == 0


def test_compiled_lm_cannot_restore_hidden_sdk_retries(tmp_path):
    from gluomem.llm.bootstrap import load_or_build, save_artifact
    from gluomem.llm.modules import build_module

    cfg = AppConfig()
    cfg.dspy.artifacts.dir = str(tmp_path)
    module = build_module("consolidate", cfg)
    module.set_lm(dspy.LM("openai/authored-fixture", num_retries=7, cache=False))
    save_artifact(module, tmp_path, "consolidate", model_ref="authored-fixture")
    loaded = load_or_build("consolidate", cfg)
    lm = loaded.predictors()[0].lm
    assert isinstance(lm, dspy.LM) and lm.num_retries == 0


def test_compiled_refine_cannot_bypass_attempt_budget(tmp_path, capsys):
    from gluomem.config.app_config import DspyRetryConfig
    from gluomem.llm.bootstrap import load_or_build, save_artifact
    from gluomem.llm.modules import build_module

    class Wrapper(dspy.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

    cfg = AppConfig()
    cfg.dspy.retry = DspyRetryConfig(max_attempts=2)
    for nested in (False, True):
        cfg.dspy.artifacts.dir = str(tmp_path / str(nested))
        inner = dspy.Refine(
            module=build_module("consolidate", cfg),
            N=2,
            reward_fn=lambda args, pred: 1.0,
            threshold=1.0,
        )
        module = Wrapper(inner) if nested else inner
        save_artifact(module, cfg.dspy.artifacts.dir, "consolidate", model_ref="authored-fixture")
        with pytest.raises(ValueError, match="compiled Refine"):
            load_or_build("consolidate", cfg)
    assert capsys.readouterr().out == ""


def test_retry_budget_refuses_nested_validation_and_refine_attempts():
    from gluomem.config.app_config import DspyConfig

    for extra in (
        {"validation_retries": 1},
        {"tasks": {"consolidate": {"module": "predict", "refine": {"enabled": True}}}},
    ):
        with pytest.raises(ValueError, match="retry"):
            DspyConfig.model_validate(
                {
                    "model": "mock",
                    "adapter": "json",
                    "retry": {"max_attempts": 2},
                    **extra,
                }
            )


def test_retry_config_has_safe_defaults_and_rejects_invalid_limits(monkeypatch):
    from gluomem.config.app_config import DspyRetryConfig

    assert DspyRetryConfig().max_attempts == 1
    for invalid in (
        {"max_attempts": 0},
        {"max_attempts": True},
        {"max_attempts": 1.5},
        {"initial_delay_seconds": -1},
        {"max_delay_seconds": float("inf")},
        {"initial_delay_seconds": 3, "max_delay_seconds": 2},
        {"attempts": 3},
    ):
        with pytest.raises(ValueError):
            DspyRetryConfig.model_validate(invalid)
    monkeypatch.setenv("GLUOMEM_DSPY__RETRY__MAX_ATTEMPTS", "3")
    monkeypatch.setenv("GLUOMEM_DSPY__RETRY__INITIAL_DELAY_SECONDS", "1")
    monkeypatch.setenv("GLUOMEM_DSPY__RETRY__MAX_DELAY_SECONDS", "30")
    assert AppConfig().dspy.retry == DspyRetryConfig(max_attempts=3)


def test_streaming_refuses_task_retries_before_dispatch():
    from gluomem.llm.outputs import ExtractionError

    cfg = AppConfig(dspy={"model": "mock", "retry": {"max_attempts": 3}})
    extractor = DspyExtractors(cfg)
    with dspy.context(send_stream=object()):
        with pytest.raises(ExtractionError, match="non-streaming"):
            extractor.extract_event_claims("PRIVATE_INPUT")


def test_nontransient_errors_and_cancellation_never_retry(monkeypatch, caplog):
    from dspy.utils.exceptions import AdapterParseError, LMAuthError, LMInvalidRequestError

    from gluomem.config.app_config import DspyRetryConfig
    from gluomem.llm import retry
    from gluomem.llm.signatures import ClaimConsolidation

    delays = []
    monkeypatch.setattr(retry.time, "sleep", delays.append)
    errors = [
        LMAuthError("PRIVATE"),
        LMInvalidRequestError("PRIVATE"),
        ValueError("PRIVATE"),
        AdapterParseError("test", cast(dspy.Signature, ClaimConsolidation), "PRIVATE"),
        asyncio.CancelledError(),
    ]
    for original in errors:
        calls = []

        def fail(calls=calls, original=original):
            calls.append(None)
            raise original

        with pytest.raises(type(original)) as caught:
            retry.run_with_retry(fail, task="consolidate", policy=DspyRetryConfig(max_attempts=3))
        assert caught.value is original and len(calls) == 1
    assert delays == []
    assert len(caplog.records) == len(errors)


def test_rate_limit_hint_is_honored_without_exceeding_delay_cap(monkeypatch):
    from dspy.utils.exceptions import LMRateLimitError

    from gluomem.config.app_config import DspyRetryConfig
    from gluomem.llm import retry

    policy = DspyRetryConfig(max_attempts=2, initial_delay_seconds=1, max_delay_seconds=5)
    delays = []
    monkeypatch.setattr(retry.time, "sleep", delays.append)
    for hint in (3, 20):
        calls = []
        original = LMRateLimitError("PRIVATE", status=429, retry_after=hint)

        def request(calls=calls, original=original):
            calls.append(None)
            if len(calls) == 1:
                raise original
            return "ok"

        if hint <= policy.max_delay_seconds:
            assert retry.run_with_retry(request, task="consolidate", policy=policy) == "ok"
            assert len(calls) == 2
        else:
            with pytest.raises(LMRateLimitError) as caught:
                retry.run_with_retry(request, task="consolidate", policy=policy)
            assert caught.value is original and len(calls) == 1
    assert delays == [3]
