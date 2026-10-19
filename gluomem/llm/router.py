"""LMRouter: config- and registry-driven ``dspy.LM`` construction."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dspy

from gluomem.config.registry import ModelRegistry

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gluomem.config.app_config import AppConfig, DspyTaskConfig, ModelEntry

_cache_lock = threading.Lock()
_cache_configured = False


def _ensure_cache(cfg: AppConfig) -> None:
    """Configure the dspy disk cache once per process from ``cfg.dspy.cache``."""
    global _cache_configured
    with _cache_lock:
        if _cache_configured:
            return
        cache_dir = str(Path(cfg.dspy.cache.dir).expanduser())
        dspy.configure_cache(
            enable_disk_cache=cfg.dspy.cache.disk,
            enable_memory_cache=False,
            disk_cache_dir=cache_dir,
        )
        _cache_configured = True


def _make_adapter(
    name: str,
    *,
    canonicalize: bool = False,
    schema_mode: str = "auto",
    schema_strict: bool = False,
    validation_retries: int = 0,
) -> dspy.Adapter:
    from gluomem.llm.structured_adapter import make_adapter

    return make_adapter(
        name,
        canonicalize=canonicalize,
        schema_mode=schema_mode,
        schema_strict=schema_strict,
        validation_retries=validation_retries,
    )


class LMRouter:
    """Per-task LM resolution and context binding."""

    def __init__(self, cfg: AppConfig, registry: ModelRegistry | None = None) -> None:
        self._cfg = cfg
        self._registry = registry if registry is not None else ModelRegistry(cfg)
        self._adapter = _make_adapter(
            cfg.dspy.adapter,
            canonicalize=cfg.dspy.canonicalize,
            schema_mode=cfg.dspy.schema_mode,
            schema_strict=cfg.dspy.schema_strict,
            validation_retries=cfg.dspy.validation_retries,
        )
        _ensure_cache(cfg)

    def _task_cfg(self, task: str) -> DspyTaskConfig:
        try:
            return self._cfg.dspy.tasks[task]
        except KeyError as exc:
            known = ", ".join(sorted(self._cfg.dspy.tasks)) or "(none)"
            raise KeyError(f"unknown dspy task '{task}'; configured tasks: {known}") from exc

    def _build_lm(
        self,
        entry: ModelEntry,
        *,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dspy.LM:
        kwargs: dict[str, Any] = dict(entry.extra)
        kwargs["extra_body"] = self._reasoning_extra_body(entry)
        if not kwargs["extra_body"]:
            del kwargs["extra_body"]
        if entry.api_base is not None:
            kwargs["api_base"] = entry.api_base
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        api_key = os.environ.get(entry.api_key_env or "", "")
        try:
            return dspy.LM(
                entry.provider_model,
                api_key=api_key,
                cache=self._cfg.dspy.cache.disk,
                **kwargs,
            )
        except dspy.utils.exceptions.LMConfigurationError:
            reasoning = (kwargs.get("extra_body") or {}).get("reasoning") or {}
            if reasoning.get("enabled") is not False:
                raise
            clean_kwargs = {
                k: v for k, v in kwargs.items() if k not in ("temperature", "max_tokens")
            }
            lm = dspy.LM(
                entry.provider_model,
                api_key=api_key,
                cache=self._cfg.dspy.cache.disk,
                **clean_kwargs,
            )
            if temperature is not None:
                lm.kwargs["temperature"] = temperature
            if max_tokens is not None:
                lm.kwargs["max_tokens"] = max_tokens
            return lm

    def _reasoning_extra_body(self, entry: ModelEntry) -> dict[str, Any]:
        extra_body: dict[str, Any] = dict(entry.extra.get("extra_body") or {})
        policy = self._cfg.dspy.reasoning_enabled
        if entry.provider == "openrouter" and policy is not None and "reasoning" not in extra_body:
            extra_body["reasoning"] = {"enabled": policy}
        return extra_body

    def lm_for(self, task: str) -> dspy.LM:
        """Return the ``dspy.LM`` for a configured task."""
        tc = self._task_cfg(task)
        entry = self._registry.resolve(tc.model)
        return self._build_lm(entry, temperature=tc.temperature, max_tokens=tc.max_tokens)

    def lm_for_model(
        self,
        name: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dspy.LM:
        """Return a ``dspy.LM`` for a registry model name."""
        entry = self._registry.resolve(name)
        return self._build_lm(entry, temperature=temperature, max_tokens=max_tokens)

    def assert_api_key(self, model_name: str) -> None:
        """Fail loudly when the model's ``api_key_env`` is missing or empty."""
        entry = self._registry.resolve(model_name)
        env = entry.api_key_env
        if env and not os.environ.get(env):
            raise RuntimeError(
                f"model '{model_name}' ({entry.provider_model}) needs the "
                f"{env} environment variable (missing or empty); export it "
                f"to run live"
            )

    @contextmanager
    def bind(self, task: str) -> Iterator[None]:
        """Bind the task's LM, adapter, and usage tracking for the duration."""
        self.assert_api_key(self._task_cfg(task).model)
        with dspy.context(
            lm=self.lm_for(task),
            adapter=self._adapter,
            track_usage=self._cfg.dspy.track_usage,
        ):
            yield
