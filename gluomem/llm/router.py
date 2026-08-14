"""gluomem.llm.router — LMRouter: config/registry-driven dspy.LM construction.

The router resolves each task's model NAME through the ``ModelRegistry``
and binds the resulting ``dspy.LM`` via ``dspy.context`` — a model swap is
a registry-name change, never a code change.

All LLM traffic rides OpenRouter (LiteLLM ``openrouter/<vendor>/<model>``
strings, ``OPENROUTER_API_KEY``). ``ModelEntry.extra`` is passed verbatim as
extra ``dspy.LM`` kwargs → LiteLLM, carrying e.g.
``extra_body={"provider": {"order": [...], "allow_fallbacks": False,
"require_parameters": True}}`` for reproducible backend pinning.

``bind()`` uses ``dspy.context`` (thread/task-local), NEVER the global
``dspy.configure`` — no cross-test or cross-caller bleed. The disk cache is
configured once, lazily, from ``cfg.dspy.cache`` (dspy would otherwise
default to ``~/.dspy_cache``).
"""

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
            disk_cache_dir=cache_dir,
        )
        _cache_configured = True


def _make_adapter(name: str) -> dspy.Adapter:
    if name == "chat":
        return dspy.ChatAdapter()
    if name == "json":
        return dspy.JSONAdapter()
    raise ValueError(f"unknown dspy.adapter {name!r}; expected 'chat' or 'json'")


class LMRouter:
    """Per-task LM resolution + context binding (config-pure)."""

    def __init__(self, cfg: AppConfig, registry: ModelRegistry | None = None) -> None:
        self._cfg = cfg
        self._registry = registry if registry is not None else ModelRegistry(cfg)
        self._adapter = _make_adapter(cfg.dspy.adapter)
        _ensure_cache(cfg)

    # --- LM construction -----------------------------------------------------

    def _task_cfg(self, task: str) -> DspyTaskConfig:
        try:
            return self._cfg.dspy.tasks[task]
        except KeyError as exc:
            known = ", ".join(sorted(self._cfg.dspy.tasks)) or "(none)"
            raise KeyError(
                f"unknown dspy task '{task}'; configured tasks: {known}"
            ) from exc

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
        return dspy.LM(
            entry.provider_model,
            api_key=os.environ.get(entry.api_key_env or "", ""),
            cache=self._cfg.dspy.cache.disk,
            **kwargs,
        )

    def _reasoning_extra_body(self, entry: ModelEntry) -> dict[str, Any]:
        """Merge the global hidden-reasoning policy into the entry's extra_body.

        ``dspy.reasoning_enabled`` (False by default) is injected for
        OpenRouter-routed models; an entry-level
        ``extra.extra_body.reasoning`` always wins; ``None`` sends nothing
        (provider default). OpenRouter ignores the parameter for models
        without a reasoning mode, so blanket injection is safe.
        """
        extra_body: dict[str, Any] = dict(entry.extra.get("extra_body") or {})
        policy = self._cfg.dspy.reasoning_enabled
        if (
            entry.provider == "openrouter"
            and policy is not None
            and "reasoning" not in extra_body
        ):
            extra_body["reasoning"] = {"enabled": policy}
        return extra_body

    def lm_for(self, task: str) -> dspy.LM:
        """The ``dspy.LM`` for a configured task."""
        tc = self._task_cfg(task)
        entry = self._registry.resolve(tc.model)
        return self._build_lm(
            entry, temperature=tc.temperature, max_tokens=tc.max_tokens
        )

    def lm_for_model(
        self,
        name: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dspy.LM:
        """A ``dspy.LM`` for a registry model name (judge / benchmark harness)."""
        entry = self._registry.resolve(name)
        return self._build_lm(entry, temperature=temperature, max_tokens=max_tokens)

    # --- live-path preflight ---------------------------------------------------

    def assert_api_key(self, model_name: str) -> None:
        """Fail loud when the model's ``api_key_env`` is missing or empty.

        LIVE-path preflight only (``bind`` and the bench runner's non-fake
        answer/judge factories). LM *construction* stays keyless — the offline
        gate builds and inspects LMs without any key — so this runs only where
        the provider is about to be called for real. An EMPTY value counts as
        missing: an empty key cannot authenticate and would otherwise surface
        as a cryptic provider 401 mid-run.
        """
        entry = self._registry.resolve(model_name)
        env = entry.api_key_env
        if env and not os.environ.get(env):
            raise RuntimeError(
                f"model '{model_name}' ({entry.provider_model}) needs the "
                f"{env} environment variable (missing or empty); export it "
                f"to run live"
            )

    # --- context binding -------------------------------------------------------

    @contextmanager
    def bind(self, task: str) -> Iterator[None]:
        """Bind the task's LM + adapter + usage tracking for the duration.

        Binding is the live entry for task execution, so the model's API key
        is preflighted here (clear error instead of a provider 401 later).
        """
        self.assert_api_key(self._task_cfg(task).model)
        with dspy.context(
            lm=self.lm_for(task),
            adapter=self._adapter,
            track_usage=self._cfg.dspy.track_usage,
        ):
            yield
