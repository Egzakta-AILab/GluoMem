"""gluomem.components — component readiness and fake gating.

Every model-backed component (embedder, reranker, extraction, answerer,
judges) is built from config by a factory that either returns the declared
implementation or raises. Nothing falls back to a test double:

- a provider with no implementation raises :class:`ProviderUnavailableError`;
- a missing optional package raises :class:`gluomem.extras.MissingExtraError`;
- a test double (``provider: fake`` embedder, ``fake`` backends) raises
  :class:`FakeComponentError` unless ``components.allow_fake`` is set.

:func:`component_report` describes the effective components of a config
without building models, downloading or calling anything, and without
exposing secrets: it reports API-key environment variable *names* and whether
they are set, never their values. :func:`assert_ready` raises
:class:`ComponentNotReadyError` when any active component is a fake or cannot
be built.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gluomem.config.app_config import AppConfig
    from gluomem.storage.db import MindDB


class ProviderUnavailableError(RuntimeError):
    """A configured component names a provider or model that cannot be built."""

    def __init__(self, role: str, name: str, reason: str) -> None:
        self.role = role
        self.name = name
        self.reason = reason
        super().__init__(f"{role} {name!r} is unavailable: {reason}")


class FakeComponentError(RuntimeError):
    """A test double was configured without ``components.allow_fake``."""

    def __init__(self, what: str) -> None:
        self.what = what
        super().__init__(
            f"{what} is a test double. Set components.allow_fake=true to use it "
            f"in tests or offline demos; production configs must name a real "
            f"implementation."
        )


class ComponentNotReadyError(RuntimeError):
    """At least one active component is a fake or cannot be built."""

    def __init__(self, blocked: list[ComponentStatus]) -> None:
        self.blocked = blocked
        lines = "; ".join(f"{c.role}: {c.reason}" for c in blocked)
        super().__init__(f"components not ready: {lines}")


def require_fake_allowed(cfg: AppConfig, what: str) -> None:
    """Raise :class:`FakeComponentError` unless ``cfg`` allows test doubles."""
    if not cfg.components.allow_fake:
        raise FakeComponentError(what)


@dataclass(frozen=True)
class ComponentStatus:
    """Effective implementation of one component role. No secrets."""

    role: str
    implementation: str
    provider: str | None = None
    model: str | None = None
    revision: str | None = None
    fake: bool = False
    ready: bool = True
    reason: str = "ok"
    api_key_env: str | None = None
    api_key_set: bool | None = None
    fingerprint: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _embedder_status(cfg: AppConfig) -> ComponentStatus:
    from gluomem.config.registry import ModelRegistry, UnknownModelError
    from gluomem.embedding.fingerprint import fingerprint_of
    from gluomem.embedding.providers import (
        EMBEDDING_PROVIDERS,
        MiniLMEmbedder,
        MockEmbedder,
    )
    from gluomem.extras import is_available

    name = cfg.embedding.model
    dim = int(cfg.storage.vector_dim)
    try:
        entry = ModelRegistry(cfg).resolve(name)
    except UnknownModelError as exc:
        return ComponentStatus("embedder", "none", model=name, ready=False, reason=str(exc))
    base = {"provider": entry.provider, "model": entry.provider_model, "revision": entry.revision}
    if entry.provider == "fake":
        fp = fingerprint_of(
            MockEmbedder(
                dim=dim,
                query_prefix=entry.query_prefix,
                passage_prefix=entry.passage_prefix,
            )
        ).as_dict()
        return ComponentStatus(
            "embedder",
            "gluomem.mock",
            **base,
            fake=True,
            ready=False,
            reason="test double (hash vectors)",
            fingerprint=fp,
        )
    if entry.provider not in EMBEDDING_PROVIDERS:
        return ComponentStatus(
            "embedder",
            "none",
            **base,
            ready=False,
            reason=f"provider {entry.provider!r} has no embedding implementation",
        )
    if entry.dim is None or entry.dim != dim:
        return ComponentStatus(
            "embedder",
            "sentence-transformers",
            **base,
            ready=False,
            reason=f"model dim {entry.dim} != storage.vector_dim {dim}",
        )
    fp = fingerprint_of(
        MiniLMEmbedder(
            entry.provider_model,
            dim=entry.dim,
            revision=entry.revision,
            query_prefix=entry.query_prefix,
            passage_prefix=entry.passage_prefix,
        )
    ).as_dict()
    if not is_available("embeddings"):
        return ComponentStatus(
            "embedder",
            "sentence-transformers",
            **base,
            ready=False,
            reason="'embeddings' extra not installed",
            fingerprint=fp,
        )
    return ComponentStatus("embedder", "sentence-transformers", **base, fingerprint=fp)


def _classifier_status(cfg: AppConfig) -> ComponentStatus:
    from gluomem.extras import is_available
    from gluomem.write.gate import CLASSIFIER_PROVIDERS

    clf = cfg.safety.scanner.classifier
    if not clf.enabled:
        return ComponentStatus("classifier", "disabled")
    entry = cfg.models[clf.model]
    base = {"provider": entry.provider, "model": entry.provider_model, "revision": entry.revision}
    if entry.provider == "fake":
        return ComponentStatus(
            "classifier",
            "gluomem.fake_classifier",
            **base,
            fake=True,
            ready=False,
            reason="test double (heuristic classifier)",
        )
    if entry.provider not in CLASSIFIER_PROVIDERS:
        return ComponentStatus(
            "classifier",
            "none",
            **base,
            ready=False,
            reason=f"provider {entry.provider!r} has no classifier implementation",
        )
    if not is_available("classifier"):
        return ComponentStatus(
            "classifier",
            "gliner2.Classifier",
            **base,
            ready=False,
            reason="'classifier' extra not installed",
        )
    return ComponentStatus("classifier", "gliner2.Classifier", **base)


def _reranker_status(cfg: AppConfig) -> ComponentStatus:
    from gluomem.embedding.reranker import RERANKER_PROVIDERS
    from gluomem.extras import is_available

    if not cfg.search.rerank.enabled:
        return ComponentStatus("reranker", "disabled")
    name = cfg.search.rerank.model
    entry = cfg.models[name]
    base = {"provider": entry.provider, "model": entry.provider_model, "revision": entry.revision}
    if entry.provider not in RERANKER_PROVIDERS:
        return ComponentStatus(
            "reranker",
            "none",
            **base,
            ready=False,
            reason=f"provider {entry.provider!r} has no reranker implementation",
        )
    if not is_available("embeddings"):
        return ComponentStatus(
            "reranker",
            "sentence-transformers.CrossEncoder",
            **base,
            ready=False,
            reason="'embeddings' extra not installed",
        )
    return ComponentStatus("reranker", "sentence-transformers.CrossEncoder", **base)


def _llm_status(
    cfg: AppConfig, role: str, backend: str, model_name: str, env: Mapping[str, str]
) -> ComponentStatus:
    from gluomem.extras import is_available

    if backend == "fake":
        return ComponentStatus(role, "fake", fake=True, ready=False, reason="test double")
    if backend == "none":
        return ComponentStatus(role, "none", reason="disabled (raw turns only)")
    entry = cfg.models[model_name]
    key_env = entry.api_key_env
    key_set = bool(env.get(key_env)) if key_env else None
    status = ComponentStatus(
        role,
        backend,
        provider=entry.provider,
        model=entry.provider_model,
        revision=entry.revision,
        api_key_env=key_env,
        api_key_set=key_set,
    )
    if not is_available("llm"):
        return _blocked(status, "'llm' extra not installed")
    if key_env and not key_set:
        return _blocked(status, f"{key_env} is not set")
    return status


def _blocked(status: ComponentStatus, reason: str) -> ComponentStatus:
    return dataclasses.replace(status, ready=False, reason=reason)


def component_statuses(
    cfg: AppConfig, *, env: Mapping[str, str] | None = None
) -> list[ComponentStatus]:
    """One status per active component role, without building any model."""
    env = os.environ if env is None else env
    statuses = [_embedder_status(cfg), _reranker_status(cfg), _classifier_status(cfg)]
    for task, task_cfg in sorted(cfg.dspy.tasks.items()):
        statuses.append(
            _llm_status(
                cfg,
                f"extraction:{task}",
                cfg.write.extraction_backend,
                task_cfg.model or cfg.dspy.model,
                env,
            )
        )
    return statuses


def component_report(
    cfg: AppConfig,
    *,
    db: MindDB | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """JSON-ready description of the effective components.

    With ``db``, also reports the embedder fingerprint recorded in that file
    and whether the configured embedder matches it.
    """
    statuses = component_statuses(cfg, env=env)
    report: dict[str, Any] = {
        "allow_fake": cfg.components.allow_fake,
        "ready": all(s.ready for s in statuses),
        "components": [s.as_dict() for s in statuses],
    }
    if db is not None:
        from gluomem.embedding.compat import has_vectors, stored_fingerprint

        stored = stored_fingerprint(db)
        configured = statuses[0].fingerprint
        report["store"] = {
            "path": str(db.path),
            "dim": db.dim,
            "has_vectors": has_vectors(db),
            "fingerprint": None if stored is None else stored.as_dict(),
            "embedder_matches": None
            if stored is None or configured is None
            else stored.as_dict() == configured,
        }
    return report


def assert_ready(cfg: AppConfig, *, env: Mapping[str, str] | None = None) -> None:
    """Raise :class:`ComponentNotReadyError` if any active component is blocked."""
    blocked = [s for s in component_statuses(cfg, env=env) if not s.ready]
    if blocked:
        raise ComponentNotReadyError(blocked)
