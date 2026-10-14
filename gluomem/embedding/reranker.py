"""Reranker protocol and cross-encoder implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from gluomem.config.app_config import AppConfig


@runtime_checkable
class Reranker(Protocol):
    """Re-score ``(query, docs)`` pairs into a logit per document."""

    def score(self, query: str, docs: list[str]) -> list[float]:
        ...


class StubReranker:
    """Deterministic offline reranker for tests."""

    def __init__(self, logit: float = 1.0) -> None:
        self._logit: float = float(logit)

    def score(self, query: str, docs: list[str]) -> list[float]:  # noqa: ARG002
        return [self._logit for _ in docs]


class CrossEncoderReranker:
    """Lazy-loaded ``sentence-transformers`` cross-encoder reranker."""

    def __init__(self, model_name: str, *, revision: str | None = None) -> None:
        self._model_name: str = model_name
        self._revision: str | None = revision
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self) -> Any:
        if self._model is None:
            from gluomem.extras import require

            require("embeddings", f"reranker model '{self._model_name}'")
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name, revision=self._revision)
        return self._model

    def score(self, query: str, docs: list[str]) -> list[float]:
        model = self._load()
        pairs = [(query, d) for d in docs]
        logits = model.predict(pairs)
        return [float(x) for x in logits]


RERANKER_PROVIDERS: tuple[str, ...] = ("inprocess",)


def make_reranker(cfg: AppConfig) -> Reranker | None:
    """Build the configured reranker, or return ``None`` when reranking is disabled."""
    if not cfg.search.rerank.enabled:
        return None
    from gluomem.components import ProviderUnavailableError
    from gluomem.extras import require

    name = cfg.search.rerank.model
    entry = cfg.models[name]
    if entry.provider not in RERANKER_PROVIDERS:
        raise ProviderUnavailableError(
            "reranker",
            name,
            f"provider {entry.provider!r} has no reranker implementation; "
            f"supported: {', '.join(RERANKER_PROVIDERS)}",
        )
    require("embeddings", f"reranker model '{entry.provider_model}'")
    return CrossEncoderReranker(entry.provider_model, revision=entry.revision)
