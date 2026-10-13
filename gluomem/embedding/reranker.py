"""gluomem.embedding.reranker — Reranker protocol + cross-encoder impl.

A :class:`Reranker` protocol re-scores ``(query, docs)`` pairs into logits.
The real impl :class:`CrossEncoderReranker` wraps
``cross-encoder/ms-marco-MiniLM-L-6-v2`` via ``sentence-transformers`` and
lazy-loads the model on first use — merely importing or constructing it does
NOT download anything. It sits behind the protocol so callers stay decoupled
and tests use a stub.

The cross-encoder is OPTIONAL and never loaded by the default test run. Any
test that loads the real model is ``@pytest.mark.slow``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from gluomem.config.app_config import AppConfig


@runtime_checkable
class Reranker(Protocol):
    """A reranker re-scores ``(query, docs)`` into a logit per doc.

    Implementations MUST return one float per doc, ordered to match the input
    ``docs`` list. The real cross-encoder is lazy-loaded and optional; tests
    use a deterministic stub.
    """

    def score(self, query: str, docs: list[str]) -> list[float]:
        """Return a logit per doc, aligned with ``docs`` (higher = more relevant)."""
        ...


class StubReranker:
    """A deterministic, offline reranker for tests.

    Returns a fixed, configurable logit per doc so tests can assert that the
    rerank pool replaces scores and re-orders by the stub logit. No model is
    loaded; no network is used.
    """

    def __init__(self, logit: float = 1.0) -> None:
        self._logit: float = float(logit)

    def score(self, query: str, docs: list[str]) -> list[float]:
        return [self._logit for _ in docs]


class CrossEncoderReranker:
    """A real cross-encoder reranker wrapping ``sentence-transformers``.

    The model is lazy-loaded on first :meth:`score` call so importing or
    constructing this class does NOT download anything. The provider model id
    comes from config (the registry's ``reranker`` entry →
    ``cross-encoder/ms-marco-MiniLM-L-6-v2``); no literal model id is hardcoded
    in callers.

    This class is OPTIONAL and never loaded by the default test run. Any test
    exercising it is ``@pytest.mark.slow``.
    """

    def __init__(self, model_name: str, *, revision: str | None = None) -> None:
        self._model_name: str = model_name
        self._revision: str | None = revision
        self._model: Any = None  # CrossEncoder, typed loose to avoid an import

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


#: Reranker providers with an implementation in this package.
RERANKER_PROVIDERS: tuple[str, ...] = ("inprocess",)


def make_reranker(cfg: AppConfig) -> Reranker | None:
    """Config-resolved reranker; ``None`` when reranking is disabled.

    The registry entry's ``provider`` selects the implementation; only
    ``inprocess`` (a ``sentence-transformers`` cross-encoder) exists, and any
    other provider raises
    :class:`~gluomem.components.ProviderUnavailableError` instead of silently
    running a different implementation. A missing ``embeddings`` extra raises
    :class:`~gluomem.extras.MissingExtraError`. Construction does not download
    the model.
    """
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
