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

    def __init__(self, model_name: str) -> None:
        self._model_name: str = model_name
        self._model: Any = None  # CrossEncoder, typed loose to avoid an import

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    def score(self, query: str, docs: list[str]) -> list[float]:
        model = self._load()
        pairs = [(query, d) for d in docs]
        logits = model.predict(pairs)
        return [float(x) for x in logits]


def make_reranker(cfg: AppConfig) -> Reranker | None:
    """Config-resolved reranker; ``None`` when reranking is disabled.

    Construction is cheap (the cross-encoder lazy-loads on first score), so
    this is safe to call on offline paths that never actually rerank.
    """
    if not cfg.search.rerank.enabled:
        return None
    entry = cfg.models[cfg.search.rerank.model]
    return CrossEncoderReranker(entry.provider_model)
