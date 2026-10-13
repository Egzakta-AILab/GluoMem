"""gluomem.embedding.providers — embedder implementations.

Two embedders plus a config-driven factory:

- **MockEmbedder** — a deterministic, offline, hash-based embedder (native dim,
  no network). When selected as a FALLBACK via :func:`make_embedder` it emits a
  LOUD all-caps ``MOCK`` banner to stderr AND ``logging.warning`` so a
  misconfigured or unavailable model is never silent. Constructed directly it
  stays quiet (tests use it that way).
- **MiniLMEmbedder** — the real ``sentence-transformers/all-MiniLM-L6-v2``
  wrapper (native dim 384, NO zero-pad). The model is lazy-loaded on first
  ``embed``/``embed_batch`` call so merely importing or constructing this class
  does NOT download anything.
- **make_embedder(cfg)** — a factory that resolves ``cfg.embedding.model`` via
  the registry and returns the right :class:`Embedder`. When the configured
  model is ``'mock'`` or unavailable (e.g. ``sentence-transformers`` is not
  importable) it falls back to :class:`MockEmbedder` and prints the banner.

The mock derives a stable native-dim float32 vector from a hash of the text:
the SHA-256 digest seeds a numpy RNG (default_rng, PCG64) which draws ``dim``
floats in [0, 1), then optionally L2-normalizes. The same text always produces
a byte-identical array; no socket is opened.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from typing import Any

import numpy as np

from gluomem.config.app_config import AppConfig
from gluomem.config.registry import ModelRegistry
from gluomem.embedding.base import Embedder

_log = logging.getLogger("gluomem.embedding")

# The LOUD all-caps banner emitted when the mock is selected as a FALLBACK.
# Written to stderr AND logging.warning so a misconfigured model is never silent.
_MOCK_BANNER = (
    "MOCK EMBEDDER FALLBACK IN USE — NO REAL MODEL LOADED. "
    "Vectors are deterministic hash-based placeholders, not MiniLM embeddings. "
    "Set embedding.model to a real, available model for production embeddings."
)


def _emit_mock_banner() -> None:
    """Emit the LOUD all-caps MOCK banner to stderr and logging.warning."""
    print(_MOCK_BANNER, file=sys.stderr, flush=True)
    _log.warning(_MOCK_BANNER)


class MockEmbedder:
    """A deterministic, offline, hash-based embedder for tests / fallback.

    Parameters
    ----------
    dim:
        The native embedding dimension. Comes from config (default 384 via
        ``AppConfig.storage.vector_dim``); never hardcoded in callers.
    normalize:
        If True, L2-normalize the output vector. Default False keeps the raw
        hash-derived floats (determinism is unaffected — normalization is a
        pure function of the same seed).
    fallback:
        If True, emit the LOUD all-caps ``MOCK`` banner on construction — used
        by :func:`make_embedder` when the mock is selected because the
        configured model is unavailable or ``embedding.model == 'mock'``.
        Default False keeps direct construction quiet (for tests).

    Determinism
    -----------
    The SHA-256 digest of the UTF-8 text seeds ``numpy.default_rng`` (PCG64),
    which is stable across processes for a fixed numpy/Python version. Drawing
    ``dim`` floats from that seed yields a byte-identical vector for identical
    text. No network I/O occurs.
    """

    def __init__(
        self, *, dim: int, normalize: bool = False, fallback: bool = False
    ) -> None:
        if dim <= 0:
            raise ValueError(f"MockEmbedder dim must be positive, got {dim}")
        self._dim: int = int(dim)
        self._normalize: bool = bool(normalize)
        if fallback:
            _emit_mock_banner()

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # 32-byte digest → seed; numpy accepts up to 624 bytes/uint32.
        seed_arr = np.frombuffer(digest, dtype=np.uint32)
        rng = np.random.default_rng(seed_arr)
        vec = rng.random(self._dim, dtype=np.float32)
        if self._normalize:
            norm = float(np.linalg.norm(vec))
            if norm > 0.0:
                vec = vec / norm
        return np.ascontiguousarray(vec, dtype=np.float32)

    def embed(self, text: str) -> np.ndarray:
        """Return a deterministic float32 vector of shape ``(dim,)`` for ``text``."""
        return self._vector(text)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Return one deterministic ``(dim,)`` vector per text (same as embed)."""
        return [self._vector(t) for t in texts]


class MiniLMEmbedder:
    """The real ``sentence-transformers/all-MiniLM-L6-v2`` embedder.

    Native dim 384, NO zero-pad. The model is lazy-loaded on first
    :meth:`embed` / :meth:`embed_batch` call — merely importing or constructing
    this class does NOT download anything.

    Parameters
    ----------
    model_name:
        The sentence-transformers model id (from the registry's
        ``provider_model``). Not hardcoded in callers.
    dim:
        The native embedding dimension (from the registry / config). Vectors
        are emitted at this dim with no padding.

    This class is OPTIONAL and never loaded by the default test run. Any test
    exercising it is ``@pytest.mark.slow``.
    """

    def __init__(self, model_name: str, *, dim: int) -> None:
        if dim <= 0:
            raise ValueError(f"MiniLMEmbedder dim must be positive, got {dim}")
        self._model_name: str = model_name
        self._dim: int = int(dim)
        self._model: Any = None  # SentenceTransformer, typed loose to avoid an import

    @property
    def dim(self) -> int:
        return self._dim

    def _load(self) -> Any:
        if self._model is None:
            from gluomem.extras import require

            require("embeddings", f"embedding model '{self._model_name}'")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _vectors(self, texts: list[str]) -> list[np.ndarray]:
        model = self._load()
        # Native dim, no pad: sentence-transformers emits at the model's native
        # dim. We assert it matches the configured dim so a mismatch fails loud
        # rather than silently zero-padding.
        mat = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        out: list[np.ndarray] = []
        for row in mat:
            vec = np.ascontiguousarray(row, dtype=np.float32)
            if vec.shape != (self._dim,):
                raise ValueError(
                    f"MiniLMEmbedder: model produced a {vec.shape} vector but "
                    f"native dim is {self._dim}; no zero-pad is performed."
                )
            out.append(vec)
        return out

    def embed(self, text: str) -> np.ndarray:
        """Embed a single ``text`` → float32 ``(dim,)`` vector (lazy-loads the model)."""
        return self._vectors([text])[0]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch → one float32 ``(dim,)`` vector per text (lazy-loads once)."""
        return self._vectors(texts)


def make_embedder(cfg: AppConfig) -> Embedder:
    """Build the configured embedder, falling back to the mock with a loud banner.

    Resolves ``cfg.embedding.model`` via the :class:`ModelRegistry`. When the
    configured model is ``'mock'`` or unavailable (e.g.
    ``sentence-transformers`` is not importable, or the model has no native
    dim), it returns a :class:`MockEmbedder` with ``fallback=True`` (which
    emits the LOUD all-caps ``MOCK`` banner to stderr and ``logging.warning``).

    The native dim is read from ``cfg.storage.vector_dim`` (cross-validated to
    equal the active embedder dim) — never hardcoded.
    """
    dim = int(cfg.storage.vector_dim)
    name = cfg.embedding.model

    # Explicit mock selection → mock with the loud fallback banner.
    if name == "mock":
        return MockEmbedder(dim=dim, fallback=True)

    registry = ModelRegistry(cfg)
    try:
        entry = registry.resolve(name)
    except KeyError:
        # Unknown model reference → fall back loud rather than crash at import.
        _emit_mock_banner()
        return MockEmbedder(dim=dim, fallback=False)

    # A model without a native dim cannot embed → fall back loud.
    if entry.dim is None:
        _emit_mock_banner()
        return MockEmbedder(dim=dim, fallback=False)

    # Try to build the real embedder. If sentence-transformers is not
    # importable (or construction raises), fall back loud. Construction itself
    # does NOT download — the model lazy-loads on first embed.
    try:
        emb: Embedder
        if entry.provider == "inprocess":
            emb = MiniLMEmbedder(entry.provider_model, dim=entry.dim)
            if emb.dim != dim:
                _emit_mock_banner()
                return MockEmbedder(dim=dim, fallback=False)
            return emb
        # Non-inprocess embedding providers are unsupported → mock fallback.
        _emit_mock_banner()
        return MockEmbedder(dim=dim, fallback=False)
    except Exception:  # noqa: BLE001 — any construction/import failure → fallback
        _emit_mock_banner()
        return MockEmbedder(dim=dim, fallback=False)
