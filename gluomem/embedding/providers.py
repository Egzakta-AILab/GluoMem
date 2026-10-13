"""gluomem.embedding.providers — embedder implementations.

Two embedders plus a config-driven factory:

- **MockEmbedder** — a deterministic, offline, hash-based embedder (native dim,
  no network). It is a test double: :func:`make_embedder` builds it only for
  a registry entry with ``provider: fake`` and only when
  ``components.allow_fake`` is set, and prints a loud ``MOCK`` banner when it
  does. Constructed directly it stays quiet (tests use it that way).
- **MiniLMEmbedder** — a ``sentence-transformers`` wrapper (``provider:
  inprocess``; native dim, NO zero-pad). The model is lazy-loaded on first
  ``embed``/``embed_batch`` call so merely importing or constructing this class
  does NOT download anything.
- **make_embedder(cfg)** — resolves ``cfg.embedding.model`` via the registry
  and builds the implementation its ``provider`` names. An unknown model, an
  unsupported provider, a missing dim, a dim mismatch, a missing extra or a
  disallowed fake raises; nothing falls back to the mock.

Each embedder declares an
:class:`~gluomem.embedding.fingerprint.EmbedderFingerprint` used to keep a
``*.mind`` file on one encoder.

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

from gluomem.components import ProviderUnavailableError, require_fake_allowed
from gluomem.config.app_config import AppConfig
from gluomem.config.registry import ModelRegistry, UnknownModelError
from gluomem.embedding.base import Embedder
from gluomem.embedding.fingerprint import EmbedderFingerprint

#: Embedding providers with an implementation in this package.
EMBEDDING_PROVIDERS: tuple[str, ...] = ("fake", "inprocess")

_log = logging.getLogger("gluomem.embedding")

# Written to stderr AND logging.warning whenever the factory builds the mock,
# so a run on placeholder vectors is never silent.
_MOCK_BANNER = (
    "MOCK EMBEDDER IN USE — NO REAL MODEL LOADED. "
    "Vectors are deterministic hash-based placeholders for tests. "
    "Set embedding.model to a real model for production embeddings."
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
        by :func:`make_embedder`. Default False keeps direct construction
        quiet (for tests).

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

    @property
    def fingerprint(self) -> EmbedderFingerprint:
        return EmbedderFingerprint(
            implementation="gluomem.mock",
            model="sha256-pcg64",
            revision=None,
            dim=self._dim,
            preprocessing=f"utf-8;normalize={self._normalize}",
        )

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
    revision:
        Optional pinned model revision passed to ``SentenceTransformer``.

    The tokenizer and pooling are part of the model at ``revision``; the
    encode options GluoMem passes are recorded as ``preprocessing`` in the
    fingerprint.

    This class is OPTIONAL and never loaded by the default test run. Any test
    exercising it is ``@pytest.mark.slow``.
    """

    def __init__(
        self, model_name: str, *, dim: int, revision: str | None = None
    ) -> None:
        if dim <= 0:
            raise ValueError(f"MiniLMEmbedder dim must be positive, got {dim}")
        self._model_name: str = model_name
        self._dim: int = int(dim)
        self._revision: str | None = revision
        self._model: Any = None  # SentenceTransformer, typed loose to avoid an import

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> EmbedderFingerprint:
        return EmbedderFingerprint(
            implementation="sentence-transformers",
            model=self._model_name,
            revision=self._revision,
            dim=self._dim,
            preprocessing="encode;normalize_embeddings=False",
        )

    def _load(self) -> Any:
        if self._model is None:
            from gluomem.extras import require

            require("embeddings", f"embedding model '{self._model_name}'")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_name, revision=self._revision
            )
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
    """Build the embedder ``cfg.embedding.model`` names, or raise.

    Raises :class:`~gluomem.components.ProviderUnavailableError` for an
    unknown model, a provider with no embedding implementation, a model with
    no dim or a dim that differs from ``cfg.storage.vector_dim``;
    :class:`~gluomem.extras.MissingExtraError` when ``sentence-transformers``
    is not installed; :class:`~gluomem.components.FakeComponentError` for a
    fake provider without ``components.allow_fake``. Construction does not
    download anything (the real model lazy-loads on first embed).
    """
    dim = int(cfg.storage.vector_dim)
    name = cfg.embedding.model
    try:
        entry = ModelRegistry(cfg).resolve(name)
    except UnknownModelError as exc:
        raise ProviderUnavailableError("embedder", name, str(exc)) from None

    if entry.provider == "fake":
        require_fake_allowed(cfg, f"embedding.model={name!r} (provider: fake)")
        return MockEmbedder(dim=dim, fallback=True)

    if entry.provider != "inprocess":
        raise ProviderUnavailableError(
            "embedder",
            name,
            f"provider {entry.provider!r} has no embedding implementation; "
            f"supported: {', '.join(EMBEDDING_PROVIDERS)}",
        )
    if entry.dim is None:
        raise ProviderUnavailableError("embedder", name, "registry entry has no dim")
    if entry.dim != dim:
        raise ProviderUnavailableError(
            "embedder",
            name,
            f"model dim {entry.dim} != storage.vector_dim {dim}",
        )
    from gluomem.extras import require

    require("embeddings", f"embedding model '{entry.provider_model}'")
    return MiniLMEmbedder(entry.provider_model, dim=entry.dim, revision=entry.revision)
