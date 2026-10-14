"""Embedding providers: ``MockEmbedder``, ``MiniLMEmbedder``, and ``make_embedder``."""

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

EMBEDDING_PROVIDERS: tuple[str, ...] = ("fake", "inprocess")

_log = logging.getLogger("gluomem.embedding")

_MOCK_BANNER = (
    "MOCK EMBEDDER IN USE — NO REAL MODEL LOADED. "
    "Vectors are deterministic hash-based placeholders for tests. "
    "Set embedding.model to a real model for production embeddings."
)


def _emit_mock_banner() -> None:
    """Emit the loud all-caps MOCK banner to stderr and logging.warning."""
    print(_MOCK_BANNER, file=sys.stderr, flush=True)
    _log.warning(_MOCK_BANNER)


class MockEmbedder:
    """Deterministic, offline, hash-based embedder for tests."""

    def __init__(
        self, *, dim: int = 384, normalize: bool = False, fallback: bool = False
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
        """Return one deterministic ``(dim,)`` vector per text."""
        return [self._vector(t) for t in texts]


class MiniLMEmbedder:
    """Real ``sentence-transformers`` embedder (native 384 dims, lazy-loaded)."""

    def __init__(
        self, model_name: str, *, dim: int, revision: str | None = None
    ) -> None:
        if dim <= 0:
            raise ValueError(f"MiniLMEmbedder dim must be positive, got {dim}")
        self._model_name: str = model_name
        self._dim: int = int(dim)
        self._revision: str | None = revision
        self._model: Any = None

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
        """Embed a single ``text`` into a float32 ``(dim,)`` vector."""
        return self._vectors([text])[0]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of texts into float32 ``(dim,)`` vectors."""
        return self._vectors(texts)


def make_embedder(cfg: AppConfig) -> Embedder:
    """Build the embedder ``cfg.embedding.model`` names, or raise."""
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
