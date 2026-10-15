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


def _prefix_preprocessing_suffix(query_prefix: str, passage_prefix: str) -> str:
    if not query_prefix and not passage_prefix:
        return ""
    return f";query_prefix={query_prefix!r};passage_prefix={passage_prefix!r}"


class MockEmbedder:
    """Deterministic, offline, hash-based embedder for tests."""

    def __init__(
        self,
        *,
        dim: int = 384,
        normalize: bool = False,
        fallback: bool = False,
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        if dim <= 0:
            raise ValueError(f"MockEmbedder dim must be positive, got {dim}")
        self._dim: int = int(dim)
        self._normalize: bool = bool(normalize)
        self._query_prefix: str = str(query_prefix)
        self._passage_prefix: str = str(passage_prefix)
        if fallback:
            _emit_mock_banner()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def query_prefix(self) -> str:
        return self._query_prefix

    @property
    def passage_prefix(self) -> str:
        return self._passage_prefix

    @property
    def fingerprint(self) -> EmbedderFingerprint:
        suffix = _prefix_preprocessing_suffix(self._query_prefix, self._passage_prefix)
        return EmbedderFingerprint(
            implementation="gluomem.mock",
            model="sha256-pcg64",
            revision=None,
            dim=self._dim,
            preprocessing=f"utf-8;normalize={self._normalize}{suffix}",
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
        """Return a deterministic float32 vector of shape ``(dim,)`` for passage ``text``."""
        return self._vector(f"{self._passage_prefix}{text}" if self._passage_prefix else text)

    def embed_query(self, text: str) -> np.ndarray:
        """Return a deterministic float32 vector of shape ``(dim,)`` for query ``text``."""
        if not self._query_prefix and not self._passage_prefix:
            return self.embed(text)
        return self._vector(f"{self._query_prefix}{text}" if self._query_prefix else text)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Return one deterministic ``(dim,)`` vector per passage text."""
        if self._passage_prefix:
            return [self._vector(f"{self._passage_prefix}{t}") for t in texts]
        return [self._vector(t) for t in texts]


class MiniLMEmbedder:
    """Real ``sentence-transformers`` embedder (native 384 dims, lazy-loaded)."""

    def __init__(
        self,
        model_name: str,
        *,
        dim: int,
        revision: str | None = None,
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        if dim <= 0:
            raise ValueError(f"MiniLMEmbedder dim must be positive, got {dim}")
        self._model_name: str = model_name
        self._dim: int = int(dim)
        self._revision: str | None = revision
        self._query_prefix: str = str(query_prefix)
        self._passage_prefix: str = str(passage_prefix)
        self._model: Any = None

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def query_prefix(self) -> str:
        return self._query_prefix

    @property
    def passage_prefix(self) -> str:
        return self._passage_prefix

    @property
    def fingerprint(self) -> EmbedderFingerprint:
        suffix = _prefix_preprocessing_suffix(self._query_prefix, self._passage_prefix)
        return EmbedderFingerprint(
            implementation="sentence-transformers",
            model=self._model_name,
            revision=self._revision,
            dim=self._dim,
            preprocessing=f"encode;normalize_embeddings=False{suffix}",
        )

    def _load(self) -> Any:
        if self._model is None:
            from gluomem.extras import require

            require("embeddings", f"embedding model '{self._model_name}'")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, revision=self._revision)
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
        """Embed a single passage ``text`` into a float32 ``(dim,)`` vector."""
        prefixed = f"{self._passage_prefix}{text}" if self._passage_prefix else text
        return self._vectors([prefixed])[0]

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query ``text`` into a float32 ``(dim,)`` vector."""
        if not self._query_prefix and not self._passage_prefix:
            return self.embed(text)
        prefixed = f"{self._query_prefix}{text}" if self._query_prefix else text
        return self._vectors([prefixed])[0]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of passage texts into float32 ``(dim,)`` vectors."""
        if self._passage_prefix:
            texts = [f"{self._passage_prefix}{t}" for t in texts]
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
        return MockEmbedder(
            dim=dim,
            fallback=True,
            query_prefix=entry.query_prefix,
            passage_prefix=entry.passage_prefix,
        )

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
    return MiniLMEmbedder(
        entry.provider_model,
        dim=entry.dim,
        revision=entry.revision,
        query_prefix=entry.query_prefix,
        passage_prefix=entry.passage_prefix,
    )
