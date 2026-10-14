"""gluomem.bench.baselines — the literature baselines, same harness.

All three consume raw conversation turns straight from the dataset JSON
(``adapter.conversations()``) — no extraction, no admission gate, no scoring
signals. That independence from the gluomem write path is what makes them
baselines; they share only the ``Retriever`` protocol, the answerer and the
judges, so every number comes from the identical pipeline.

- FullContextRetriever — the whole transcript in the prompt (upper-bound
  reference at small scale; LoCoMo fits default limits).
- Bm25Retriever — keyword-only: per-conversation in-memory FTS5 over turns.
- VectorRagRetriever — naive dense retrieval: embed turns, cosine top-k;
  no fusion, no KG, no temporal, no rerank BY DESIGN.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from gluomem.bench.types import Prediction, RetrievedDoc, Sample
from gluomem.search.sanitize import sanitize

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gluomem.bench.types import AnswerFn
    from gluomem.config.app_config import AppConfig
    from gluomem.embedding.base import Embedder
    from gluomem.write.ingest import Conversation


def transcript(conv: Conversation) -> str:
    """Whole-conversation transcript with session date headers."""
    blocks: list[str] = []
    for session in conv.sessions:
        lines = [f"Session {session.index} ({session.date_time.isoformat()}):"]
        lines.extend(f"{t.speaker}: {t.text}" for t in session.turns)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _turn_docs(conv: Conversation) -> list[tuple[str, str, str]]:
    """(dia_id, date_iso, content) per turn; content matches ingest's format."""
    docs: list[tuple[str, str, str]] = []
    for session in conv.sessions:
        date_iso = session.date_time.isoformat()
        for turn in session.turns:
            if turn.dia_id is None:
                continue
            docs.append((turn.dia_id, date_iso, f"{turn.speaker}: {turn.text}"))
    return docs


class FullContextRetriever:
    """No memory system at all: the (truncated) transcript is the context."""

    name = "full_context"

    def __init__(
        self,
        cfg: AppConfig,
        conversations: Sequence[Conversation],
        answer_fn: AnswerFn,
    ) -> None:
        self._cfg = cfg
        self._answer_fn = answer_fn
        self._transcripts = {c.conv_id: transcript(c) for c in conversations}

    def answer(self, s: Sample) -> Prediction:
        start = time.perf_counter()
        text = self._transcripts[s.conv_id]
        limit = self._cfg.benchmark.full_context.max_chars
        if len(text) > limit:
            keep = self._cfg.benchmark.full_context.keep
            text = text[-limit:] if keep == "tail" else text[:limit]
        out = self._answer_fn(s.question, text)
        return Prediction(
            sample_id=s.id,
            answer=out.text,
            prompt_tokens=out.prompt_tokens,
            completion_tokens=out.completion_tokens,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )


class Bm25Retriever:
    """Keyword-only baseline: sqlite FTS5 (stdlib) over raw turns."""

    name = "bm25"

    def __init__(
        self,
        cfg: AppConfig,
        conversations: Sequence[Conversation],
        answer_fn: AnswerFn,
    ) -> None:
        self._cfg = cfg
        self._answer_fn = answer_fn
        self._conns: dict[str, sqlite3.Connection] = {}
        self._docs: dict[str, list[tuple[str, str, str]]] = {}
        # The runner fans answer() out to threads; FTS queries are serialized
        # (µs each) so only the LLM call overlaps.
        self._lock = threading.Lock()
        for conv in conversations:
            docs = _turn_docs(conv)
            conn = sqlite3.connect(":memory:", check_same_thread=False)
            conn.execute("CREATE VIRTUAL TABLE docs USING fts5(content)")
            conn.executemany(
                "INSERT INTO docs(rowid, content) VALUES (?, ?)",
                [(i + 1, content) for i, (_, _, content) in enumerate(docs)],
            )
            conn.commit()
            self._conns[conv.conv_id] = conn
            self._docs[conv.conv_id] = docs

    def answer(self, s: Sample) -> Prediction:
        start = time.perf_counter()
        docs = self._docs[s.conv_id]
        # sanitize() phrase-quotes each token (FTS5 syntax neutralized); the
        # space join it returns is implicit AND — too strict for a keyword
        # baseline, so re-join the phrases with OR (classic BM25 semantics).
        query = " OR ".join(sanitize(s.question).split())
        rows: list[tuple[int, float]] = []
        if query:  # empty sanitize output = no usable tokens; retrieve nothing
            with self._lock:
                rows = self._conns[s.conv_id].execute(
                    "SELECT rowid, bm25(docs) FROM docs WHERE docs MATCH ? "
                    "ORDER BY bm25(docs) LIMIT ?",
                    (query, self._cfg.benchmark.answer.context_k),
                ).fetchall()
        retrieved: list[RetrievedDoc] = []
        lines: list[str] = []
        for rank, (rowid, bm25_score) in enumerate(rows):
            dia_id, date_iso, content = docs[rowid - 1]
            # bm25() returns lower-is-better; negate so higher = more relevant.
            retrieved.append(
                RetrievedDoc(
                    rank=rank, text=content, score=-bm25_score, dia_id=dia_id
                )
            )
            lines.append(f"[{date_iso}] {content}")
        if not retrieved:
            return Prediction(
                sample_id=s.id,
                answer=self._cfg.benchmark.answer.abstain_text,
                abstained=True,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                abstention={"reason": "no_hits", "candidates": 0},
            )
        out = self._answer_fn(s.question, "\n".join(lines))
        return Prediction(
            sample_id=s.id,
            answer=out.text,
            retrieved=tuple(retrieved),
            prompt_tokens=out.prompt_tokens,
            completion_tokens=out.completion_tokens,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )


class VectorRagRetriever:
    """Naive dense-retrieval baseline: embed turns once, cosine top-k."""

    name = "vector_rag"

    def __init__(
        self,
        cfg: AppConfig,
        conversations: Sequence[Conversation],
        answer_fn: AnswerFn,
        *,
        embedder: Embedder | None = None,
    ) -> None:
        if embedder is None:
            from gluomem.embedding.providers import make_embedder

            embedder = make_embedder(cfg)
        self._cfg = cfg
        self._answer_fn = answer_fn
        self._embedder = embedder
        # Query embedding under the runner's thread fan-out is serialized
        # (real encoders aren't guaranteed thread-safe); LLM calls overlap.
        self._lock = threading.Lock()
        self._docs: dict[str, list[tuple[str, str, str]]] = {}
        self._matrices: dict[str, np.ndarray] = {}
        for conv in conversations:
            docs = _turn_docs(conv)
            vectors = embedder.embed_batch([content for _, _, content in docs])
            matrix = np.stack(vectors).astype(np.float64)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            self._docs[conv.conv_id] = docs
            self._matrices[conv.conv_id] = matrix / norms

    def answer(self, s: Sample) -> Prediction:
        start = time.perf_counter()
        docs = self._docs[s.conv_id]
        matrix = self._matrices[s.conv_id]
        with self._lock:
            q = np.asarray(self._embedder.embed(s.question), dtype=np.float64)
        q_norm = np.linalg.norm(q)
        if q_norm > 0.0:
            q = q / q_norm
        sims = matrix @ q
        k = min(self._cfg.benchmark.answer.context_k, len(docs))
        top = np.argsort(-sims)[:k]
        retrieved: list[RetrievedDoc] = []
        lines: list[str] = []
        for rank, idx in enumerate(top):
            dia_id, date_iso, content = docs[int(idx)]
            retrieved.append(
                RetrievedDoc(
                    rank=rank, text=content, score=float(sims[idx]), dia_id=dia_id
                )
            )
            lines.append(f"[{date_iso}] {content}")
        out = self._answer_fn(s.question, "\n".join(lines))
        return Prediction(
            sample_id=s.id,
            answer=out.text,
            retrieved=tuple(retrieved),
            prompt_tokens=out.prompt_tokens,
            completion_tokens=out.completion_tokens,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )


__all__ = [
    "Bm25Retriever",
    "FullContextRetriever",
    "VectorRagRetriever",
    "transcript",
]
