"""gluomem.bench.retriever — GluoMemRetriever: the system under test.

Wraps ``HybridSearch`` over per-conversation ``.mind`` files
(``<db_dir>/{conv_id}.mind``, produced by ``gluomem bench ingest``): LoCoMo QA
is per-conversation and HybridSearch has no conversation scoping, so one DB
per conversation keeps retrieval uncontaminated and matches the "one user's
memory" semantics of the published numbers.

Context lines are date-prefixed (``[event_time or created_at] content``) —
deliberate prompt signal for the temporal question category.

Abstention: when the search abstains (or returns nothing) the retriever
answers ``benchmark.answer.abstain_text`` WITHOUT an LLM call; usable LoCoMo
categories carry no abstention gold, so the simple policy is the honest one.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from gluomem.bench.types import Prediction, RetrievedDoc, Sample
from gluomem.embedding.providers import make_embedder
from gluomem.embedding.reranker import make_reranker
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.search.hybrid import HybridSearch
from gluomem.storage.db import MindDB

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gluomem.bench.types import AnswerFn
    from gluomem.config.app_config import AppConfig
    from gluomem.embedding.base import Embedder
    from gluomem.embedding.reranker import Reranker
    from gluomem.frames.store import Frame


def _is_raw_turn(frame: Frame) -> bool:
    return ":session:" in frame.gop_id and frame.source == "import"


class GluoMemRetriever:
    """``Retriever`` protocol impl over per-conversation memory DBs."""

    name = "gluomem"

    def __init__(
        self,
        cfg: AppConfig,
        *,
        db_dir: Path,
        answer_fn: AnswerFn,
        content_to_dias: Mapping[str, tuple[str, ...]],
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._cfg = cfg
        self._db_dir = Path(db_dir)
        self._answer_fn = answer_fn
        self._content_to_dias = content_to_dias
        self._embedder = embedder
        self._reranker = reranker
        # ``None`` is a legitimate *resolved* reranker (rerank disabled), so a
        # plain None-check can't tell "not yet resolved" from "resolved to off".
        # An injected reranker (e.g. a test stub) is treated as already resolved
        # and wins over config.
        self._reranker_resolved = reranker is not None
        self._stacks: dict[str, HybridSearch] = {}
        # The runner fans answer() out to threads; SQLite access (stack
        # creation + search) is serialized here so only the LLM call overlaps.
        self._lock = threading.Lock()

    def _resolve_embedder(self, db: MindDB) -> Embedder:
        """The query-side embedder MUST match the one that embedded the
        frames at ingest — else vector similarities are noise and hybrid
        search silently degrades to keyword-only. Resolve from config
        (real MiniLM by default), built once and shared across stacks. The
        mock is only built when config names it and allows fakes; each
        store also checks the embedder fingerprint recorded at ingest.
        """
        if self._embedder is None:
            self._embedder = make_embedder(self._cfg)
        return self._embedder

    def _resolve_reranker(self) -> Reranker | None:
        """Resolve the cross-encoder reranker from config, mirroring the
        embedder. ``HybridSearch`` only reranks when ``search.rerank.enabled``
        AND a reranker is present; without this the flag is inert and rerank
        silently no-ops. Built once (the model lazy-loads on first score) and
        shared across every per-conversation stack. ``make_reranker`` returns
        ``None`` when rerank is disabled, so that is a valid terminal state.
        """
        if not self._reranker_resolved:
            self._reranker = make_reranker(self._cfg)
            self._reranker_resolved = True
        return self._reranker

    def _search_for(self, conv_id: str) -> HybridSearch:
        stack = self._stacks.get(conv_id)
        if stack is None:
            db_path = self._db_dir / f"{conv_id}.mind"
            if not db_path.is_file():
                raise FileNotFoundError(
                    f"no memory DB for conversation '{conv_id}' at {db_path} "
                    f"— run `gluomem bench ingest --dataset locomo --db-dir "
                    f"{self._db_dir}` first"
                )
            db = MindDB(db_path, dim=self._cfg.storage.vector_dim)
            embedder = self._resolve_embedder(db)
            store = FrameStore(
                db, embedder=embedder, chunking=self._cfg.chunking
            )
            kg = KnowledgeGraph(db)
            stack = HybridSearch(
                db,
                store,
                self._cfg,
                embedder=embedder,
                reranker=self._resolve_reranker(),
                kg=kg,
            )
            self._stacks[conv_id] = stack
        return stack

    def answer(self, s: Sample) -> Prediction:
        start = time.perf_counter()
        with self._lock:
            result = self._search_for(s.conv_id).search(
                s.question, limit=self._cfg.benchmark.answer.context_k
            )
        if result.abstained or not result.hits:
            return Prediction(
                sample_id=s.id,
                answer=self._cfg.benchmark.answer.abstain_text,
                abstained=True,
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        evidence = set(s.evidence)
        docs: list[RetrievedDoc] = []
        lines: list[str] = []
        for hit in result.hits:
            frame = hit.frame
            dia_id: str | None = None
            if _is_raw_turn(frame):
                dias = self._content_to_dias.get(frame.content)
                if dias:
                    # Repeated statements share content; when the retrieved
                    # text IS an evidence turn's text, credit that dia_id
                    # (same information, deterministic tie-break).
                    dia_id = next((d for d in dias if d in evidence), dias[0])
            docs.append(
                RetrievedDoc(
                    rank=hit.rank,
                    text=frame.content,
                    score=hit.final,
                    dia_id=dia_id,
                    frame_id=frame.id,
                )
            )
            lines.append(f"[{frame.event_time or frame.created_at}] {frame.content}")
        out = self._answer_fn(s.question, "\n".join(lines))
        return Prediction(
            sample_id=s.id,
            answer=out.text,
            retrieved=tuple(docs),
            prompt_tokens=out.prompt_tokens,
            completion_tokens=out.completion_tokens,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )

    def manifest_extra(self) -> dict[str, object]:
        """Effective retrieval provenance for the run manifest (the runner wires
        this into ``retriever_info``). Records what ACTUALLY ran, not just what
        config asked for: ``rerank.active`` is whether a reranker was resolved
        (non-None), the guard against a silent regression where config claims
        ``enabled`` but no reranker is wired.
        """
        rerank = self._cfg.search.rerank
        return {
            "embedder": self._cfg.embedding.model,
            "rerank": {
                "enabled": rerank.enabled,
                "model": rerank.model,
                "pool": rerank.pool,
                "active": self._resolve_reranker() is not None,
            },
            "abstain": {"enabled": self._cfg.search.abstain.enabled},
        }


__all__ = ["GluoMemRetriever"]
