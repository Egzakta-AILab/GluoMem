"""Baselines — bm25/vector-RAG/full-context under the same Retriever protocol."""

from __future__ import annotations

from gluomem.bench.baselines import (
    Bm25Retriever,
    FullContextRetriever,
    VectorRagRetriever,
    transcript,
)
from gluomem.bench.types import Sample
from gluomem.embedding.providers import MockEmbedder


def _sample(question: str, evidence: tuple[str, ...] = ()) -> Sample:
    return Sample(
        id="conv-tiny-1:qa:0",
        conv_id="conv-tiny-1",
        question=question,
        gold="g",
        category=4,
        category_name="single-hop",
        evidence=evidence,
    )


def test_bm25_finds_evidence_turn(bench_cfg, fixture_adapter, answer_fn) -> None:
    r = Bm25Retriever(bench_cfg, fixture_adapter.conversations(), answer_fn)
    p = r.answer(_sample("Who moved to Seattle?", evidence=("D3:1",)))
    assert not p.abstained
    assert "D3:1" in {d.dia_id for d in p.retrieved}
    # Higher score = more relevant (bm25 sign flipped); order respects rank.
    assert [d.rank for d in p.retrieved] == list(range(len(p.retrieved)))
    (_, context), = answer_fn.calls
    assert context.splitlines()[0].startswith("[")  # date-prefixed like gluomem


def test_bm25_survives_fts5_hostile_query(bench_cfg, fixture_adapter, answer_fn) -> None:
    r = Bm25Retriever(bench_cfg, fixture_adapter.conversations(), answer_fn)
    # Unsanitized, this is FTS5 syntax soup; sanitize() phrase-quotes it.
    p = r.answer(_sample('cat AND (job* OR "promo:tion) - NEAR'))
    assert p.answer  # no sqlite OperationalError, some answer produced


def test_bm25_no_usable_tokens_abstains(bench_cfg, fixture_adapter, answer_fn) -> None:
    r = Bm25Retriever(bench_cfg, fixture_adapter.conversations(), answer_fn)
    p = r.answer(_sample("*** :::"))
    assert p.abstained is True
    assert answer_fn.calls == []


def test_vector_rag_returns_k_docs_with_dias(
    bench_cfg, fixture_adapter, answer_fn
) -> None:
    r = VectorRagRetriever(
        bench_cfg,
        fixture_adapter.conversations(),
        answer_fn,
        embedder=MockEmbedder(dim=bench_cfg.storage.vector_dim),
    )
    p = r.answer(_sample("Where does Bob live?"))
    # 9 turns in the fixture; context_k=10 caps at corpus size.
    assert len(p.retrieved) == 9
    assert all(d.dia_id is not None for d in p.retrieved)
    # Scores are descending cosine similarities.
    scores = [d.score for d in p.retrieved]
    assert scores == sorted(scores, reverse=True)


def test_full_context_carries_all_sessions(
    bench_cfg, fixture_adapter, answer_fn
) -> None:
    r = FullContextRetriever(bench_cfg, fixture_adapter.conversations(), answer_fn)
    p = r.answer(_sample("What happened?"))
    assert p.retrieved == ()  # no retrieval diagnostics for full-context
    (_, context), = answer_fn.calls
    for date in ("2023-05-08", "2023-06-21", "2023-08-02"):
        assert date in context
    assert "Session 1" in context and "Session 3" in context


def test_full_context_truncation_head_vs_tail(
    bench_cfg, fixture_adapter, answer_fn, recording_answer_fn_cls
) -> None:
    convs = fixture_adapter.conversations()
    full = transcript(convs[0])
    limit = len(full) // 2

    tail_cfg = bench_cfg.with_overrides(
        {"benchmark.full_context.max_chars": limit}
    )
    tail_fn = recording_answer_fn_cls()
    FullContextRetriever(tail_cfg, convs, tail_fn).answer(_sample("q"))
    (_, tail_ctx), = tail_fn.calls
    assert len(tail_ctx) == limit
    assert tail_ctx == full[-limit:]  # tail = most recent sessions survive

    head_cfg = bench_cfg.with_overrides(
        {
            "benchmark.full_context.max_chars": limit,
            "benchmark.full_context.keep": "head",
        }
    )
    head_fn = recording_answer_fn_cls()
    FullContextRetriever(head_cfg, convs, head_fn).answer(_sample("q"))
    (_, head_ctx), = head_fn.calls
    assert head_ctx == full[:limit]
