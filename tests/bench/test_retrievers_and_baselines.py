"""GluoMemRetriever and baseline retrievers (BM25, VectorRAG, FullContext)."""

from __future__ import annotations

from pathlib import Path

import pytest

from gluomem.bench.baselines import (
    Bm25Retriever,
    FullContextRetriever,
    VectorRagRetriever,
    transcript,
)
from gluomem.bench.retriever import GluoMemRetriever
from gluomem.bench.types import Sample
from gluomem.embedding.providers import MockEmbedder
from gluomem.embedding.reranker import StubReranker


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


def _retriever(bench_cfg, ingested_db_dir, fixture_adapter, answer_fn):
    return GluoMemRetriever(
        bench_cfg,
        db_dir=ingested_db_dir,
        answer_fn=answer_fn,
        content_to_dias=fixture_adapter.content_to_dias(),
    )


def test_gluomem_retriever_dia_mapping_and_date_prefix(
    bench_cfg, ingested_db_dir, fixture_adapter, answer_fn
) -> None:
    r = _retriever(bench_cfg, ingested_db_dir, fixture_adapter, answer_fn)
    p = r.answer(_sample("Where does Bob live now Seattle move"))
    assert not p.abstained
    assert p.answer == "scripted answer"
    assert p.retrieved
    assert {d.dia_id for d in p.retrieved if d.dia_id is not None}
    assert all(d.frame_id is not None for d in p.retrieved)
    (_, context), = answer_fn.calls
    assert all(line.startswith("[") for line in context.splitlines())


def test_gluomem_dia_collision_prefers_evidence(
    bench_cfg, ingested_db_dir, fixture_adapter, answer_fn
) -> None:
    r = _retriever(bench_cfg, ingested_db_dir, fixture_adapter, answer_fn)
    p = r.answer(_sample("adopted a cat yesterday", evidence=("D2:3",)))
    cat_docs = [d for d in p.retrieved if "adopted a cat" in d.text]
    assert cat_docs and cat_docs[0].dia_id == "D2:3"
    p2 = r.answer(_sample("adopted a cat yesterday"))
    cat_docs2 = [d for d in p2.retrieved if "adopted a cat" in d.text]
    assert cat_docs2 and cat_docs2[0].dia_id == "D1:3"


def test_gluomem_abstain_and_missing_db(
    bench_cfg, ingested_db_dir, fixture_adapter, answer_fn, tmp_path: Path
) -> None:
    cfg = bench_cfg.with_overrides({"search.abstain.threshold": 1e9})
    r = GluoMemRetriever(
        cfg,
        db_dir=ingested_db_dir,
        answer_fn=answer_fn,
        content_to_dias=fixture_adapter.content_to_dias(),
    )
    p = r.answer(_sample("anything at all"))
    assert p.abstained is True
    assert p.answer == cfg.benchmark.answer.abstain_text
    assert p.retrieved == ()
    assert answer_fn.calls == []

    r_missing = GluoMemRetriever(
        bench_cfg,
        db_dir=tmp_path / "empty",
        answer_fn=answer_fn,
        content_to_dias=fixture_adapter.content_to_dias(),
    )
    with pytest.raises(FileNotFoundError, match="bench ingest"):
        r_missing.answer(_sample("q"))


def test_gluomem_embedder_and_reranker_factory_resolution(
    bench_cfg, ingested_db_dir, fixture_adapter, answer_fn, monkeypatch
) -> None:
    import gluomem.bench.retriever as retr

    sentinel = MockEmbedder(dim=bench_cfg.storage.vector_dim)
    emb_calls: list[str] = []

    def fake_make_embedder(cfg):
        emb_calls.append(cfg.embedding.model)
        return sentinel

    monkeypatch.setattr(retr, "make_embedder", fake_make_embedder)
    cfg = bench_cfg.with_overrides({"embedding.model": "minilm"})
    r = GluoMemRetriever(
        cfg,
        db_dir=ingested_db_dir,
        answer_fn=answer_fn,
        content_to_dias=fixture_adapter.content_to_dias(),
    )
    r.answer(_sample("cat"))
    assert emb_calls == ["minilm"]

    rerank_calls: list[str] = []

    def fake_make_reranker(cfg):
        rerank_calls.append(cfg.search.rerank.model)
        return StubReranker(logit=7.0)

    monkeypatch.setattr(retr, "make_reranker", fake_make_reranker)
    cfg_rerank = bench_cfg.with_overrides({"search.rerank.enabled": True})
    r_rerank = GluoMemRetriever(
        cfg_rerank,
        db_dir=ingested_db_dir,
        answer_fn=answer_fn,
        content_to_dias=fixture_adapter.content_to_dias(),
    )
    p = r_rerank.answer(_sample("cat adopted yesterday"))
    assert rerank_calls == ["reranker"]
    assert p.retrieved[0].score == pytest.approx(7.0)


def test_gluomem_reranker_injection_negative_logit_and_manifest_extra(
    bench_cfg, ingested_db_dir, fixture_adapter, answer_fn, monkeypatch
) -> None:
    import gluomem.bench.retriever as retr

    r_off = _retriever(bench_cfg, ingested_db_dir, fixture_adapter, answer_fn)
    assert r_off.manifest_extra()["rerank"]["active"] is False

    monkeypatch.setattr(
        retr, "make_reranker", lambda cfg: pytest.fail("must not resolve from config")
    )
    cfg = bench_cfg.with_overrides(
        {"search.rerank.enabled": True, "search.abstain.enabled": False}
    )
    r = GluoMemRetriever(
        cfg,
        db_dir=ingested_db_dir,
        answer_fn=answer_fn,
        content_to_dias=fixture_adapter.content_to_dias(),
        reranker=StubReranker(logit=-5.0),
    )
    p = r.answer(_sample("cat"))
    assert not p.abstained
    assert p.retrieved[0].score == pytest.approx(-5.0)
    extra = r.manifest_extra()
    assert extra["embedder"] == "mock"
    assert extra["rerank"]["active"] is True
    assert extra["abstain"]["score"] == "fusion"


def test_bm25_baseline(bench_cfg, fixture_adapter, answer_fn) -> None:
    r = Bm25Retriever(bench_cfg, fixture_adapter.conversations(), answer_fn)
    p = r.answer(_sample("Who moved to Seattle?", evidence=("D3:1",)))
    assert not p.abstained
    assert "D3:1" in {d.dia_id for d in p.retrieved}
    assert [d.rank for d in p.retrieved] == list(range(len(p.retrieved)))
    (_, context), = answer_fn.calls
    assert context.splitlines()[0].startswith("[")

    assert r.answer(_sample('cat AND (job* OR "promo:tion) - NEAR')).answer
    answer_fn.calls.clear()
    p_empty = r.answer(_sample("*** :::"))
    assert p_empty.abstained is True
    assert answer_fn.calls == []


def test_vector_rag_and_full_context_baselines(
    bench_cfg, fixture_adapter, answer_fn, recording_answer_fn_cls
) -> None:
    convs = fixture_adapter.conversations()
    vec_r = VectorRagRetriever(
        bench_cfg,
        convs,
        answer_fn,
        embedder=MockEmbedder(dim=bench_cfg.storage.vector_dim),
    )
    p = vec_r.answer(_sample("Where does Bob live?"))
    assert len(p.retrieved) == 9
    assert all(d.dia_id is not None for d in p.retrieved)
    scores = [d.score for d in p.retrieved]
    assert scores == sorted(scores, reverse=True)

    answer_fn.calls.clear()
    fc_r = FullContextRetriever(bench_cfg, convs, answer_fn)
    p_fc = fc_r.answer(_sample("What happened?"))
    assert p_fc.retrieved == ()
    (_, context), = answer_fn.calls
    for date in ("2023-05-08", "2023-06-21", "2023-08-02"):
        assert date in context

    full = transcript(convs[0])
    limit = len(full) // 2
    tail_fn = recording_answer_fn_cls()
    FullContextRetriever(
        bench_cfg.with_overrides({"benchmark.full_context.max_chars": limit}),
        convs,
        tail_fn,
    ).answer(_sample("q"))
    assert tail_fn.calls[0][1] == full[-limit:]

    head_fn = recording_answer_fn_cls()
    FullContextRetriever(
        bench_cfg.with_overrides(
            {
                "benchmark.full_context.max_chars": limit,
                "benchmark.full_context.keep": "head",
            }
        ),
        convs,
        head_fn,
    ).answer(_sample("q"))
    assert head_fn.calls[0][1] == full[:limit]
