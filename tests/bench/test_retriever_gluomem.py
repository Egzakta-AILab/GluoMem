"""GluoMemRetriever — dia mapping, date-prefixed context, abstention, errors."""

from __future__ import annotations

from pathlib import Path

import pytest

from gluomem.bench.retriever import GluoMemRetriever
from gluomem.bench.types import Sample


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


def test_retrieved_docs_carry_dia_ids(
    bench_cfg, ingested_db_dir, fixture_adapter, answer_fn
) -> None:
    r = _retriever(bench_cfg, ingested_db_dir, fixture_adapter, answer_fn)
    p = r.answer(_sample("Where does Bob live now Seattle move"))
    assert not p.abstained
    assert p.answer == "scripted answer"
    assert p.retrieved
    # Raw-turn frames map back to dia ids through the content bridge.
    dia_ids = {d.dia_id for d in p.retrieved if d.dia_id is not None}
    assert dia_ids  # at least one raw turn resolved
    assert all(d.frame_id is not None for d in p.retrieved)


def test_context_lines_are_date_prefixed(
    bench_cfg, ingested_db_dir, fixture_adapter, answer_fn
) -> None:
    r = _retriever(bench_cfg, ingested_db_dir, fixture_adapter, answer_fn)
    r.answer(_sample("cat adopted yesterday"))
    (_, context), = answer_fn.calls
    assert all(line.startswith("[") for line in context.splitlines())


def test_dia_collision_prefers_evidence(
    bench_cfg, ingested_db_dir, fixture_adapter, answer_fn
) -> None:
    """'Alice: I adopted a cat yesterday.' exists as D1:3 AND D2:3 (dedup keeps
    one frame); when the question's evidence names D2:3 the doc credits it."""
    r = _retriever(bench_cfg, ingested_db_dir, fixture_adapter, answer_fn)
    p = r.answer(_sample("adopted a cat yesterday", evidence=("D2:3",)))
    cat_docs = [d for d in p.retrieved if "adopted a cat" in d.text]
    assert cat_docs
    assert cat_docs[0].dia_id == "D2:3"
    # Without evidence preference the deterministic tie-break is the first dia.
    p2 = r.answer(_sample("adopted a cat yesterday"))
    cat_docs2 = [d for d in p2.retrieved if "adopted a cat" in d.text]
    assert cat_docs2[0].dia_id == "D1:3"


def test_abstain_answers_without_llm_call(
    bench_cfg, ingested_db_dir, fixture_adapter, answer_fn
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
    assert answer_fn.calls == []  # no LLM call on abstention


def test_query_embedder_resolves_from_config_not_mock(
    bench_cfg, ingested_db_dir, fixture_adapter, answer_fn, monkeypatch
) -> None:
    """The query-side embedder must come from config, not a silent
    MockEmbedder fallback — a mock query vector against real (MiniLM) frame
    vectors is noise and collapses hybrid search to keyword-only. No embedder
    injected + non-mock config MUST call make_embedder."""
    import gluomem.bench.retriever as retr
    from gluomem.embedding.providers import MockEmbedder

    sentinel = MockEmbedder(dim=bench_cfg.storage.vector_dim)
    calls: list[str] = []

    def fake_make_embedder(cfg):
        calls.append(cfg.embedding.model)
        return sentinel

    monkeypatch.setattr(retr, "make_embedder", fake_make_embedder)
    # Config says minilm (not mock) and no embedder is injected.
    cfg = bench_cfg.with_overrides({"embedding.model": "minilm"})
    r = GluoMemRetriever(
        cfg,
        db_dir=ingested_db_dir,
        answer_fn=answer_fn,
        content_to_dias=fixture_adapter.content_to_dias(),
    )
    r.answer(_sample("cat"))
    assert calls == ["minilm"], "query embedder must resolve from config"


def test_mock_config_builds_mock_through_the_factory(
    bench_cfg, ingested_db_dir, fixture_adapter, answer_fn
) -> None:
    """A config naming the mock (with fakes allowed) gets it from
    make_embedder; there is no separate mock shortcut to bypass the gate."""
    r = GluoMemRetriever(  # bench_cfg pins embedding.model=mock + allow_fake
        bench_cfg,
        db_dir=ingested_db_dir,
        answer_fn=answer_fn,
        content_to_dias=fixture_adapter.content_to_dias(),
    )
    p = r.answer(_sample("cat"))
    assert p.retrieved  # mock embedder works end to end


def test_reranker_resolves_from_config_not_silently_off(
    bench_cfg, ingested_db_dir, fixture_adapter, answer_fn, monkeypatch
) -> None:
    """With ``search.rerank.enabled`` true and no reranker injected, the
    retriever MUST build one from config — else HybridSearch's gate
    (``... and self._reranker is not None ...``) silently no-ops rerank while
    the manifest claims it ran. The stub replaces every pooled hit's ``final``
    with its logit."""
    import gluomem.bench.retriever as retr
    from gluomem.embedding.reranker import StubReranker

    calls: list[str] = []

    def fake_make_reranker(cfg):
        calls.append(cfg.search.rerank.model)
        return StubReranker(logit=7.0)

    monkeypatch.setattr(retr, "make_reranker", fake_make_reranker)
    cfg = bench_cfg.with_overrides({"search.rerank.enabled": True})
    r = GluoMemRetriever(
        cfg,
        db_dir=ingested_db_dir,
        answer_fn=answer_fn,
        content_to_dias=fixture_adapter.content_to_dias(),
    )
    p = r.answer(_sample("cat adopted yesterday"))
    assert calls == ["reranker"], "reranker must resolve from config, once"
    assert not p.abstained
    assert p.retrieved
    # Cross-encoder logit REPLACES rrf*relevance for the reranked pool.
    assert p.retrieved[0].score == pytest.approx(7.0)


def test_rerank_disabled_leaves_reranker_inactive(
    bench_cfg, ingested_db_dir, fixture_adapter, answer_fn
) -> None:
    """bench_cfg pins rerank off → no reranker resolved (active False), search
    still works on the rrf*relevance scale. Uses the real make_reranker, which
    returns None when disabled (no model, no network)."""
    r = _retriever(bench_cfg, ingested_db_dir, fixture_adapter, answer_fn)
    assert r.manifest_extra()["rerank"] == {
        "enabled": False,
        "model": "reranker",
        "pool": bench_cfg.search.rerank.pool,
        "active": False,
    }
    p = r.answer(_sample("cat"))
    assert p.retrieved  # works without reranking


def test_injected_reranker_takes_precedence_over_config(
    bench_cfg, ingested_db_dir, fixture_adapter, answer_fn, monkeypatch
) -> None:
    """An explicitly injected reranker (e.g. a test stub) wins and config
    resolution is skipped entirely — make_reranker is never called."""
    import gluomem.bench.retriever as retr
    from gluomem.embedding.reranker import StubReranker

    monkeypatch.setattr(
        retr, "make_reranker", lambda cfg: pytest.fail("must not resolve from config")
    )
    cfg = bench_cfg.with_overrides({"search.rerank.enabled": True})
    r = GluoMemRetriever(
        cfg,
        db_dir=ingested_db_dir,
        answer_fn=answer_fn,
        content_to_dias=fixture_adapter.content_to_dias(),
        reranker=StubReranker(logit=3.0),
    )
    p = r.answer(_sample("cat"))
    assert p.retrieved
    assert p.retrieved[0].score == pytest.approx(3.0)


def test_negative_logit_does_not_abstain_when_abstain_off(
    bench_cfg, ingested_db_dir, fixture_adapter, answer_fn
) -> None:
    """The bench run disables abstain for gluomem precisely because rerank puts a
    raw logit (often negative) into ``hits[0].final`` while abstain.threshold is
    0.0. With abstain off, a negative-logit top hit is still ANSWERED, not
    dropped — proving the runner's override protects answerable questions."""
    from gluomem.embedding.reranker import StubReranker

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
    assert p.answer == "scripted answer"
    assert p.retrieved
    assert p.retrieved[0].score == pytest.approx(-5.0)


def test_manifest_extra_reports_effective_config(
    bench_cfg, ingested_db_dir, fixture_adapter, answer_fn
) -> None:
    from gluomem.embedding.reranker import StubReranker

    cfg = bench_cfg.with_overrides({"search.rerank.enabled": True})
    r = GluoMemRetriever(
        cfg,
        db_dir=ingested_db_dir,
        answer_fn=answer_fn,
        content_to_dias=fixture_adapter.content_to_dias(),
        reranker=StubReranker(logit=1.0),
    )
    extra = r.manifest_extra()
    assert extra["embedder"] == "mock"
    assert extra["rerank"] == {
        "enabled": True,
        "model": "reranker",
        "pool": cfg.search.rerank.pool,
        "active": True,  # a reranker was actually resolved, not just requested
    }
    assert extra["abstain"] == {"enabled": cfg.search.abstain.enabled}


def test_missing_db_names_bench_ingest(
    bench_cfg, fixture_adapter, answer_fn, tmp_path: Path
) -> None:
    r = GluoMemRetriever(
        bench_cfg,
        db_dir=tmp_path / "empty",
        answer_fn=answer_fn,
        content_to_dias=fixture_adapter.content_to_dias(),
    )
    with pytest.raises(FileNotFoundError, match="bench ingest"):
        r.answer(_sample("q"))
