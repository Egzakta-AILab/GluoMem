# GluoMem

**The binding layer.** In particle physics, gluons bind quarks into stable matter. GluoMem binds an agent's scattered observations into stable, queryable memory — a minimal, fully-configurable agentic-memory core paired with a rigorous, reproducible benchmark harness for evaluating which memory mechanisms measurably improve retrieval and answer quality.

GluoMem is a from-scratch Python system providing typed **I/P/B memory frames** (a video-codec model of memory: keyframe facts, delta refinements, bidirectional corrections), a **bitemporal knowledge graph**, **hybrid retrieval** (BM25 + vector fused with RRF) over a separable 4-signal relevance model, **DSPy** structured extraction, **offline tuning**, and a **protocol-honest** evaluation harness.

**Build status:** complete and validated — deterministic core (config, SQLite + I/P/B frames, hybrid search + scoring, bitemporal KG), DSPy write path, benchmark harness (LoCoMo, LongMemEval-S/Oracle adapters, mem0 head-to-head), and offline tuning. 509 tests, all offline/deterministic by default. Measured results live in [RESULTS.md](./RESULTS.md), [LONGMEMEVAL_RESULTS.md](./LONGMEMEVAL_RESULTS.md), and [LOCOMO_RESULTS.md](./LOCOMO_RESULTS.md).

## The memory model

Memory is encoded like video:

- **I-frames** — self-contained facts (keyframes), content-hash deduped per GOP (group-of-pictures = topic/session scope).
- **P-frames** — deltas: refinements or new observations on a base I-frame.
- **B-frames** — bidirectional corrections written by the contradiction detector, carrying `{description, references}` and inheriting the corrected frame's importance.
- **Compaction** — GOPs periodically merge P-frames into their base I (re-embed + referential delete), like re-encoding a keyframe.

Every write funnels through a synchronous, failure-atomic index hook (FTS + whole-frame vector + chunk vectors in one transaction), so search never sees a half-indexed frame.

## Quickstart

Fully offline and deterministic by default. The only tests that touch a real model are marked `slow` and excluded (`addopts = "-m 'not slow'"`); live-integration tests skip without `OPENROUTER_API_KEY`.

```bash
uv sync                  # or: make install
make hooks               # one-time: pre-commit gate (lint+typecheck) + pre-push gate (tests)

uv run pytest            # or: make test       → 509 passed, offline
uv run ruff check .      # or: make lint       → clean
uv run pyrefly check     # or: make typecheck  → 0 errors

make check               # lint + typecheck + test in one shot

uv run gluomem --help    # CLI: ingest / bench run|rescore|gate|sample|fetch|ingest|verify
```

## Documents (read in order)

1. **[SPEC.md](./SPEC.md)** — purpose, scope, requirements, acceptance criteria.
2. **[DESIGN.md](./DESIGN.md)** — architecture, schema, every layer, config system, DSPy layer, tuning engine, harness.
3. **[PAPER.md](./PAPER.md)** — research questions, ablations, baselines, metrics, reproducibility, paper outline.
4. **[RESULTS.md](./RESULTS.md)** — headline benchmark results and how to reproduce them.

## At a glance

- **Memory model:** typed I/P/B frames over a bitemporal knowledge graph; the schema carries both `event_time` and an explicit trust/source column.
- **Everything is config** (pydantic-settings, `GLUOMEM_` env prefix): profile weights are data, ablations are config flips; precedence `defaults < conf/config.yaml < variant < env < with_overrides()`.
- **All LLM I/O via DSPy 3.x:** typed Pydantic outputs, optimized offline, version-pinned; all traffic rides OpenRouter behind a registry — a model swap is a config change.
- **Defaults:** `sentence-transformers/all-MiniLM-L6-v2` embeddings (384 dims), sqlite-vec exact KNN (no ANN), single mind file.
- **Tuning is offline:** cache candidate generations, re-rank thousands of configs against cached outputs, then validate the top-N with the LLM judge.
- **Evaluation is reproducible:** committed run dirs rescore byte-identically offline (`gluomem bench rescore <run> --check`); metric-protocol discipline is enforced at runtime; fake backends bannered loudly.

## References

- LoCoMo benchmark — arXiv:2402.17753.
- LongMemEval — arXiv:2410.10813.
- MemBench (objective multiple-choice; ACL Findings 2025) — arXiv:2506.21605. Distinct from the unrelated text-to-image MemBench.
- RRF — Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods," SIGIR 2009.
- DSPy — dspy.ai documentation.
- sqlite-vec — github.com/asg017/sqlite-vec.
- pydantic-settings — docs.pydantic.dev.
