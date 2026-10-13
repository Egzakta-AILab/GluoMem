# GluoMem

**The binding layer.** In particle physics, gluons bind quarks into stable matter. GluoMem binds an agent's scattered observations into queryable memory. It is a configurable agentic-memory core paired with a benchmark harness for measuring which memory mechanisms improve retrieval and answer quality.

GluoMem is a research-stage Python library (version 0.0.1). It runs locally against a single SQLite file per memory store. It is not a service: there is no HTTP API, no authentication, and no tenant, workspace or access-control model. The `source` values `personal`, `workspace` and `team_sync` are provenance labels on a frame, not authorization boundaries.

## What is implemented

- **I/P/B memory frames** (a video-codec model of memory): I-frames hold self-contained facts and are content-hash deduplicated per GOP (group of pictures, a topic or session scope); P-frames hold refinements on a base I-frame; B-frames hold corrections written by the contradiction detector. Compaction merges a GOP's P-frames into its I-frame.
- **Temporal knowledge graph** with `valid_from`/`valid_to` on relations and relation retirement.
- **Hybrid retrieval**: SQLite FTS5 (BM25) and sqlite-vec exact KNN, fused with RRF, then a four-signal relevance model (temporal, popularity, contextual, importance) and an optional cross-encoder reranker.
- **DSPy write path**: typed extraction behind an `Extractors` protocol, with deterministic fake extractors for offline tests, an admission gate (regex injection scanner, quarantine or reject), and an extraction cache. Quarantined frames are kept for inspection by id but excluded from search, current-state reads, graph linking and compaction ([eligibility rule](./gluomem/frames/eligibility.py)).
- **Benchmark harness**: LoCoMo and LongMemEval (S and oracle) adapters, full-context, BM25 and vector-RAG baselines, an optional mem0 comparison, judge panels, protocol tags, offline rescore of a run directory, and usage accounting.
- **Configuration**: one typed config tree (pydantic-settings, `GLUOMEM_` prefix). Precedence is `defaults < conf/config.yaml < variant < env < with_overrides()`.

## What is not implemented

- **Offline numeric tuning.** `gluomem/tune` is a placeholder, and `gluomem tune` and `gluomem ablate` exit with "not implemented". The tuning design in [DESIGN.md §9](./DESIGN.md#9-tuning-subsystem) is a plan.
- **Prompt optimization quality metric.** `make optimize-prompts` runs DSPy MIPROv2 with a parse-or-exact-match metric; it spends real tokens and is never run automatically.
- **Service features**: HTTP or MCP API, multi-user identity, tenants, workspaces, RBAC, durable jobs, deployment packaging.

## Known limitations at 0.0.1

These are tracked for the first hardening iteration and are being fixed:

- Frame insertion and its FTS/vector indexing are separate commits; an indexing failure can leave a stored but unsearchable frame that deduplication then hides on retry.
- Reasserting a retired graph relation rewrites the original row's validity interval.
- A timezone-aware `created_at` can raise `TypeError` during search recency scoring.
- Embedding compatibility is checked by vector dimension only.
- With reranking enabled, the abstention threshold is compared against reranker logits instead of fusion scores.

## Evidence status

The benchmark reports in [docs/research/](./docs/research/README.md) are historical. The `experiments/` run directories they cite were never published, so their numbers cannot currently be rescored. See the [evidence inventory](./docs/research/EVIDENCE_INVENTORY.md). Use them as research notes, not as product claims, until a new rescorable baseline is published.

## Installation

The core package is small. Model-backed components are optional extras. When a configured component needs an extra that is not installed, GluoMem raises `MissingExtraError` naming the extra; it never substitutes a fake or mock component because a package is missing.

| Install | Adds | Readiness |
|---|---|---|
| `gluomem` (core) | Config, SQLite storage, I/P/B frames, hybrid search, KG, fake extractors, mock embedder | Works offline; real embedding, reranking and DSPy extraction raise `MissingExtraError` |
| `gluomem[llm]` | DSPy 3.2.1: extraction, answerer, judges, prompt optimization | Needs `OPENROUTER_API_KEY` at call time; calls are paid |
| `gluomem[embeddings]` | sentence-transformers MiniLM embedder and cross-encoder reranker | Downloads models from Hugging Face on first use |
| `gluomem[all]` | `llm` and `embeddings` | Both of the above |
| `gluomem[mem0]` | mem0ai 2.0.11 comparison baseline | Benchmark comparison only |

`gluomem.extras.status()` reports which extras are installed without importing or downloading anything.

## Quickstart

Requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). Every command below runs offline and makes no paid model calls.

```bash
make install             # uv sync --locked --extra all (full development environment)
# or: make install-offline  (uv sync --locked --extra llm; enough for make check, no torch)
make check               # ruff lint + pyrefly type check + offline tests
make wheel-check         # build the wheel, install it core-only outside the checkout, write and read
uv run gluomem --help    # CLI: ingest, bench run|rescore|gate|sample|fetch|ingest|verify
```

CI (`.github/workflows/ci.yml`) runs `make check` on Python 3.11, 3.12 and 3.13 and `make wheel-check` on every pull request. Slow tests that download models run only when the workflow is dispatched manually with `run_model_tests`.

Deterministic ingest of a bundled synthetic fixture, using the fake extractors and mock embeddings:

```bash
uv run gluomem ingest tests/fixtures/locomo_tiny.json --db /tmp/demo.mind \
  --override write.extraction_backend=fake --override embedding.model=mock
```

Tests that download real models are marked `slow` and excluded by default (`addopts = "-m 'not slow'"`). Live provider tests skip unless `OPENROUTER_API_KEY` is set. The default configuration uses DSPy extraction through OpenRouter, so `gluomem ingest` and `gluomem bench run` without the overrides above call a paid provider when a key is present. Copy `.env.example` to `.env` only when you intend to make live calls.

Optional: `make hooks` installs the pre-commit (lint and type check) and pre-push (tests) gates.

## Repository boundary

This repository holds the memory library, its CLI and the evaluation harness. It does not contain a service, client SDKs, assistant adapters or KVARK integration code. Those are planned as separate work that uses this library.

## Documents

1. **[SPEC.md](./SPEC.md)** — original research specification: purpose, scope, requirements, acceptance criteria.
2. **[DESIGN.md](./DESIGN.md)** — architecture, schema, layers, config system, DSPy layer, planned tuning engine, harness.
3. **[docs/research/](./docs/research/README.md)** — archived experimental plan and historical benchmark reports.

## License

Apache License 2.0. See [LICENSE](./LICENSE) and [NOTICE](./NOTICE). Benchmark datasets are not included and keep their own licenses; LoCoMo is CC BY-NC 4.0.

## References

- LoCoMo benchmark — arXiv:2402.17753.
- LongMemEval — arXiv:2410.10813.
- MemBench (objective multiple-choice; ACL Findings 2025) — arXiv:2506.21605. Distinct from the unrelated text-to-image MemBench.
- RRF — Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods," SIGIR 2009.
- DSPy — dspy.ai documentation.
- sqlite-vec — github.com/asg017/sqlite-vec.
- pydantic-settings — docs.pydantic.dev.
