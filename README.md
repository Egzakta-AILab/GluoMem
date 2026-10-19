# GluoMem

**The binding layer.** In particle physics, gluons bind quarks into stable matter. GluoMem binds an agent's scattered observations into queryable memory. It is a configurable agentic-memory core paired with a benchmark harness for measuring which memory mechanisms improve retrieval and answer quality.

GluoMem is an in-development Python memory framework (package version 0.1.0), not a certified production service. Embedded use runs against a single SQLite file per memory store. An optional HTTP service and worker run against SQLite or PostgreSQL 18 with pgvector. Every stored row belongs to a `MemoryScope` (`tenant_id`, `principal_id`, `namespace`); the service authenticates trusted hosts with one shared bearer key and takes the scope from request headers. It has no public per-user identity or RBAC, so the host application must decide which scope a caller may use. Provenance labels are not authorization boundaries.

## What is implemented


- **I/P/B memory frames** (a video-codec model of memory): I-frames hold self-contained facts and are content-hash deduplicated per GOP (group of pictures, a topic or session scope); P-frames hold refinements on a base I-frame; B-frames hold corrections written by the contradiction detector. Compaction merges a GOP's P-frames into its I-frame and records each merged frame's identity as lineage. A frame row and its FTS/vector index commit together; `FrameStore.repair_indexes()` rebuilds missing or stale indexes in files written by 0.0.1.
- **Temporal knowledge graph** with `valid_from`/`valid_to` on relations and relation retirement. A relation that is retired and later re-asserted keeps its earlier interval in `knowledge_relation_history`, so `as_of()` answers false inside the gap. Conflicting facts are ordered by valid time, not by when they were recorded.
- **One UTC convention**: naive timestamps are UTC, offset-aware ones are converted to UTC on write and read, and search recency reads time from an injectable clock ([timeutil](./gluomem/timeutil.py)).
- **Hybrid retrieval**: SQLite FTS5 (BM25) and sqlite-vec exact KNN, fused with RRF, then a four-signal relevance model (temporal, popularity, contextual, importance) and an optional cross-encoder reranker. Abstention is separate from ranking: `search.abstain.score` names the scale the threshold is read on (`fusion`, the default, gives the same decision with reranking on or off; `rerank` requires a declared calibration matching the reranker model), and every search returns an `AbstentionDecision` with the reason, score identity, top score and calibration. No threshold has been calibrated; the default `0.0` abstains only when no eligible evidence exists.
- **DSPy write path**: typed extraction behind an `Extractors` protocol, with deterministic fake extractors for offline tests, an admission gate (regex injection scanner, quarantine or reject; imports that restate a trusted fact with a different value are held as review candidates, `safety.conflict_review`), and an extraction cache. Quarantined frames are kept for inspection by id but excluded from search, current-state reads, graph linking and compaction ([eligibility rule](./gluomem/frames/eligibility.py)).
- **Fail-closed components**: each embedder, reranker and LLM backend is built from the provider its config names, or construction raises; nothing falls back to the mock embedder. Test doubles (`provider: fake` embedders, `fake` backends) need `components.allow_fake: true`. A `.mind` file records the fingerprint of the embedder that wrote its vectors (implementation, model, revision, dim, preprocessing, index schema) and rejects any other embedder before reading or writing a vector. `gluomem components` prints the effective components and their readiness without secrets; `gluomem index rebuild` and `gluomem index adopt` are the explicit ways to change or record a file's embedder.
- **Scoped storage**: scope and recall eligibility are pushed into every vector, full-text and LIKE query before `LIMIT`, so one scope's rows never fill another scope's top-k. PostgreSQL uses HNSW with iterative scans so filtered searches still return k rows. The Postgres schema is SQLAlchemy metadata in [pg_schema.py](./gluomem/storage/pg_schema.py) with Alembic revisions generated from it (`gluomem migrate upgrade|downgrade|current|heads|revision --autogenerate`); a backend refuses to open a database that is not at the head revision.
- **Memory lifecycle**: frames carry a kind (`fact`, `profile`, `preference`, `decision`, `procedure`, `episode`, `correction`, `summary`) and a status (`candidate`, `active`, `superseded`, `expired`, `forgotten`, `tombstoned`). Source events are idempotent per `(event_id, revision)`, and a reused key with a different payload is a conflict. Low-confidence writes become review candidates (`lifecycle.review_threshold`), and episodes expire after `lifecycle.episode_ttl_days` (`sweep_expired_episodes` marks due episodes `expired` and removes their FTS, vector and chunk projections). Correct, forget, scope erasure (`forget_scope`) and retract-source scrub the old plaintext from frames, indexes, stored event payloads, queued jobs and orphaned graph rows, then record a suppression key so a replay of the same content stays forgotten. Each of these bumps the scope generation.
- **Service and worker** (`gluomem[service]`): `gluomem serve` exposes `/v1/events`, `/v1/memories:recall`, `/v1/memories:sweep-expired`, memory listing, lineage, review, correct, forget, source retraction, export, job status and `/v1/status`; `/healthz` and `/readyz` are unauthenticated and minimal. `gluomem worker` processes durable jobs with leases, retries, dead-lettering and an optional `should_pause` backpressure hook. A job leased before a forget or correction is re-run at the new scope generation instead of writing stale data. `gluomem[client]` provides `GluoMemClient` and `AsyncGluoMemClient`.
- **Configuration**: one typed config tree (`pydantic-settings`, `GLUOMEM_` prefix). Precedence is `Pydantic defaults < env < with_overrides()`. `dspy.model` is required; GluoMem does not select a generation model for you. Optional per-task model overrides use the same registry.

## Native Hermes integration

The independently packaged [native Hermes provider](./integrations/hermes/gluomem_hermes/README.md) talks to a private GluoMem service; it does not load DSPy, embedding models, or the memory database into Hermes. Its explicit-profile installer does not activate the provider or migrate existing Hermes memory. Automatic human-turn capture defaults off, observations require review, and user confirmation is reserved for trusted operator controls. Model-facing tools cannot select arbitrary scopes or mark their own proposals user-verified.

The provider source and wheel are separate from the core wheel. Follow its README for installation, service setup, profile-scoped status/pause/resume/review/provenance/correct/forget/export commands, and current host-certification limits. Never expose the internal service bearer key to browser clients or use it as public SaaS tenant authentication.

## Evaluation & Benchmarks (`gluomem-bench`)

The benchmark harness, LoCoMo / LongMemEval / BEAM 128K adapted-rubric / MemoryAgentBench replay / MINJA adversarial / SR-EN evaluation suites, Optuna retrieval tuning (`tune`, `ablate`), `mem0` comparison adapter, and DSPy prompt optimization live in the companion `gluomem-bench` repository (`https://github.com/UkisAI-Egzakta/gluomem-bench`). Adapter availability and integration smokes are not full-campaign quality results; see that repository's evidence reports before making comparisons.

## Installation

The core package is small. Model-backed components are optional extras. When a configured component needs an extra that is not installed, GluoMem raises `MissingExtraError` naming the extra; it never substitutes a fake or mock component because a package is missing.

| Install | Adds | Readiness |
|---|---|---|
| `gluomem` (core) | Config, SQLite storage, I/P/B frames, hybrid search, KG, fake extractors, mock embedder | Works offline; real embedding, reranking and DSPy extraction raise `MissingExtraError` |
| `gluomem[llm]` | DSPy >= 3.4.1: extraction | Requires an explicitly selected registered model, its endpoint and credentials where needed; local OpenAI-compatible servers are supported |
| `gluomem[embeddings]` | sentence-transformers MiniLM embedder and cross-encoder reranker | Downloads models from Hugging Face on first use |
| `gluomem[postgres]` | `psycopg`, `pgvector`, Alembic and SQLAlchemy for the PostgreSQL backend | Connects via `storage.backend: postgres` and `storage.dsn`; run `gluomem migrate upgrade` first |
| `gluomem[service]` | FastAPI, uvicorn and httpx for `gluomem serve` and `gluomem worker` | Needs a service key in `GLUOMEM_SERVICE_KEY` or `--service-key-file` |
| `gluomem[client]` | httpx for the `gluomem.client` SDK | Talks to a running service |
| `gluomem[classifier]` | Optional GLiNER2 local classifier dependencies | Model provisioning and readiness remain explicit |
| `gluomem[all]` | `postgres`, `llm`, `embeddings`, `classifier` and `service` | All of the above |

`gluomem.extras.status()` reports which extras are installed without importing or downloading anything.

## Quickstart

Requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). The checks below make no paid model calls; dependency installation may require network access.

```bash
make install             # uv sync --locked --extra all (full development environment)
# or: make install-offline  (uv sync --locked --extra llm --extra service --extra postgres)
make check               # ruff lint + pyrefly type check + offline tests
make wheel-check         # build the wheel, install it core-only outside the checkout, write and read
uv run gluomem --help    # CLI: components, index, migrate, serve, worker
```

CI (`.github/workflows/ci.yml`) runs `make check` and `make wheel-check` on Python 3.11, 3.12 and 3.13 on every pull request, plus the live PostgreSQL tests (`pytest -m postgres`) against a pgvector PG18 service container. Locally those tests skip unless `GLUOMEM_TEST_PG_DSN` points at a database where they may create and drop schemas. Slow tests that download models run only when the workflow is dispatched manually with `run_model_tests`.

Inspect component readiness:

Select your registered generation model explicitly, either with
`GLUOMEM_DSPY__MODEL` or `load_config(overrides={"dspy.model": "your-model"})`.
Register custom deployments in `models`; no model or paid provider is selected
implicitly. This replaces the old `dspy.default_model` setting. The test suite
selects its own fixture model and does not make model calls by default.

```bash
uv run gluomem components
```

Tests that download real models are marked `slow` and excluded by default (`addopts = "-m 'not slow'"`). Live provider tests skip unless `OPENROUTER_API_KEY` is set. Copy `.env.example` to `.env` only when you intend to make live calls.

Optional: `make hooks` installs the pre-commit (lint and type check) and pre-push (tests) gates.

## Safety boundaries

Source retractions permanently suppress the exact scoped reference, including sources
retracted before ingestion. Logical erasure does not guarantee removal from offline
hosts, WAL, backups or storage media. Restored stores must remain quarantined until
revocations are reconciled; an old backup alone is not a safe privacy rollback.
API/database readiness does not certify worker or model readiness. Native delivery
limits and acknowledgement semantics are documented in the Hermes provider README.

## Repository boundary

This repository holds the memory library, CLI, HTTP service/worker, Python SDK, and runtime adapters under `integrations/`. The native Hermes provider is independently packaged and is not installed by the core wheel. Application-specific authorization and deployment wiring belong in the consuming application. Benchmarks and offline tuning live in `gluomem-bench`.

## Documents

1. **[SPEC.md](./SPEC.md)** — original research specification: purpose, scope, requirements, acceptance criteria.
2. **[DESIGN.md](./DESIGN.md)** — architecture, schema, layers, config system, DSPy layer.
3. **[CONTEXT.md](./CONTEXT.md)** — domain terms, code mapping and architectural decisions.
4. **[Structured extraction policy](./docs/structured-extraction.md)** — opt-in schema transport, validation, and native output-field limits.


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
