# GluoMem — Product & Requirements Specification

> This document is the governing specification: it describes the full intended system, its requirements, and its acceptance criteria.

> Companions: `DESIGN.md` (architecture), [`docs/research/PAPER.md`](./docs/research/PAPER.md) (experimental design).

> **Implementation status (2026-10-13, version 0.0.1).** This is the original research specification, not a statement of what ships. Not implemented: the tuning engine (G5, §7 tuning, FR-14 `tune`/`ablate`, acceptance criterion 4) and the committed reproducible run artifacts (Q-4, acceptance criteria 2 and 6; see [the evidence inventory](./docs/research/EVIDENCE_INVENTORY.md)). MemBench has no adapter. The `source` enum (§9 schema constant) labels provenance; it is not tenant, workspace or access control. The README lists the current scope and known limitations.

---

## 0. Scope

`gluomem` is a minimalistic, clean Python implementation of an agentic-memory layer, evaluated rigorously on standard benchmarks.

The deliverable is twofold:
1. **A package** — a small, well-factored memory layer (`gluomem/`) that ingests conversations, stores typed memories plus a bitemporal KG, and serves relevance-ranked recall.
2. **An evaluation** — a reproducible benchmark and ablation harness that produces protocol-honest numbers.

Out of scope: a production system. No production hardening, no SLAs, no multi-tenant server. The design optimizes for clarity, configurability, and measurability.

---

## 1. Design decisions

| # | Topic | Decision |
|---|-------|----------|
| **D1** | Frame model | **Real I/P/B frames.** Full implementation with production writers, so typed corrections plus bitemporal update can be measured against an I-only configuration. |
| **D2** | Repo layout | **Standalone, self-contained repository.** The memory layer lives in a single package (`gluomem/`). |
| **D3** | Default LLM providers | LLM tasks → **Anthropic Claude (Haiku-class)** via DSPy/LiteLLM (`ANTHROPIC_API_KEY`); embeddings → **in-process MiniLM** by default (Ollama optional). |
| **D4** | Scale target | Not a focus. sqlite-vec **exact KNN only**; the target benchmarks (LongMemEval-S, MemBench, LoCoMo) fit under the exact ceiling. ANN backends are out of scope. |
| **D5** | Gold-label budget | **Reuse benchmark-provided labels** (answer spans / MCQ keys) as relevance gold. No separate labeling spend. |
| **D6** | Multi-user | **Single mind, no RBAC, no workspaces.** |

---

## 2. Goals

- **G1 — Faithful, minimal implementation.** I/P/B frames + GOP sessions, bitemporal KG, hybrid search (BM25 + vector → RRF) × a 4-signal relevance multiplier, DSPy extraction. Small and readable, not feature-complete.
- **G2 — Everything configurable** (for ablations). Models, profile weights, RRF `k`, half-life, importance multipliers, BFS depth/steps, chunk sizes — all config, nothing hardcoded. Ablations are config flips.
- **G3 — DSPy for all LLM I/O.** Typed Pydantic inputs/outputs; prompts optimized offline; reproducible artifacts.
- **G4 — Rigorous evaluation.** Pluggable benchmark harness; metric-protocol discipline as a runtime invariant; offline reproducible rescore; ablation and baseline studies; statistical significance.
- **G5 — Cost-efficient tuning.** Find good configs offline at near-zero cost (cache candidates → re-rank → IR metric), validate the top few with the LLM judge.
- **G6 — Correct by construction.** Known failure modes (DESIGN architecture notes) are designed out, so measured effects are attributable to the design.

## 3. Non-goals

- Production surfaces: MCP server, lifecycle hooks, wiki compiler, enrichment/synthesis, identity/awareness, multi-adapter harvest → out of scope or stubbed. Only what the benchmarks need.
- Scale/ANN, multi-user/RBAC, multimodal, parametric memory, RL controller — out (some are future work).
- A polished CLI/UX. A minimal experiment-runner CLI is sufficient.

---

## 4. Functional requirements

### 4.1 Memory representation
- **FR-1** Typed frames **I/P/B** with real writers: I=fact, P=delta/update, B=correction (`{description, references}`). B-frames inherit importance from the corrected frame.
- **FR-2** GOP/session grouping; reconstruct current state = latest I + subsequent P.
- **FR-3** Content-hash dedup; importance levels with configurable multipliers.
- **FR-4** Compaction (configurable trigger/keep-window) that re-embeds the merged frame and cleans references.
- **FR-5** Bitemporal KG: entities/relations with `valid_from`/`valid_to`, retire-not-delete, `as_of(T)`.

### 4.2 Retrieval
- **FR-6** Hybrid: FTS5 BM25 + vector KNN → RRF (config `k`); `final = rrf × relevance`.
- **FR-7** Relevance = configurable weighted sum of temporal / popularity / contextual(KG) / importance; profiles are data.
- **FR-8** Optional cross-encoder reranker; chunk-level retrieval with whole-frame fallback.
- **FR-9** Working **abstain** gate (config threshold).
- **FR-10** Temporal range filter on `event_time` — a column distinct from `created_at`.

### 4.3 Write path
- **FR-11** Ingest a benchmark conversation/document into memory (the critical path for evaluation): session → frames, lane extraction (fact/event/profile) and cognify (entities/relations) via DSPy.
- **FR-12** A single admission point for writes (dedup + index + a minimal injection scan). Security is not a research focus, but routing all writes through one gate keeps the architecture clean.
- **FR-13** Adapters limited to benchmark formats (LoCoMo/LongMemEval JSON, plain text).

### 4.4 Experiment surface
- **FR-14** Minimal CLI: `gluomem ingest`, `gluomem bench run`, `gluomem bench rescore`, `gluomem tune`, `gluomem ablate`.
- **FR-15** Reproducibility artifact: SHA-pinned datasets, committed per-row judgments, version-pinned DSPy programs, seeds.

---

## 5. Quality / evaluation requirements (see `docs/research/PAPER.md`)

- **Q-1** Pluggable harness: `DatasetAdapter · Retriever · Judge · MetricRegistry`.
- **Q-2** Metric discipline is a runtime invariant. Every score carries a `ProtocolTag`; the aggregator refuses to mix metric/judge families. F1, Recall@k, MCQ-acc, LLM-judge-acc, and nDCG never share a column.
- **Q-3** Judges: objective-MCQ (no LLM) · single · **trio-strict** (default headline). The single-judge score is recorded alongside to quantify self-judge inflation: a single self-judge tends to score a few points above an independent trio-strict panel, and the gap is reported as a measured quantity per dataset rather than assumed.
- **Q-4** Offline rescore: commit per-row judgments + dataset SHA + manifest; `bench rescore` re-derives the headline numbers offline against committed artifacts with no network or model calls.
- **Q-5** Target benchmarks: **LoCoMo** (baseline anchor), **LongMemEval-S** (P0), **MemBench** (P0, objective MCQ), **MemoryAgentBench-AR** (P1), **ConvoMem** small (P1).
- **Q-6** Baselines: full-context (no memory), naive vector RAG, BM25-only — run in-harness. Vendor numbers (mem0/Zep) are cited with their protocol, never co-tabulated with gluomem's.
- **Q-7** Ablations (config flips, see `PAPER.md`): each scoring signal on/off, profiles, rerank on/off, chunk vs whole-frame, KG on/off, I/P/B vs I-only, RRF-k sweep, embedder variants.
- **Q-8** Statistical rigor: train/val/test splits, paired significance tests, seed control; a 0.2pp nDCG difference is noise, not a result.

### 5.1 Benchmark properties

- **LoCoMo** — long-term conversational memory QA. Average conversation ~9.2k tokens (~19 sessions, ~300 turns). arXiv:2402.17753.
- **LongMemEval-S** — ~115k tokens, ~50 sessions, 500 questions; 7 question types spanning 5 core abilities (information extraction, multi-session reasoning, temporal reasoning, knowledge update, abstention). arXiv:2410.10813.
- **MemBench** — objective multiple-choice evaluation, no LLM judge. arXiv:2506.21605 (ACL Findings 2025). This is the memory-agent MemBench; it is unrelated to the text-to-image generation benchmark of the same name.
- **MemoryAgentBench** and **ConvoMem** — additional conversational-memory suites; see their respective project sources.

---

## 6. Configurability requirements (G2)

- One typed config tree (`AppConfig`, pydantic-settings v2); sections `models, storage, embedding, chunking, search, scoring, kg, write, dspy, tuning, benchmark`.
- Precedence: defaults < `conf/config.yaml` < variant overlays < env < CLI/override.
- **Model registry**: every embedder/LLM/judge by name. Profile weights are a validated simplex in config. Cross-validation (embedder dim == vector dim).
- Ablations and sweeps inject overrides without editing files; each override is logged with its result row.

The complete configuration surface is enumerated in `conf/config.yaml`.

## 7. LLM and tuning requirements (G3, G5)

- All LLM I/O via **DSPy 3.x**: typed Pydantic types on `dspy.OutputField`; optimize offline only; load version-pinned artifacts; production never compiles. Validity retries use `dspy.Refine`. Prompt/weight optimization uses `MIPROv2` / `BootstrapFewShot`; programs are persisted with `save(save_program=True)`. (DSPy 3.x removed `TypedPredictor`, `Assert`, and `Suggest`.)
- Per-role routing via the registry: a cheap model for extraction, strong and independent models for the trio judge.
- Tuning is offline-first: cache each query's candidate pool once → re-rank thousands of param vectors with zero model calls, scored by nDCG/Recall@k against benchmark-derived gold labels → two-tier promotion of the top-N to the LLM judge. Sweeps use Optuna (TPE sampler, Hyperband pruner).

### 7.1 Model identifiers

| Role | Model id |
|------|----------|
| Extraction / default LLM | `anthropic/claude-haiku-4-5-20251001` (dated snapshot for reproducibility) |
| Trio judge | `anthropic/claude-sonnet-4-6` (pin a dated snapshot when finalized) |
| Embedder | `sentence-transformers/all-MiniLM-L6-v2` (384 dims) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

---

## 8. Principles

- **P-1** Minimal and readable over complete. Cut anything not needed for the package or the paper.
- **P-2** Swappable behind protocols (`VectorStore`, `Embedder`, `Reranker`, `LLM`) — but only sqlite-vec + MiniLM + Anthropic are implemented in v1.
- **P-3** Exact recall by default (reproducibility moat).
- **P-4** No dead features. I/P/B writers run; abstain gates engage; the injection scanner is wired.
- **P-5** Reproducibility is sacred: seeds, SHAs, committed judgments, pinned artifacts.
- **P-6** Fix known failure modes so measured effects are real.

---

## 9. Design constants

These values are gluomem's design defaults; all are config-reachable (G2).

- **Hybrid fusion** — RRF with `k=60`; `finalScore = rrfScore × relevanceScore`.
- **Relevance signals** — weighted sum of temporal, popularity, contextual (KG), and importance. The `balanced` profile weights are `{temporal: 0.4, popularity: 0.2, contextual: 0.2, importance: 0.2}`.
- **Temporal decay** — `temporal × 0.5^(days/30)` (30-day half-life) with a 7-day flat boost anchored to `created_at`.
- **Popularity** — `1 + log10(1 + access_count) × 0.1`.
- **Contextual** — step function `1.0 / 0.7 / 0.4 / 0.2 / 0.0` at BFS distances 0–4 from KG seed entities.
- **Importance multipliers** — critical 2.0, important 1.5, normal 1.0, temporary 0.7, deprecated 0.3.
- **Compaction** — fires when a GOP accumulates more than 10 P-frames; the keep-window retains the 5 most recent, merging older P-frames into the I-frame body.
- **Chunking** — 2000-char chunks with 200-char overlap; whole-frame fallback at 1500 chars.
- **Lane multipliers** — fact/event/profile lanes carry distinct importance weighting applied at admission.
- **Schema** — frames carry both `created_at` and `event_time`; entities/relations carry `valid_from`/`valid_to`. The `source` column uses a CHECK constraint over 8 values: `user_stated`, `tool_verified`, `agent_inferred`, `import`, `system`, `personal`, `workspace`, `team_sync`. A trust column records provenance confidence.
- **Search tokenization** — the FTS query path preserves technical tokens (version strings, error codes, dotted method names, short identifiers) rather than stripping them.

---

## 10. Success criteria (acceptance)

1. The package ingests a benchmark and serves recall end-to-end (LoCoMo + MemBench first).
2. It runs LongMemEval-S + MemBench, emits protocol-tagged numbers, and `bench rescore` reproduces them offline against committed artifacts with no network or model calls.
3. Every constant is config-reachable; an ablation (e.g. KG-off, I-only, rerank-off) runs from a config flip with no code edit.
4. A weight-tuning run executes thousands of Tier-1 trials with zero model calls, validates top-N with the trio judge, and never ships a regression.
5. All LLM calls are DSPy with pinned artifacts; swapping the extraction model is a registry-name change.
6. The research questions in `PAPER.md` are answerable from committed, reproducible artifacts; self-judge vs trio-strict inflation is reported as a measured per-dataset quantity.
7. No dead features; ablation effects are attributable to the design.

---

## 11. Deferred / future work
ANN scale, RL/learned controller, procedural memory, sleep-time reorganization, multimodal, production surfaces (MCP/hooks), multi-user. Reachable via protocols where cheap; implementation out of scope.

---

## References

- **RRF** — Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods," SIGIR 2009.
- **DSPy** — dspy.ai documentation.
- **sqlite-vec** — github.com/asg017/sqlite-vec.
- **Optuna** — optuna.org (TPE sampler, Hyperband pruner).
- **pydantic-settings** — docs.pydantic.dev.
- **LoCoMo** — arXiv:2402.17753.
- **LongMemEval** — arXiv:2410.10813.
- **MemBench** (memory agents, ACL Findings 2025) — arXiv:2506.21605.
- **MemoryAgentBench**, **ConvoMem** — cited by project source.
- **Anthropic model ids** — platform.claude.com model overview.
