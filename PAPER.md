# gluomem — Experimental Design & Paper Plan

> The deterministic core — configuration, SQLite storage with typed I/P/B frames, hybrid search + scoring, and the bitemporal KG — is implemented and tested; the DSPy ingest pipeline, benchmark harness, offline tuning, and experiment matrix described below build on that core. This document is the governing experimental plan.

This document defines what is measured and claimed; every number must be answerable from committed, offline-reproducible artifacts.

---

## 1. Framing & contributions

**Title:** *gluomem: A Minimal, Configurable Agentic-Memory Layer and an Honest Evaluation of What Actually Helps.*

**Thesis.** Most agentic-memory systems bundle many mechanisms and report numbers under permissive, non-comparable protocols. gluomem is a small, fully-configurable memory layer in which each mechanism is a separable config flag, so every mechanism can be ablated cleanly and evaluated under a strict, reproducible protocol. The goal is to determine *which memory mechanisms actually move which metrics*, and to show that good configurations can be found offline against committed artifacts with no model calls.

**Contributions (C1–C5):**
- **C1 — A clean, configurable reference implementation** of a hybrid (BM25+vector→RRF) memory layer with a separable 4-signal relevance model, a bitemporal KG, and typed I/P/B frames. Every mechanism is a config flag, so every ablation is reproducible.
- **C2 — A mechanism ablation study**: the marginal contribution of each scoring signal (temporal/popularity/contextual/importance), the KG contextual signal, the reranker, chunking, and the I/P/B correction model — on standard benchmarks.
- **C3 — An honest-protocol evaluation**: trio-strict vs single self-judge inflation quantified per dataset, with a metric-discipline harness that refuses to mix metric families, plus an offline rescore that runs against committed artifacts with no network or model calls.
- **C4 — A cost-efficient tuning method**: cache-candidates-then-rescore reduces relevance/fusion tuning to an offline numeric optimization (thousands of trials, zero model calls); tuned configs match or beat hand-set defaults.
- **C5 — A reproducibility artifact**: SHA-pinned data, committed per-row judgments, version-pinned DSPy programs, one-command re-derivation.

---

## 2. Research questions & hypotheses

| RQ | Question | Hypothesis | How tested |
|----|----------|-----------|-----------|
| **RQ1** | How does the layered hybrid memory compare to full-context and naive RAG on conversational memory? | gluomem > naive-RAG; loses to full-context at small scale, wins as history grows | Main table vs baselines, LongMemEval-S / LoCoMo / MemBench |
| **RQ2** | What does each **scoring signal** contribute? | temporal + importance carry most; popularity marginal; contextual helps multi-hop/temporal | Leave-one-out + add-one-in ablation of the 4 signals |
| **RQ3** | Does the **bitemporal KG contextual** signal help, and where? | helps multi-hop & entity-centric queries, neutral on single-hop | KG-on vs KG-off, sliced by question type |
| **RQ4** | Does the **I/P/B correction model** improve knowledge-update / conflict-resolution vs I-only/overwrite? | I/P/B > I-only on update/abstention categories | `write.frame_model = ipb \| i_only \| overwrite` on LongMemEval update + ConvoMem changing-facts |
| **RQ5** | Can **cheap offline tuning** find configs that match/beat hand-set defaults with no model calls? | yes; Tier-1 nDCG-optimized config ≥ default on Tier-2 QA | tuned vs default, with Tier-2 trio-judge validation |
| **RQ6** | How large is **self-judge inflation** vs independent trio-strict? | a single self-judge scores a few points above an independent trio-strict panel; magnitude is dataset-dependent and reported as measured | single vs trio-strict on the same predictions |
| **RQ7** | Does the **reranker / chunking** earn its cost? | rerank helps single-hop; chunk-level > whole-frame on long frames | rerank-on/off, chunk vs whole-frame |

---

## 3. Datasets

| Dataset | Role | Task | Native metric | Judge | Scale | Notes |
|---------|------|------|---------------|-------|-------|-------|
| **MemBench** | **P0** | factual + reflective MCQ | MCQ accuracy | **objective (no LLM)** | fits exact KNN | cheapest, deterministic — first signal |
| **LongMemEval-S** | **P0** | 7 question types over 5 core abilities incl. knowledge-update + abstention | QA accuracy | LLM-judge (category-routed) | ~115k tok, ~50 sessions, 500 questions | where KG + abstain + I/P/B differentiate |
| **LoCoMo** | baseline anchor | multi-session conv QA | LLM-judge / F1 | single + trio | ~9.2k tok avg (~19 sessions, ~300 turns) | comparison anchor |
| **MemoryAgentBench-AR** | P1 | accurate-retrieval slice | recall / exact-match | objective | mixed | run AR slice only |
| **ConvoMem** (small tiers) | P1 | assistant/changing-facts/abstention | dual cheap-judge | single | small tiers | RAG-vs-full-context crossover; changing-facts → RQ4 |

LongMemEval-S spans 5 core abilities (information extraction, multi-session reasoning, temporal reasoning, knowledge update, abstention) across 7 question types. MemBench here is the objective multiple-choice agentic-memory benchmark (arXiv:2506.21605); it is unrelated to the text-to-image MemBench of the same name.

All datasets are **SHA-pinned** in `benchmarks_data/` with a sampling manifest (seed, subset). Gold relevance labels for tuning are derived from each benchmark's own answer spans / MCQ keys — no separate labeling spend.

**Out of scope** (justified in "limitations"): BEAM/PersonaMem-1M (above exact-KNN scale), DMR (saturated), AMA-Bench/MemoryCD (off-modality), Procedural-mem (diagnostic-only).

---

## 4. Metrics & protocol discipline

- Every number carries a `ProtocolTag {metric_family, judge_family, judge_count, backbone, dataset_sha}`; the harness **refuses** to average across incompatible tags. **F1, Recall@k, MCQ-accuracy, LLM-judge-accuracy, nDCG never share a column.**
- **Headline** for free-form benchmarks = **trio-strict** (3 independent backbones: Anthropic + OpenAI-class + open). **Single-judge always reported alongside** to quantify inflation (RQ6).
- MCQ benchmarks (MemBench, AR) use **objective match — no LLM judge.**
- Retrieval-quality diagnostics (for ablations/tuning): **nDCG@10, Recall@20, MRR** against benchmark-derived gold (no LLM).
- Report **both** end-task QA accuracy and retrieval diagnostics; never present a retrieval proxy as QA accuracy.

**Judge models.** The default extraction/cheap-judge model id is `anthropic/claude-haiku-4-5-20251001` (dated snapshot, for reproducibility). The strong judge in the trio is `anthropic/claude-sonnet-4-6` (pin a dated snapshot when finalized). The embedder is `sentence-transformers/all-MiniLM-L6-v2` (384 dims) and the reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2`.

---

## 5. Baselines (run in-harness, same protocol)

1. **Full-context** — no memory; entire (truncated) history in the prompt. Upper-bound reference at small scale.
2. **Naive vector RAG** — chunk → embed → top-k, no fusion/KG/temporal.
3. **BM25-only** — keyword retrieval, no vectors.
4. **gluomem (default config)** — the full layer.
5. **gluomem (tuned)** — Tier-2-validated tuned config (RQ5).

Vendor systems (mem0, Zep, Letta) are **cited with their reported protocol**, never co-tabulated with the trio-strict numbers (a separate "reported numbers, different protocols" table with explicit tags).

---

## 6. Ablation matrix (config flips — `experiments/ablations.yaml`)

| Axis | Settings | Targets RQ |
|------|----------|-----------|
| Scoring signals | full · −temporal · −popularity · −contextual · −importance · each-alone | RQ2 |
| KG contextual | on · off | RQ3 |
| Frame model | ipb · i_only · overwrite | RQ4 |
| Reranker | on · off | RQ7 |
| Chunking | chunk-level · whole-frame | RQ7 |
| Fusion | RRF · vector-only · BM25-only | RQ1 |
| RRF k | sweep {10,30,60,100} (cheap, via cache) | RQ5 |
| Profiles | balanced/recent/important/connected + tuned | RQ2/RQ5 |
| Embedder | minilm · ollama-nomic | robustness |
| Tuning | default-weights · Tier-1-tuned · LTR | RQ5 |

Each cell is one harness run, one tagged `Score`, committed judgments. Cheap axes (signals, RRF-k, profiles, weights) ride the tuning cache, so they re-run with no model calls.

---

## 7. Experimental protocol

- **Splits.** Per dataset: dev (tuning/labeling) vs test (reporting). Tuning never sees test. Gold labels derived only from dev for tuning; test judged once, committed.
- **Seeds.** All sampling/judge rollouts seeded (`tuning.seed`, judge `rollout_id`s). Report mean over seeds where stochastic.
- **Significance.** Paired bootstrap / paired t-test on per-question correctness for every headline comparison; report CIs. **A sub-1pp difference without significance is reported as "no difference," not a win.** Multiple-comparison correction (Holm) across the ablation matrix.
- **Cost accounting.** `track_usage` logs tokens/cost per run; report total experiment cost and the Tier-1-tuning cost explicitly. Tier-1 tuning re-runs against the candidate cache with no model calls.
- **Judge validation.** Sample N rows for human spot-check of judge agreement; report judge–human agreement to defend the LLM-judge protocol.

---

## 8. Reproducibility artifact (C5)

Committed to the repo:
- `benchmarks_data/` — SHA-pinned datasets + sampling manifest.
- `experiments/` — every run's config (resolved, with overrides), per-row `JudgeRecord` JSONL, `ProtocolTag` manifest.
- `artifacts/` — version-pinned compiled DSPy programs (+ `metadata.json` with dep versions).
- `make reproduce` — re-derives **every table/figure** in the paper from committed judgments with **zero network, zero LLM** (only re-running `bench rescore` + the analysis notebooks). `gluomem bench rescore` verifies dataset SHAs (tamper-evident).
- `make experiments` — the (costed) path that re-runs models from scratch, for full replication.

DSPy programs are compiled with MIPROv2 / BootstrapFewShot and persisted via `save(save_program=True)`. Output schemas use Pydantic types on `dspy.OutputField`; validity retries use `dspy.Refine`. (DSPy 3.x removed `TypedPredictor`, `Assert`, and `Suggest`.)

---

## 9. Threats to validity (pre-registered)

- **Construct** — LLM-judge ≠ truth: mitigated by trio-strict + human spot-check + reporting single-judge inflation.
- **Internal** — confounds from implementation bugs: mitigated by closing the engine correctness list before experiments; ablations are pure config flips on one codebase.
- **External** — small scale (exact KNN ceiling): scoped explicitly; claims limited to ≤~100k-memory / ≤~115k-token regime; ANN scale named as future work.
- **Statistical** — overfitting tuning to dev / many comparisons: train/val/test discipline, paired tests, Holm correction, Tier-2 validation.
- **Benchmark** — LoCoMo saturation: used as anchor only; primary claims on LongMemEval-S + MemBench.

---

## 10. Paper outline (target)

1. Intro — memory mechanisms are bundled & evaluated inconsistently; disentangle + measure honestly.
2. Related work — agentic memory systems, benchmarks, the protocol-incomparability problem.
3. Method — the configurable memory layer (frames I/P/B, hybrid RRF, separable relevance, bitemporal KG, DSPy extraction).
4. Cost-efficient tuning — cache-candidates-then-rescore; two-tier.
5. Evaluation protocol — metric discipline, trio-strict, offline rescore.
6. Experiments — RQ1–RQ7: main results, ablations, tuning, inflation.
7. Discussion — what actually helps; honest-protocol takeaways.
8. Limitations & future work — scale/ANN, RL controller, procedural memory, multimodal.
9. Reproducibility statement — the artifact + `make reproduce`.

---

## References

- **RRF (reciprocal rank fusion)** — Cormack, Clarke & Buettcher, *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*, SIGIR 2009.
- **DSPy** — dspy.ai documentation.
- **sqlite-vec** — github.com/asg017/sqlite-vec.
- **Optuna** (TPE sampler, Hyperband pruner) — optuna.org.
- **pydantic-settings** — docs.pydantic.dev.
- **LoCoMo** — *Evaluating Very Long-Term Conversational Memory of LLM Agents*, arXiv:2402.17753.
- **LongMemEval** — *Benchmarking Chat Assistants on Long-Term Interactive Memory*, arXiv:2410.10813.
- **MemBench** — *MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents*, arXiv:2506.21605 (ACL Findings 2025).
- **MemoryAgentBench** — *Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions* (cite by title/source).
- **ConvoMem** — conversational-memory benchmark (cite by title/source).
- **Anthropic model ids** — platform.claude.com model overview.
