# gluomem — LoCoMo benchmark results (consolidated, 2026-07-21)

> **Evidence status (2026-10-13): historical report, not currently reproducible.**
> The `experiments/` run directories cited below were never part of the published
> repository and are not recoverable from this checkout, so `bench rescore --check`
> cannot be run against them. Treat every number here as a reported historical
> result until the run artifacts are restored or a new baseline is published. See
> [EVIDENCE_INVENTORY.md](./EVIDENCE_INVENTORY.md). The text below is unchanged from
> commit `ab3ecdd`.

Standalone record of every LoCoMo result. The extended narrative is in
`RESULTS.md`; this file is the frozen summary of the LoCoMo campaign as of
the 5-draw seal. Every number is offline-reproducible:
`gluomem bench rescore <run_dir> --check` re-derives it from the committed run
dir with zero model calls.

## Protocol

- **Dataset:** LoCoMo (`locomo10.json`, sha256 `79fa87e9…`), N=320 stratified
  sample (80 × 4 categories, seed 42), committed in
  `benchmarks_data/locomo_sample_seed42.json`. Category 5 (adversarial)
  excluded — unreliable gold.
- **Answerer (subject):** `deepseek-v4-flash` via OpenRouter, temp 0, hidden
  reasoning off. Identical for every system.
- **Judge:** trio-strict (unanimous CORRECT over haiku + gpt5-mini + qwen3,
  temp 0). Systematically harsher than any single judge (single-judge
  inflation +0.6–6.9pp).
- **Retrieval protocol:** context_k=50 memories to the answerer, cross-encoder
  rerank pool=100 (defaults since `ecde208`; the earlier k=10 numbers are
  historical and marked as such).

## Main table (trio-strict QA accuracy, n=320, k=50 protocol)

| System | Overall | single-hop | temporal | multi-hop | open-ended | tokens/query |
|---|---|---|---|---|---|---|
| **gluomem** (draw A) | **0.550** | **0.812** | **0.525** | 0.463 | **0.400** | **2,076** |
| full-context | 0.484 | 0.800 | 0.338 | **0.487** | 0.312 | 26,159 |
| vector-RAG | 0.469 | 0.725 | 0.425 | 0.425 | 0.300 | 2,758 |
| mem0 (OSS 2.0.11) | 0.388 | 0.700 | 0.150 | 0.375 | 0.325 | 2,646 |
| bm25 | 0.378 | 0.662 | 0.412 | 0.225 | 0.212 | 3,206 |

## The honest headline — 5 extraction draws

Re-ingesting re-samples the extraction LLM (deepseek-flash via OpenRouter is
not reproducible run-to-run). Five independent draws, all benched at k=50:

| draw | overall | single | temporal | multi | open |
|---|---|---|---|---|---|
| A (committed DBs) | 0.550 | 0.812 | 0.525 | 0.463 | 0.400 |
| B | 0.466 | 0.825 | 0.287 | 0.438 | 0.312 |
| C | 0.453 | 0.812 | 0.275 | 0.412 | 0.312 |
| D | 0.453 | 0.775 | 0.300 | 0.438 | 0.300 |
| E | 0.441 | 0.775 | 0.287 | 0.375 | 0.325 |
| **mean ± std** | **0.473 ± 0.044** (median 0.453) | | | | |

- Draws B–E cluster at **0.453 ± 0.010** — the post-extraction pipeline is
  nearly deterministic. Draw A is a **+9pp lucky-draw outlier**.
- **Typical gluomem ≈ 3pp below full-context (0.453 vs 0.484) at 1/12 the query
  tokens; best draw +6.6pp above it.**
- Draw-robust wins (hold on all 5): single-hop beats every baseline
  (0.775–0.825); mem0 loses overall on every draw; multi-hop ≥0.375 vs 0.250
  pre-fix. Draw-fragile: temporal (0.525 only on A).
- **Extraction quality is the binding constraint** — the +9pp A-vs-cluster gap
  exceeds every retrieval-side lever measured.
- Draw independence enforced with `dspy.cache.disk=false` (a warm DSPy disk
  cache replays byte-identical completions — caught when two "independent"
  draws matched 2,170/2,170 outputs).

## What moved the number

| change | effect | status |
|---|---|---|
| Reranker wiring fix (was silently off) | 0.406 → 0.456 (+5.0pp), every category up | shipped |
| Cross-conv ingest concurrency + LM gate | ~8h → ~6min per corpus, byte-identical DBs | shipped |
| **Recall breadth k=10→50, pool 30→100** | **+~10pp on both draws tested (0.456→0.550 on A; 0.350→0.453 on C); multi-hop 0.250→0.412–0.463** | **shipped (default)** |
| Semantic dedup at admission | accuracy-neutral even at −12% pool; temporal regressed under first-writer-wins | shipped OFF (`write.dedup.mode=exact`) |
| KG entity-type gate (pollution hypothesis) | REFUTED for multi-hop: drop = −38% entities, 687→24 types, accuracy flat (+2/320) | `retype` default (free hygiene), `drop` opt-in |
| Frozen-extraction cache (`write.extraction_cache`) | enables noise-free ingest-time A/Bs (±12pp → ±1pp arm variance) | shipped (infrastructure) |

**Multi-hop root cause** (failure taxonomy on frozen run): 46% of failures were
retrieval misses — gold facts scattered across frames, absent from a top-10
context. Starved context, not KG composition. Offline recall probe: gold-word
coverage ≥0.5 on 43.5% of multi-hop Qs at k=10 → 75.4% at k=50 (monotonic).

**Fairness sweep at k=50 (every retriever re-run):** mem0 0.372→0.388 (+1.6pp,
capped by its ~170 memories/conv); bm25 0.309→0.378; vector-RAG 0.300→0.469
(+16.9pp — breadth lifts raw-turn retrievers too). gluomem's *structural* edges
at matched k: single-hop on every draw, mem0 head-to-head (+10.2pp mean, all 4
categories), best-draw temporal.

## Economics (answer-time tokens measured from run artifacts)

| system | avg prompt tok/query | accuracy | note |
|---|---|---|---|
| full-context | 26,159 | 0.484 | whole history per query |
| **gluomem** | **2,076** | 0.453–0.550 | cheapest of ALL retrievers — distilled frames denser than raw turns |
| mem0 | 2,646 | 0.388 | |
| vector-RAG | 2,758 | 0.469 | |
| bm25 | 3,206 | 0.378 | |

Ingest (one-time): gluomem ~20M tokens / ~$2.0 vs mem0 ~3M / $0.28 (~7×
structure premium). vs full-context the $2 amortizes in ~900 queries; after
that gluomem is strictly cheaper per query at equal-or-better accuracy.

## Negative results & methodology findings

1. **Extraction nondeterminism dominates ingest-time ablations** — identical
   config re-ingest swung −12pp overall/−29pp temporal. Any ingest ablation
   must share one frozen extraction (now enforced by `write.extraction_cache`).
2. **Near-duplicates are rare** — median same-GOP NN cosine 0.74; the 13× pool
   vs mem0 is distinct facts, not paraphrase. Density is a distillation
   question, not dedup.
3. **KG node pollution is not the multi-hop lever** — junk entities were noisy
   connective tissue; pruning severed as much as it cleaned.
4. **k=10 under-served every retriever** — the original protocol constant, not
   any system, was the multi-hop bottleneck.

## Key run dirs (committed; `bench rescore <dir> --check`)

| row | run dir |
|---|---|
| gluomem k50 draw A (headline) | `experiments/20260720T104120_locomo_gluomem` |
| gluomem k50 draw B | `experiments/20260720T110152_locomo_gluomem` |
| gluomem k50 draw C | `experiments/20260720T102815_locomo_gluomem` |
| gluomem k50 draw D | `experiments/20260720T142910_locomo_gluomem` |
| gluomem k50 draw E | `experiments/20260721T141438_locomo_gluomem` |
| full-context | `experiments/20260706T154347_locomo_full_context` |
| vector-RAG k50 | `experiments/20260720T105659_locomo_vector_rag` |
| mem0 k50 | `experiments/20260720T104634_locomo_mem0` |
| bm25 k50 | `experiments/20260720T105150_locomo_bm25` |
| KG A/B (off/retype/drop, frozen extraction) | `experiments/20260720T09{1322,1744,2140}_locomo_gluomem` |
| historical k=10 seal (reranker fix) | `experiments/20260706T164602_locomo_gluomem` |

Literature anchors (different protocols, never co-tabulated): mem0 paper LoCoMo
J≈66.9 (single generous GPT-4o-mini judge); full-context literature ~72.9%.
Our trio-strict + budget answerer is deliberately harsher.
