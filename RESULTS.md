# gluomem — LoCoMo results

Every system runs in the **same harness**
under one fixed protocol; every headline is reproducible offline with
`gluomem bench rescore <run_dir> --check` (zero model calls).

## Protocol

- **Dataset:** LoCoMo (`locomo10.json`, sha256 `79fa87e9…`), N=320 stratified
  sample (80 × 4 categories, seed 42) — committed in
  `benchmarks_data/locomo_sample_seed42.json`.
- **Answerer (subject):** `deepseek-v4-flash` via OpenRouter, temperature 0,
  hidden reasoning off. Identical for every retriever.
- **Judge:** trio-strict (unanimous CORRECT) over three independent
  backbones — `haiku` + `gpt5-mini` + `qwen3` — temperature 0, generous
  grading rubric. Single-judge scores recorded alongside (see
  "Self-judge inflation" below).
- **Metric:** LLM-judge QA accuracy. `MetricRegistry` refuses to mix
  protocol tags; every score carries `{metric_family, judge_family,
  backbone, dataset_sha}`.
- Category-5 (adversarial) excluded — LoCoMo gold is unreliable there.

## Main table (trio-strict QA accuracy, n=320, k=50 protocol)

Every ranked-memory retriever hands the answerer its top **context_k=50**
memories (was 10 — see "Recall breadth" below for the change and its A/B);
full-context is k-independent. gluomem row = the committed draw-A DBs; its
extraction-draw error bars are quoted beneath the table.

| System | Overall | single-hop | temporal | multi-hop | open-ended |
|---|---|---|---|---|---|
| **gluomem** (draw A) | **0.550** | **0.812** | **0.525** | 0.463 | **0.400** |
| full-context | 0.484 | 0.800 | 0.338 | **0.487** | 0.312 |
| vector-RAG | 0.469 | 0.725 | 0.425 | 0.425 | 0.300 |
| mem0 (OSS 2.0.11) | 0.388 | 0.700 | 0.150 | 0.375 | 0.325 |
| bm25 | 0.378 | 0.662 | 0.412 | 0.225 | 0.212 |

Bold = per-column leader across systems.

**Extraction-draw error bars (the honest headline — 5 draws).**
Re-ingesting the corpus re-samples the extraction LLM; five independent
extraction draws benched at k=50: **0.550 (A, committed) · 0.466 (B) · 0.453
(C) · 0.453 (D) · 0.441 (E) → mean 0.473 ± 0.044, median 0.453**. The structure
inside the bars matters more than the mean: **draws B–E cluster at 0.453 ±
0.010** (retrieval + judging are near-deterministic once extraction is fixed)
while **draw A sits +9pp above the cluster** — the committed DBs are a lucky
extraction draw, not the typical result. Honest claims: *typical* gluomem is
~3pp below full-context (0.453 vs 0.484) at 1/12 the query tokens; *best-draw*
gluomem leads it by +6.6pp; single-hop beats every baseline on **all five draws**
(0.775–0.825 vs vector-RAG's 0.725) and mem0 loses overall on every draw. The
multi-hop fix is draw-robust (0.375–0.463 on all five vs 0.250 pre-fix);
temporal is the draw-fragile category (0.525 on A, 0.275–0.300 otherwise) —
backdated `created_at` only pays when the draw captured the dated facts.
**Implication: extraction quality is now the binding constraint.** The
A-vs-cluster gap (+9pp from an identical pipeline on a good extraction day)
exceeds every retrieval-side lever measured so far, making extraction
robustness (stronger or ensembled extractor, provider pinning) the top open
lever. Draw independence is enforced by `dspy.cache.disk=false` (a warm DSPy
disk cache otherwise replays byte-identical completions — caught when two
"independent" draws matched 2,170/2,170 outputs). Draws A/B predate the
`retype` default and ran `off`; the paired A/B below measured off-vs-retype
accuracy-neutral, so pooling the five is sound.

gluomem runs its full stack: hybrid BM25+vector RRF → 4-signal scoring →
**cross-encoder rerank** (`ms-marco-MiniLM-L-6-v2`, pool=100).

**Reranker fix (was silently off).** The first sealed gluomem row (0.406) ran
with the cross-encoder reranker never wired: `GluoMemRetriever` resolved the
embedder from config but not the reranker, so `HybridSearch`'s gate
(`rerank.enabled AND reranker is not None`) short-circuited while the
manifest recorded `rerank.enabled: true` — a config that lied. Resolving the
reranker from config (mirroring the embedder) lifts gluomem **+5.0pp overall
(0.406 → 0.456)**, every category up, biggest on the fact-surfacing ones
(single-hop +7.5, open-ended +7.5) — exactly the "right conversation, wrong
fact" failure a cross-encoder exists to fix. Runs now assert
`retriever_info.rerank.active == true` in the manifest so this can't silently
regress. Because rerank replaces `hits[0].final` with a raw logit (scale
≈ [-11, +11]) while `abstain.threshold` is calibrated for `rrf*relevance`,
abstain is disabled for the gluomem bench run — LoCoMo's usable categories carry
no abstention gold, so abstaining is always scored wrong (0/320 abstained,
either way).

**No-rerank control (same protocol, abstain off):** overall 0.406 · single-hop
0.625 · temporal 0.463 · multi-hop 0.250 · open-ended 0.287. Verified 0/320
abstentions in that run, so it is a clean control — the +5.0pp delta is the
pure reranking effect.

## Findings

> Historical note: the findings below were sealed under the earlier **k=10**
> protocol (gluomem 0.456). The k=50 table above supersedes the absolute numbers;
> the reranker delta and the structural observations stand.

**Layered memory vs baselines.** gluomem (0.456) beats both
retrieval baselines (bm25 0.309, vector-RAG 0.300) and the mem0
competitor (0.372) by a clear margin. Full-context (0.484) still leads at
LoCoMo's scale (~26k tokens/conversation fits the prompt), but the gap
narrows to **2.8pp** (was 7.8pp before the reranker fix) — consistent with
the hypothesis that memory systems win as history *grows* beyond the
context window; testing that needs the longer LongMemEval benchmark
(see `LONGMEMEVAL_RESULTS.md`).

**gluomem vs mem0 — gluomem now wins 3 of 4 categories.** With the reranker
active, gluomem leads overall by **+8.4pp** (0.456 vs 0.372) and wins
single-hop (0.700 vs 0.675), temporal (0.488 vs 0.175, **+31.3pp**) and
open-ended (0.363 vs 0.338); mem0 holds only multi-hop (0.300 vs 0.275).
This reverses the pre-rerank picture, where mem0's aggressive distillation
won 3/4 on the strength of a smaller, cleaner memory set — reranking closes
that gap by surfacing the right fact from gluomem's ~13× larger frame pool.
The temporal lead is structural and unchanged by reranking: mem0's
distillation discards session-time grounding, while gluomem's backdated
`created_at` + bitemporal KG preserve it — the category the design targets,
and by the widest margin. The one hold-out, **multi-hop (0.275)**, is where
the KG *should* dominate (composing across entities) and does not — entity
pollution diluting the contextual signal was the leading suspect (tested
in the entity-type ablation below, and refuted).

**Self-judge inflation.** Single-judge over trio-strict, same
predictions: gluomem +0.6–6.9pp (singles 0.46–0.53 vs trio 0.456), mem0
+1.6–4.7pp. The trio-strict headline is systematically lower than any
single self-judge — replicated across systems.

## Efficiency (ingest, all 10 conversations)

| System | Memory objects | Ingest tokens (in+out) | Ingest time | Ingest cost |
|---|---|---|---|---|
| gluomem | ~23,000 frames | ~17M + ~2.9M | ~8 h † | ~$2.0 |
| mem0 | 1,703 memories | ~2.9M + ~0.13M | 52 min | $0.28 |

† The ~8 h reflects the original one-conversation-at-a-time batch. Ingest is
now cross-conversation parallel (`write.conv_parallelism`) under a process-wide
API-concurrency cap (`write.max_concurrent_llm`); token count/cost are
**unchanged** (identical calls, byte-identical DBs — the 0.456 row reproduces),
only wall-time drops.

gluomem's richer structure (raw turns + facts + events + per-speaker
profiles + bitemporal KG + I/P/B correction; ~13× the objects, ~6× the
input tokens, five extraction passes + sequential contradiction checks)
costs ~7× mem0's flat two-pass distillation. The paper's cost/accuracy
question: gluomem's structure (with the reranker surfacing from its larger
pool) buys **+8.4pp overall over mem0 and 3 of 4 categories**, temporal by
+31pp — defensible where time-reasoning matters, though mem0 remains
Pareto-favourable on raw cost/accuracy for simple lookups. Documented
optimization headroom (input-token re-send, sequential contradiction,
semantic dedup of the frame pool) remains. Note: reranking is a
retrieval-time cost only; ingest figures above are unchanged by the fix.

## Semantic dedup ablation + the extraction-variance finding

Tested semantic (paraphrase) dedup (`write.dedup.mode=semantic`, an
embedding-cosine gate at admission, same-GOP scope, events/raw/B excluded).

**Calibration (offline, deterministic — no ingest).** Replaying the cosine gate
over the baseline DBs' own vectors (14,509 eligible fact/profile frames): the
fraction with a same-GOP neighbour ≥ threshold is **0.6% at 0.95, 3.1% at 0.90,
10% at 0.85, 25% at 0.80, 45% at 0.75**; median nearest-neighbour cosine is only
**0.74**. So genuine near-duplicates are **rare** — gluomem's ~13× larger pool vs
mem0 is mostly *distinct* extracted facts, not verbatim restatement. Shrinking the
pool toward mem0's density means merging related-but-distinct facts (low
threshold), not deduping paraphrases. The density gap is a **distillation**
question (what to extract), not a dedup one.

**A/B (matched re-ingested extraction, trio-strict, n=320).**

| dedup mode | frames | sem_deduped | overall | single | temporal | multi | open |
|---|---|---|---|---|---|---|---|
| exact (off) | 22,204 | – | 0.334 | 0.638 | 0.200 | 0.238 | 0.263 |
| semantic 0.90 | 21,925 | 656 | 0.341 | 0.650 | 0.188 | 0.250 | 0.275 |
| semantic 0.80 | 19,477 | 4,445 | 0.341 | 0.663 | 0.163 | 0.263 | 0.275 |

- **Dedup is accuracy-neutral even aggressive.** At 0.80 (−12% of the pool, 4,445
  frames dropped) overall QA is unchanged (+0.6pp, within noise). Single-hop
  edges up monotonically (0.638→0.663) — the hypothesised cleaner-pool benefit,
  but small. So dedup is a **safe optional pool-hygiene / ingest-cost knob**, not
  an accuracy lever.
- **Temporal trends DOWN** under dedup (0.200→0.163), even with events excluded —
  first-writer-wins can discard a more temporally-specific *restatement* of a fact
  ("promoted in June" deduped against an earlier "promoted"). A keep-longest
  variant (not first-wins) would likely fix it.

**The dominant finding — extraction is the largest variance source.** The
committed headline row was **0.456** (overall) / **0.488** (temporal); an
*identical-config re-ingest* (mode=exact) lands at **0.334 / 0.200**. That
**−12pp overall, −29pp temporal** gap is pure extraction nondeterminism
(deepseek-flash via OpenRouter at temperature 0 is not reproducible run-to-run —
provider routing / MoE batching), dwarfing the ≤1pp dedup effect. **Implications:**
(1) absolute gluomem numbers carry ~±6pp extraction error bars — the reranker +5.0pp
delta stands only because it was measured on *fixed* DBs (retrieval-time change);
(2) any clean ingest-time ablation (dedup, frame-model, chunking) MUST freeze/cache
one extraction and share it across arms — re-extracting per arm buries the signal.

**Verdict:** semantic dedup shipped, `mode=exact` default (off). It's a correct,
safe pool-shrink knob (−12% frames, accuracy-neutral) but not a headline lever;
the temporal regression wants keep-longest before it's enabled by default.

## KG entity-type ablation — the first *frozen-extraction* A/B

Acting on the extraction-variance finding, we built a **frozen-extraction cache**
(`write.extraction_cache`, `gluomem/llm/cache.py`): one `record` ingest freezes every
extraction call to a JSONL keyed on the *inputs*; later arms `replay` it, so a
post-extraction ablation is measured against **one shared extraction** instead of a
fresh ±12pp draw per arm. This is the tool the dedup section above called for; here
it gets its first use.

**Hypothesis under test — KG entity pollution.** The committed DBs sprawl
to **687 distinct entity_types over 9,070 entities** (336 singletons; 711 untyped
`unknown` endpoints). Junk nodes (`coffee scene`, `golden retriever puppy`) were the
leading suspect for **multi-hop = 0.275** — they short the BFS *contextual* signal.
A resolution-side gate (`kg.entity_type_mode: off | retype | drop`) over a ~24-type
allowlist: `retype` relabels non-allowlisted types to `other` (graph intact); `drop`
prunes the node AND relations touching any non-allowlisted/untyped endpoint.

**Setup.** One frozen extraction (26,571 calls, sha `1ba94b8f`), replayed into all three
arms. Frame *content* is identical across arms (~22,150 frames each); only the KG
differs — a clean isolation of the contextual signal. Readthrough live-miss impurity
(a few entity-overlap-triggered contradiction pairs shift when the graph changes):
off 0 / retype 804 (3.0%) / drop 644 (2.4%).

| arm | entities | rel'ns | types | unknown | overall | single | temporal | multi | open |
|---|---|---|---|---|---|---|---|---|---|
| off (control) | 9,070 | 17,725 | 687 | 711 | 0.344 | 0.637 | 0.188 | **0.287** | 0.263 |
| retype | 8,526 | 17,615 | 26 | 709 | 0.350 | 0.650 | 0.200 | 0.250 | 0.300 |
| drop | 5,584 | 13,980 | 24 | 0 | 0.350 | 0.625 | **0.225** | 0.263 | 0.287 |

(trio-strict qa_accuracy, N=320; `bench rescore --check` verified all three.)

- **Hypothesis REFUTED — pollution is not the multi-hop lever.** Cleaning the graph does
  not lift multi-hop: off 23/80 → drop 21 → retype 20 (flat-to-slightly-down). The junk
  nodes were evidently (noisy) connective tissue for composition — pruning them severs
  paths as often as it removes noise. **Overall is dead flat: 110→112→112 / 320 (+2 Qs).**
- **The one soft signal is elsewhere:** `drop` nudges **temporal +3.8pp** (15→18/80) —
  a leaner graph helps time-grounded lookups — and open-ended trends up; both within the
  ~2.4% miss impurity + judge noise, so directional at best.
- **The methodology is the real win.** Under the old re-extract-per-arm approach these
  ±2pp KG effects would be invisible beneath the ±12pp extraction swing; sharing one
  frozen extraction lands all three overalls at 0.344/0.350/0.350 and makes the (null)
  KG delta *legible*. First clean ingest-time ablation in the project.

**Verdict:** entity-type gate shipped; after the A/B, `entity_type_mode=retype` was
made the default (accuracy-neutral, collapses the type vocabulary 687→26 with the
graph intact — free hygiene; `drop` stays opt-in). Not an accuracy lever — multi-hop
had a different cause (found next section: recall breadth, not the KG at all). The
frozen-extraction cache is the durable deliverable: every future ingest-time ablation
(frame-model, chunking, dedup-v2) now pins extraction first.

## Recall breadth — context_k=50 + rerank pool=100 (the multi-hop fix)

**Failure taxonomy first (iteration 0, free).** On the frozen-extraction retype run,
the 60 multi-hop failures (of 80) decompose as: **46% retrieval miss** (gold facts
absent from the top-10 given to the answerer), 16% partial enumerations, 12%
gold-present-but-won't-aggregate, **0% truncation**. LoCoMo "multi-hop" is mostly
*aggregation over scattered facts* ("what foods does Audrey like") — the answer lives
in many frames; one query + top-10 surfaces one cluster. The lever is recall breadth,
not chain reasoning and not the KG.

**Offline recall probe (local, zero API).** Gold-word coverage of retrieved context
over the 80 multi-hop Qs: k=10/pool=30 → **43.5%** of questions reach coverage ≥0.5;
k=30/60 → 65.2%; k=50/100 → **75.4%**. Monotonic, no plateau by k=50.

**Paid A/B on fixed DBs (zero extraction noise), trio-strict n=320:**

| config | overall | single | temporal | multi | open |
|---|---|---|---|---|---|
| k=10 pool=30 (old default) | 0.350 | 0.650 | 0.200 | 0.250 | 0.300 |
| k=30 pool=60 | 0.391 | 0.725 | 0.175 | 0.362 | 0.300 |
| **k=50 pool=100 (new default)** | **0.453** | 0.812 | 0.275 | 0.412 | 0.312 |
| k=80 pool=150 | 0.456 | 0.787 | 0.250 | **0.438** | 0.350 |

(draw-C DBs; k=80 trades single/temporal for multi at flat overall → plateau; k=50
chosen: balanced, cheaper, strongest guard categories.) **Transfer:** on the committed
draw-A DBs the same flip lifts 0.456 → **0.550** (+9.4pp; multi-hop 0.275 → 0.463,
+18.8pp) — consistent ~+10pp on both draws, every category up on at least one draw.

**Fairness sweep (everyone gets k=50).** mem0 0.372 → 0.388 (+1.6pp — its ~170
memories/conv cap what breadth can add); bm25 0.309 → 0.378; **vector-RAG 0.300 →
0.469 (+16.9pp)** — breadth is a tide that lifts every raw-turn retriever too. What
survives as gluomem's *structural* edge at matched k: **single-hop** (0.812–0.825 vs
vector-RAG 0.725 on every draw), **mem0 head-to-head** (+10.2pp mean, all four
categories), and **best-draw temporal** (0.525 vs 0.425). Multi-hop lands at parity
with vector-RAG (0.412–0.463 vs 0.425) — the category was a k artifact, not a
structure win: distilled frames only pay there via breadth, same as raw turns.

**Verdict:** `benchmark.answer.context_k=50` + `search.rerank.pool=100` shipped as
defaults. Multi-hop 0.250→0.412/0.463 (cause: starved context, now fixed); the old
k=10 protocol under-served every retriever. Answer-context stays ~1.5k tokens — ~17×
under full-context's ~26k.

## Reproducibility & fairness notes

- Committed run dirs under `experiments/` hold per-row predictions +
  judgments + manifest (config snapshot, dataset sha, model slugs,
  judge-prompt sha, token/cost/time). `bench rescore --check`
  re-derives every headline offline — verified for all five.
- **mem0 fairness:** same extraction backbone (deepseek-flash via
  litellm), qdrant local store (mem0's default — faiss silently disables
  their hybrid scoring), hidden-reasoning disabled to match gluomem's
  policy; everything else mem0 defaults. Versions + effective config in
  `manifest.retriever_info`.
- **CI gate (`evidence_hit@5`) applies only to raw-turn retrievers.**
  gluomem and mem0 return distilled memories (no turn identity), so
  evidence-overlap metrics are ~0 by construction — qa_accuracy is their
  sole headline. The gate validates bm25/vector-RAG.
- Numbers carry run-to-run noise (~2–3pp) from OpenRouter backend and
  judge variance; provider pinning + the committed judgments make any
  single reported table exactly reproducible.

## Literature anchors (different protocols — NOT comparable rows)

- Mem0 paper: LoCoMo J≈66.9 (single GPT-4o-mini judge, generous rubric,
  ~1,540 Qs). Our mem0-OSS 37.2% is trio-strict on N=320 with a
  deepseek-flash subject — a stricter protocol, not the same number.
- Full-context reference (literature): ~72.9%.

These sit here as prose, never co-tabulated with our trio-strict column.
