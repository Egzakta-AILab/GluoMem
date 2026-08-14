# gluomem — LongMemEval results (sealed 2026-07-25)

Standalone record of the LongMemEval campaign, mirroring `LOCOMO_RESULTS.md`.
Every number is offline-reproducible: `gluomem bench rescore <run_dir> --check`
re-derives it from the committed run dir with zero model calls.

## Why this benchmark

LoCoMo's conversations run ~18k tokens each, well inside any answerer's
context window, so "read everything" is always a live option there.
LongMemEval-S is the test the thesis actually needs: histories that do not
fit a prompt at all, so a structured memory either earns its keep or it
doesn't.

## Protocol

- **Dataset:** LongMemEval (arXiv:2410.10813), "cleaned" HF release
  (`xiaowu0162/longmemeval-cleaned`), sha-pinned. Two variants:
  - **oracle** — evidence-only sessions per question (the retrieval ceiling),
    all 500 questions.
  - **S** — full haystacks, ~48 sessions / ~494 turns / ~122k tokens per
    question, 10-day span. 105-question stratified sample (15 × 7 ability
    types), seed 42, committed in
    `benchmarks_data/longmemeval_s_sample_seed42.json`.
- **Answerer, judge, embedder, reranker:** identical to LoCoMo (see
  `LOCOMO_RESULTS.md`), plus per-question-type judge rubrics
  (`JudgeQATyped`) transcribed from the official `evaluate_qa.py` semantics
  (temporal off-by-one-day forgiveness, knowledge-update accepts old+new,
  abstention correct iff the prediction declines).
- **Scale, for context:** LoCoMo averages 27 sessions / 588 turns / ~18.2k
  tokens per conversation. LongMemEval-oracle averages 1.9 sessions / 22
  turns / ~6.6k tokens (the easy end). LongMemEval-S averages 48 sessions /
  494 turns / ~122k tokens, roughly 7× LoCoMo's length and well past any
  answerer's practical context budget.

## Oracle (n=500, ceiling reading)

| System | Overall | ss-user | ss-asst | ss-pref | multi | temporal | k-update | abstain |
|---|---|---|---|---|---|---|---|---|
| **gluomem** | **0.662** | 0.922 | 0.929 | 0.233 | **0.628** | **0.449** | 0.819 | 0.700 |
| bm25 | 0.570 | **0.938** | 0.946 | 0.000 | 0.339 | 0.402 | 0.819 | 0.700 |
| vector-RAG | 0.506 | 0.859 | 0.911 | 0.033 | 0.314 | 0.331 | 0.625 | **0.733** |
| full-context | 0.480 | 0.578 | **0.946** | 0.033 | 0.306 | 0.354 | 0.639 | 0.700 |
| mem0 | 0.246 | 0.000 | 0.054 | 0.100 | 0.207 | 0.220 | 0.667 | 0.533 |

Even with the ceiling handed to every system (only the sessions that
contain the answer), full-context loses to gluomem by 18pp: attention
dilution on a budget answerer, not a truncation story. gluomem beats mem0
head-to-head on all seven ability types.

## S (n=105, the real test)

| System | Overall | ss-user | ss-asst | ss-pref | multi | temporal | k-update | abstain |
|---|---|---|---|---|---|---|---|---|
| **gluomem** | **0.676** | **1.000** | **0.933** | 0.267 | **0.667** | 0.400 | 0.733 | 0.733 |
| bm25 | 0.562 | 0.933 | 0.867 | 0.000 | 0.200 | **0.400** | 0.667 | 0.867 |
| mem0 | 0.533 | 0.800 | 0.467 | 0.067 | 0.400 | 0.333 | **0.733** | **0.933** |
| vector-RAG | 0.495 | 0.933 | 0.867 | 0.067 | 0.200 | 0.267 | 0.400 | 0.733 |
| full-context | 0.152 | 0.000 | 0.067 | 0.000 | 0.000 | 0.000 | 0.000 | **1.000** |

full-context collapses: it wins only abstention, and that is trivial (a
truncated history genuinely contains nothing, so declining is correct there
by accident). gluomem beats mem0 on five categories outright, ties on
knowledge-update, loses only abstention. Multi-session is the clearest
transfer of the LoCoMo recall-breadth fix: 0.667 here, 0.628 on oracle, both
field-leading, on a config never re-tuned for this dataset.

## A data-integrity incident, caught and fixed before sealing

While preparing worked examples for this report, one conversation's memory
looked implausibly thin (124 extracted frames against a ~122k-token
haystack). Auditing all 105 conversations by comparing distinct
`{conv_id}:session:N` GOPs against the true session count in the source
data found **16 of 105 (15%) silently incomplete** — 2 to 47 of their 44-51
sessions, missing between 1 and 44 sessions each.

**Root cause:** a multi-hour ingest was interrupted mid-run (a machine
reboot). `ingest_per_conversation`'s resume path (`--skip-existing`) checks
only whether a conversation's `.mind` file exists, not whether it is
complete — a `.mind` file created right before the interruption reads as
"already done" on resume and is never revisited. This is an ingest
robustness gap, not a scoring or protocol bug; it produced a real, silent
undercount. Resume has since been hardened to verify completeness, not just
existence (completion sentinels — exercised below).

**Fix applied:** the 16 partial `.mind` files were deleted and
re-ingested via `write.extraction_cache.mode=readthrough` against the
frozen S extraction cache (cache-eligible content replayed free; the missing
sessions synthesized live, ~$1.87, 21,992 live calls). All 105 conversations
were re-verified complete before gluomem was re-benched.

**Effect on the headline: it moved up, not down.** 0.657 → **0.676**
overall; multi-session 0.600 → 0.667; single-session-preference 0.200 →
0.267. The bug was understating gluomem, not flattering it. No other system's
numbers are affected (bm25, vector-RAG, full-context and mem0 each build
their own index directly from the dataset or their own ingest, independent
of gluomem's `.mind` files).

## Two findings from the worked examples

**A genuine contradiction-handling gap.** On a knowledge-update question
("what was my previous 5K personal best?"), gluomem's memory kept an old value
(26:30) and its update (27:45, the gold answer) as two equally-live facts
instead of retiring the old one — the entity-overlap contradiction trigger
never fired on this pair. The answerer saw four near-duplicate memories and
picked from the wrong side. BM25 and mem0 got this one right, by retrieval
luck rather than design. This is a structural gap in same-subject detection
and the contradiction-trigger mechanism — LongMemEval's knowledge-update
category is a concrete test case for it (measured directly in the
contradiction-trigger ablation below).

**Preference grading is stricter than "topically correct."** After the
data-integrity fix, gluomem's retrieval for a preference question (cookie
recipe, gold references the user's prior turbinado-sugar experiment) now
surfaces the exact right memory, and the answerer's response substantively
names turbinado sugar and its flavor profile. Two of three trio judges still
marked it wrong: the response phrased the ingredient as a new suggestion
rather than explicitly acknowledging the user's own prior experimentation.
Retrieval and fact recall are not the bottleneck on this category; the
answerer prompt does not require "build on what the user already tried" as
distinct from "use the context." **Follow-up (2026-07-27): built as
`benchmark.answer.style: base | preference_aware`** — the variant prompt
instructs the answerer to acknowledge the user's own prior
attempts/preferences instead of re-suggesting them (recorded in the run
manifest's `answer.style`); `base` keeps the sealed prompt byte-identical.
The A/B is in the next section.

## Follow-up ablations (2026-07-27)

Both findings above became config knobs (commit `09e6edf`) and were then
measured. All six run dirs below are committed and `rescore --check`
verified.

**Preference-aware answering: oracle pref doubled.**
`benchmark.answer.style=preference_aware` adds one instruction: acknowledge
the user's own prior attempts/preferences instead of re-suggesting them.
A/B on the pref-category subsets (base arms reproduced the sealed numbers
exactly):

| arm | base | preference_aware |
|---|---|---|
| oracle pref (n=30) | 0.233 | **0.467** (10 fixed, 3 broke) |
| S pref (n=15) | 0.267 | 0.267 (2 fixed, 2 broke) |

The mechanism is visible verbatim: on the same question the base answer
said "the context does not mention specific accessories" while the
preference-aware answer opens "Based on your history, you use a Sony A7R IV
and have already upgraded to a Godox V1 flash" and then recommends. At
S-scale the prompt cannot fix what retrieval did not surface — pref at S
stays retrieval-bound, consistent with the finding above.

**Promoted to default (2026-07-29)** after a full collateral-damage check:
oracle-500 with the preference-aware prompt scored 0.670 overall, above
all three control noise draws (0.650/0.658/0.660); pref replicated at
0.467 (+24.4pp vs the pooled control) and no other category moved beyond
the observed per-draw noise band (temporal +5.2pp, knowledge-update
−5.6pp, rest ±3pp). S-105: 0.667 vs 0.676, one question of pure churn,
pref unchanged. `benchmark.answer.style` now defaults to
`preference_aware`; `base` reproduces every pre-promotion sealed run and
the style is recorded in each run manifest. Check runs:
`experiments/20260729T075512_longmemeval_oracle_gluomem`,
`experiments/20260729T080735_longmemeval_s_gluomem`.

**Contradiction trigger: +1.8pp overall, +4.2pp knowledge-update.**
Paired frozen-extraction A/B on oracle-500 (both arms rebuilt from the
recorded extraction cache; only the trigger config differs):
`write.subject.trigger=both` + two data-derived predicate families
(residence: LIVES_IN/LIVED_IN/RESIDES_IN/MOVED_TO/RELOCATED_TO/BASED_IN;
employment: WORKS_AT/WORKS_FOR/WORKS_IN/WORKS_AS/WORKED_*/EMPLOYED_*) +
`write.contradiction.top_k=2`.

| scope | control | trigger | delta |
|---|---|---|---|
| overall (n=500) | 0.650 | **0.668** | +0.018 |
| knowledge-update (n=72) | 0.792 | **0.833** | +0.042 |
| multi-session (n=121) | 0.595 | 0.628 | +0.033 |
| abstention (n=30) | 0.700 | 0.733 | +0.033 |
| single-session-preference (n=30) | 0.233 | 0.267 | +0.033 |
| temporal-reasoning (n=127) | 0.457 | 0.465 | +0.008 |
| single-session-user (n=64) | 0.922 | 0.922 | 0.000 |
| single-session-assistant (n=56) | 0.911 | 0.893 | -0.018 |

Ingest-side effect: B-frames ~2.3× (≈375 → 872), retired edges ~2.9×
(≈1,120 → 3,214), P-frames flat — the trigger fires on updates without the
P-inflation the threshold=1 experiment showed, because candidate nomination
is decoupled from P-attachment. Per-question: 23 fixed, 14 broke, net +9.
(Control 0.650 vs the sealed 0.662 is rebuild + answer/judge
nondeterminism; the paired comparison shares one extraction.)

**Promotion campaign (2026-07-28/29): the single-draw gain did not
survive replication — trigger stays OFF by default.** Three follow-up
measurements, all committed and rescore-verified:

- *Oracle noise draws* (3 benches per arm, dspy disk-cache off so provider
  noise is real): pooled overall 0.656 vs 0.661 (+0.5pp, within noise);
  knowledge-update +1.9pp (the +4.2pp above was single-draw luck); only 2
  questions consistently fixed across all draws vs 4 consistently broken.
- *LoCoMo regression* (paired readthrough rebuild on the frozen LoCoMo
  cache, backdated, N=320): 0.544 vs 0.538 (−0.6pp); single-hop and
  multi-hop byte-identical. No damage, but no gain either.
- *S-105 confirmation* (paired ingest on the frozen S cache; 60,711 live
  contradiction calls, 3,128 B-frames, 12,130 retired edges): **0.676 vs
  0.638 (−3.8pp)**. Knowledge-update, the target category, went DOWN
  (0.733 → 0.667), and single-session-preference collapsed (0.267 → 0.067,
  4 broke / 1 fixed). Only multi-session improved (+6.7pp).

Reading: the mechanism fires exactly as designed, but at haystack scale
the extra B-frame text and mass edge retirement pollute the retrieval pool
— oracle's evidence-only contexts masked the cost. Honest outcome: a
structurally correct contradiction trigger whose answer-time effect is
noise at oracle scale and net-negative at S scale. The knob ships off;
the negative result is the finding.

**A second resume-integrity face, caught by the completion sentinels.** Rebuilding
the control arm under strict `replay`, 16/500 conversations raised on
`check_contradiction` cache misses: the original recorded run went through
interrupted/resumed trajectories, so a CLEAN ingest produces (observation,
base) contradiction pairs the cache never saw. The new completion sentinels
refused the 16 partial `.mind` files instead of letting them be scored —
the exact failure class of the S incident, this time caught at ingest time.
Repaired via readthrough (misses served live); `bench verify` then gated
the dir at 500 complete / 0 incomplete before any benching.

## Reproducibility

Committed run dirs (`bench rescore <dir> --check` reproduces each row):

| row | run dir |
|---|---|
| gluomem oracle | `experiments/20260721T224430_longmemeval_oracle_gluomem` |
| full-context oracle | `experiments/20260722T072611_longmemeval_oracle_full_context` |
| bm25 oracle | `experiments/20260722T074437_longmemeval_oracle_bm25` |
| vector-RAG oracle | `experiments/20260722T075757_longmemeval_oracle_vector_rag` |
| mem0 oracle | `experiments/20260722T085051_longmemeval_oracle_mem0` |
| gluomem S-105 (corrected) | `experiments/20260725T140047_longmemeval_s_gluomem` |
| full-context S-105 | `experiments/20260724T183931_longmemeval_s_full_context` |
| bm25 S-105 | `experiments/20260724T184450_longmemeval_s_bm25` |
| vector-RAG S-105 | `experiments/20260724T185606_longmemeval_s_vector_rag` |
| mem0 S-105 | `experiments/20260724T185803_longmemeval_s_mem0` |

The pre-fix gluomem S-105 run (`experiments/20260724T183654_longmemeval_s_gluomem`,
overall 0.657) is superseded by the corrected run above and was never
committed.

Follow-up ablation run dirs (2026-07-27); pref sample subsets committed as
`benchmarks_data/longmemeval_oracle_pref30.json` /
`benchmarks_data/longmemeval_s_pref15.json`:

| row | run dir |
|---|---|
| S pref-15, style=base rerun | `experiments/20260727T081854_longmemeval_s_gluomem` |
| S pref-15, preference_aware | `experiments/20260727T081924_longmemeval_s_gluomem` |
| oracle pref-30, style=base rerun | `experiments/20260727T083719_longmemeval_oracle_gluomem` |
| oracle pref-30, preference_aware | `experiments/20260727T083832_longmemeval_oracle_gluomem` |
| trigger ablation, control | `experiments/20260727T142452_longmemeval_oracle_gluomem` |
| trigger ablation, treatment | `experiments/20260727T143314_longmemeval_oracle_gluomem` |

Promotion-campaign run dirs (2026-07-28/29):

| row | run dir |
|---|---|
| LoCoMo regression, control | `experiments/20260728T005258_locomo_gluomem` |
| LoCoMo regression, trigger | `experiments/20260728T010111_locomo_gluomem` |
| oracle noise draw 1, control | `experiments/20260728T010558_longmemeval_oracle_gluomem` |
| oracle noise draw 1, trigger | `experiments/20260728T011914_longmemeval_oracle_gluomem` |
| oracle noise draw 2, control | `experiments/20260728T013324_longmemeval_oracle_gluomem` |
| oracle noise draw 2, trigger | `experiments/20260728T014700_longmemeval_oracle_gluomem` |
| S-105 trigger arm | `experiments/20260729T033957_longmemeval_s_gluomem` |
| S-105 control re-bench | `experiments/20260729T034202_longmemeval_s_gluomem` |
