# Offline tuning and multilingual benchmark corrections

## Observable contracts

- Runtime search and cached rescoring use the same scoped, logical-clock candidate pipeline and scoring.
- Repeated seeded tuning is offline and deterministic; promotion cannot regress on held-out queries.
- SR/EN retrieval discriminates evidence from distractors; MINJA reports admission, quarantine leakage, persistence and benign false positives honestly.

## Owned paths

`gluomem/tune/**`, `gluomem/search/**`, `gluomem/bench/{adversarial,replay,registry}.py`, a shared bench helper, generated bench fixtures, `InjectionScanner` only in `gluomem/write/gate.py`, safety patterns and scoring field types in `gluomem/config/app_config.py`, tune/ablate CLI sections and stale module docstring, and corresponding tune/search/bench/scanner tests.

## Delete before reshape

Delete copied cache retrieval/time-filter helpers, copied rescore signals, redundant grade-derived qrels, Hyperband pruning, trial re-evaluation, ablation weight renormalization, tautological fake-backend hit assertion, flaky five-second assertion, and three duplicated tamper tests.

Replace the duplicated tamper tests with one parameterized generated-fixture pin contract. Add runtime/cache/scalar/vectorized parity, deterministic split/promotion, SQLite cross-script morphology and gibberish-negative replay contracts.

## Decisions

No dependencies, storage schema, CI, deploy or stream changes. Query holdout is within-dataset evidence, not independent-history generalization. Ambiguous parking rules are benign unless the object names instructions, context, prompts or assistant-specific rules. Held-out attacks will not be used to tune scanner patterns. The bench CLI section is outside ownership: no adversarial subcommand will be added.

## Verification

`uv run ruff check . && uv run pyrefly check && uv run pytest -q`; repeat seeded CLI tuning; measure scalar/vectorized rescore over a fixed synthetic cache and report actual speedup; `org dev census --diff 30b7d4a` and `org dev perf` if available. Commit locally; no push or PR from this subagent.

## Results

Run in `/Users/pavluhin/Documents/Projects/Work/egzakta/GluoMem-wt-tune`:

```text
uv run ruff check . && uv run pyrefly check && uv run pytest -q
All checks passed!
INFO 0 errors (53 suppressed, 130 warnings not shown)
401 passed, 1 skipped, 4 deselected, 1 warning in 18.80s
```

The warning is the existing Starlette/httpx deprecation. Intermediate verification caught an invalid test status (`deleted` instead of `forgotten`) and an incorrect expected persistence numerator (8 instead of the measured 7); both test errors were corrected, without fitting scanner patterns to held-out attacks.

Seeded CLI verification used the eight-query synthetic cache from `tests/tune/test_optuna_tune.py`, saved at `/tmp/gluomem-tune-JpIwOL/cache`. This exact command ran twice (changing only `--out` to `run-2`):

```text
uv run gluomem tune --cache /tmp/gluomem-tune-JpIwOL/cache --out /tmp/gluomem-tune-JpIwOL/run-1 --trials 25 --k 1 --seed 7 --override scoring.profiles.balanced.weights.temporal=0.85 --override scoring.profiles.balanced.weights.popularity=0.05 --override scoring.profiles.balanced.weights.contextual=0.05 --override scoring.profiles.balanced.weights.importance=0.05 --override scoring.half_life_days=7 --override scoring.flat_boost_days=2
tuned 25 trial(s) on 8 query(s): incumbent ndcg@1=0.0000 -> best=1.0000 (+1.0000, trial #1, model_calls=0)
diff /tmp/gluomem-tune-JpIwOL/stdout-1 /tmp/gluomem-tune-JpIwOL/stdout-2
diff -r /tmp/gluomem-tune-JpIwOL/run-1 /tmp/gluomem-tune-JpIwOL/run-2
(both comparisons exited 0, no output)
```

All three artifacts were byte-identical. The manifest hash was `2410dd4182293cf36650579932d4e710f76d40ae3546f42e69a2878ebdd9be8b`.

Rescoring 1,000 trials × 500 queries × 100 candidates with identical ranks/features: original scalar loop **41.355778 s**, cached NumPy batch ranking **4.285160 s**, **9.65× faster**. Array preparation happened once, outside trial timing.

```text
org dev census --repo /Users/pavluhin/Documents/Projects/Work/egzakta/GluoMem-wt-tune --diff 30b7d4a
files=0, functions=0, loc=0, src loc=0, all deltas=0
org dev perf tune/rescore
unknown surface tune/rescore; known: search, chat, ingest, acl, connector
org dev perf search --repo /Users/pavluhin/Documents/Projects/Work/egzakta/GluoMem-wt-tune --repeats 1 --users 1 --seconds 1
no scenario loadtest/users/search.py — add a Locust HttpUser for this surface
```

The organization tools target KVARK/TypeScript, not this Python repository. A scoped Python physical-line/AST census against `30b7d4a` reports source LOC **21,962 → 21,885 (-77)**, test LOC **9,461 → 9,823 (+362)**, test functions **218 → 235 (+17)**. Positive test and fixture deltas implement scoped parity, deterministic promotion, negative retrieval controls, literal marker matching, false-positive controls, and the requested ≥50-event attack denominator.

With the mock embedder and SQLite:

| Observable | Original fixture | Strengthened fixture |
| --- | ---: | ---: |
| SR/EN evidence hit@1 | 33.33% | 50% |
| SR/EN evidence hit@3 | 100% | 100% |
| SR/EN gibberish evidence hit@1 / @3 | vacuous @5 control previously reproduced | 16.67% / 33.33% |
| MINJA poison admission | 0% (5 attacks) | 13.33% (8/60 attacks) |
| MINJA quarantined recall leak@5 | not separately counted | 0% (0/52 quarantined attacks) |
| MINJA unquarantined persistence@5 | not separately counted | 11.67% (7/60 attacks) |
| MINJA poison recall leak@5 (scenario denominator) | 0% | 50% (4/8 scenarios) |
| MINJA clean retention@5 | 100% | 100% |
| MINJA benign quarantine | not counted | 0% (0/160 benign events) |

Held-out MINJA admission is **40% (8/20)**; development admission is **0% (0/40)**. These are authored contract fixtures, not external benchmark results or independent observations. SR/EN proves cross-script lexical discrimination and English retrieval, not semantic English↔Serbian translation.

## Public contracts and integration requirements

- `HybridSearch.candidates(query, *, scope, now, lane_fetch=None, limit=10, since=None, until=None)` returns `CandidatePool` (`rows`, effective `lane_fetch`, `kg_lane_errors`); each `SearchCandidate` contains the frame and raw lane ranks/signals.
- Cache builders default to the runtime fetch size; the DB builder accepts `scope` and both builders accept `limit`. Benchmark clocks use the latest logical conversation session, not wall time. Raw pools bypass the response abstention gate and do not require a reranker. Qrels are explicit; `CandidateRow.grade` is removed.
- `ScoringConfig.half_life_days` and `flat_boost_days` are floats; fractional config overrides are no longer rounded.
- `TuneConfig.held_out_fraction=0.3` replaces pruning. At least two labeled queries are required. Trial metrics and optimization use development queries; only the development winner is checked on held-out queries. Returned `incumbent_score` / `best_score` are held-out scores. Elapsed timing remains on `TuneResult` but is excluded from deterministic artifacts/stdout.
- CLI adds `--held-out-fraction` and cache `--limit`; removes `--no-prune`. `TuneSpace` owns raw-weight, first-BFS-step and importance bounds.
- `fold_query_text` retains its signature but now returns a literal `to_tsquery` expression. The storage owner must change `PgTsVectorStore.search` to `(to_tsquery('simple', ?) || plainto_tsquery('english', ?))`; its current websearch consumers are incompatible. Update the SQL-wiring test and verify against real PostgreSQL. Existing FTS projections need reindexing after normalization changes.

## Unverified

- The ≤2% unquarantined-persistence target is not met; held-out attacks remain visible and were not used to tune the scanner.
- Semantic English↔Serbian translation recall and real PostgreSQL FTS behavior were not proven.
- No adversarial bench CLI entry was added because that CLI section is outside ownership.
- Organization census/perf tools do not support this repository; the Python census and direct timing are the substitute evidence.
