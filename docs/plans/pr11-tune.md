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
