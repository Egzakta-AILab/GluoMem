# Verification status and limits

This change publishes source-attribution, temporal-state, usable-evidence,
erasure/atomicity and optional typed-writer improvements. It preserves the required
explicit `dspy.model`, upstream retry/output-bound controls, default transport and
legacy opt-in boundaries. It does not certify production deployment or benchmark
readiness.

## What the portable regressions establish

The critical authored tests use real temporary SQLite stores, deterministic mock
embeddings and explicit extractor decisions. They cover scoped source binding,
I/P/B dependency reconstruction, temporal views, stale-plan refusal, duplicate
freshness, reentrant erasure, worker ownership, source-fragment revocation,
whole-event receipt coherence, loaded typed contracts, transient cache privacy and
composed migration behavior. Offline Alembic SQL generation checks the revision
chain and downgrade guards; it is not a live PostgreSQL round trip.

The compact suite intentionally does not publish the historical research test
archive, captured provider requests/responses, operator journals or private
qualification artifacts. All inherited upstream tests remain. Published source and
authored fixtures are portable; a passing targeted subset is not an all-tests,
installed-wheel, complete native-SDK or paired benchmark-harness qualification.

## Static verification

The normal locked Pyrefly error gate is clean after explicit interface narrowing
and typed-fixture repairs. The original model schemas, loaded producer contracts
and legacy capability dispatch were compared independently of that error count.
The optional expanded warning audit is not clean; it is not represented as a
zero-warning result. Locked Ruff lint and formatting checks pass.

## Gates that remain separate

- The final combined library/harness gate and remote CI results must be checked
  against the actual PR heads; an older runtime's pass does not certify merged or
  sanitized bytes.
- Whole-suite collection under the strict local no-child boundary is blocked by
  the inherited documentation test's import-time Git invocation. Focused test
  passes do not establish that the full `make check` gate passed.
- Live PostgreSQL upgrade/rollback, concurrent connections and provider/native-SDK
  compatibility require their own environment and execution evidence.
- No new paid/live model generation, encoder loading, tracking or service activation
  is authorized or performed by these publication checks.
- A prior real-model ingestion campaign failed. A broad real-model QA comparison
  has not been completed successfully; these changes do not convert that failed
  run into a score or demonstrate model-quality improvements.
- Source-bound SDK qualification and the evaluator's origin/catalogue verification
  are still pending. Bounded schemas and retrospective offline mechanics are not
  an operator activation certificate.
- Supported negative reports may retain an old literal as evidence of a past
  mistaken report. Unsupported paraphrases are not made valid by schema acceptance.
  Literal hygiene, supportedness, assertion identity and answer formatting are
  distinct policies; old outputs or grades must not be silently reinterpreted.

Historical/as-of access never overrides hard erasure. Retained-fragment checks do
not establish entailment. Process-local semantic cache invalidation is not a
cross-process service, and logical SQL scrubbing does not erase external copies,
WAL, backups or provider retention.
