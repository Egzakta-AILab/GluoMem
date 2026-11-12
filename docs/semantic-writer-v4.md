# Semantic state writer V4 (explicit opt-in)

This capability retains canonical I/P/B frames and the existing per-claim writer.
It is a **consistency restriction over untrusted source-derived proposals**, not a
semantic entailment checker, entity resolver, new truth store, or quality claim.
Default/legacy, bounded-writer-v2 and action-target-writer-v3 types, instructions,
cache namespaces and saved artifacts are not reinterpreted.

## Configuration and invocation

```yaml
write:
  writer_contract: semantic-state-writer-v4
  source_capture_policy: attributed-interaction  # or trusted-only
  extraction_backend: dspy
  dedup:
    mode: exact
dspy:
  model: my-model  # Required registered model; configure it explicitly.
  adapter: json
  schema_mode: required
  cache:
    disk: false
```

Use `make_extractors(cfg)` and the existing `prepare_source_event` /
`commit_source_event` (or `ingest_source_event`) APIs with an explicit scoped
`SourceEvent`. Extraction, including explicit-memory inputs, must pass through the
V4-capable extractor. A missing capability or missing source policy refuses;
there is no raw-text/legacy-meaning fallback. The inherited task names remain
`event_claims` and `consolidate`. Legacy string-only/context-only consolidation is
unavailable on a V4 facade: use its bound contextual projection method.

The loaded event-claims artifact is finalized with a required `meaning` on every
claim and an actual `source_context` input. V4 consolidation uses
`SemanticClaimContext`, `SemanticStoredCandidate` and `SemanticProjection`.
Invocation-owned copies retain loaded instructions/demos/CoT and configured
output bounds; predictor-bound LMs, sample counts and token caps cannot replace
the task's explicit routing/count policy. Both tasks require single-send retry
policies. No model-generated target is inferred from prose or repaired by code.

## Typed proposal

`gluomem.llm.semantic` is DSPy-free:

- `AssertionKey(subject, predicate, perspective)` identifies an assertion address.
- `ClaimMeaning(key, value_key, independently_versioned, refines)` carries a
  normalized value proposal and an explicit dependency shape. Independently
  versioned state has no anchor; a dependent detail requires one.
- The perspective and anchor perspective must match the **bound source role**.
  This avoids silently treating an assistant statement as a user assertion.
- `SemanticSourceWitness` retains the exact scope/event/revision/source/message,
  role, trusted observation time and literal quote. Receipt revisions also bind
  the complete retained extraction context; the quote is never expanded.

These fields are proposed in the actual model-facing extraction response, before
consolidation selection. Host code never invents them from frame prose, kind,
date, GOP or the selected target. Literal membership and matching labels do not
prove that a quote entails the claim or the proposed labels. Incorrect labels
can still authorize an incorrect semantic relationship; that limitation is
explicitly tested.

## Eligibility and publication

V4 intersects, and never expands, the ordinary evolving action/target menu:

| Action | Additional consistency requirement |
|---|---|
| REFINE | Incoming dependent detail explicitly anchors the independent target key. |
| TRANSITION | Both claims are independently versioned with the same key, plus all existing temporal requirements. |
| SUPERSEDE | Same assertion key, independence flag and dependency anchor. |
| NOOP | The SUPERSEDE shape requirements plus equivalent value identity. |

Unknown legacy target meaning grants **no targeted action**. Independent state
cannot become P merely because its person/topic/kind/date resembles another
claim. A dependent detail without an observed anchor can be explicitly held
CANDIDATE; the writer does not silently choose ADD. Ordinary SUPERSEDE retains the
existing I-replacement codec; no new B action is introduced. Existing B correction
creation, payload, references and reconstruction paths remain unchanged. A stored
B is never made a REFINE/TRANSITION target. A dependent correction receives no
fresh independent-publication handle.

Exact or semantic admission reuse requires a compatible surviving source-bound
meaning. Same text with a different key/value/dependency/perspective, or an
unknown legacy binding, **refuses for review**. It is not silently relabelled,
backfilled, duplicated to defeat deduplication, or given a new conditional handle.
Genuine original handles survive compatible event-local reuse. Suppressed,
quarantined or review-held claims do not gain handles. Existing trust, pin,
verification, expiry, consumed-target, scope and temporal dependency rules remain
in force.

Preparation freezes incoming source/meaning/context and scoped stored semantic
receipts. Commit checks these at entry, around callbacks/admission and at event
end. Only named provenance additions and lifecycle removals advance the
invocation-owned shadow. Metadata cannot be refreshed from an unrestricted
post-callback reread. A forged target meaning/menu, stale token, changed context,
invalid target, erased witness or missing actual publication refuses atomically.
The existing transaction-owner erasure-replay interface is retained.

## Durable schema and erasure

`memory_claim_semantics` is keyed by actual `memory_provenance.id` and references
its frame. It stores the scope, V4 contract, proposal JSON and an opaque random
revision token. It does **not** store content/value/subject hashes or a second
quote store. Receipts reconstruct source witnesses from the existing scoped
provenance and source-event rows. No historical record is automatically converted.

- SQLite schema version **6**; migration
  `storage/migrations/0006_semantic_bindings.sql` follows source-support schema 5.
  Opening a supported scoped schema 1–5 applies each missing migration in order.
  Fresh DDL includes temporal, source-support and semantic tables; verification
  of an existing schema 6 requires the final table columns.
- PostgreSQL metadata and Alembic revision **0006_semantic_bindings** are provided.
  The single revision chain is `0004_source_attribution -> 0005_source_support
  -> 0006_semantic_bindings`, preserving the existing source-retraction revision.
  Populated semantic bindings prohibit PostgreSQL downgrade. Offline DDL
  generation is not a live PostgreSQL upgrade/rollback qualification.
- Frame scrubbing deletes its proposals. Provenance deletion cascades. Full
  source-context redaction deletes proposals backed by that context, including
  those on a surviving sibling; a surviving frame is not silently relabelled.
  This is deliberately conservative: a retained sibling fragment is evidence of
  that exact quote, not permission to recreate an erased semantic proposal.
- `erase_semantic_frames(backend, frame_ids, scope=...)` and
  `erase_semantic_sources(backend, event_ids, scope=...)` are the explicit lifecycle
  hooks. Keep them inside the owner's transaction, before deleting provenance.
  Source withdrawal also invalidates the transient V4 caches.

## Cache privacy boundary

Native DSPy disk caching is refused for V4, including mutated configurations.
The V4 `CachedExtractors` wrapper preserves the capability and validates cached
selections against the exact menu and rationale bound. Its cache is **private to
that wrapper in that process**, keyed with an ephemeral secret, and never reads
or writes a disk transcript. `persistent_cache_supported` is false. A fresh
wrapper/process in strict replay misses explicitly rather than reading a legacy
transcript. Keys bind the contract, method, source context, candidate semantic
receipts, contextual meaning, projection and effective rationale bound.

Erasure/retraction conservatively flushes all process-local V4 wrappers; an
in-flight result cannot repopulate an invalidated generation. This is not a
cross-process cache invalidation service, storage-media erasure, or a promise to
scrub caller-owned request/response archives, provider retention or diagnostic
fixtures. Default/V2/V3 cache behavior remains unchanged.

## Verification boundary

Portable authored tests exercise proposal types, loaded artifacts, durable commit,
current/as-of/known-at views, independent state, anchored detail, correction,
source-role witnesses, NOOP/reuse, withdrawal, hard forgetting and transient cache
invalidation. Deterministic embeddings and authored decisions are substitutions,
not semantic retrieval measurements or evidence of model label reliability.

A supported negative report may legitimately contain the previous literal value.
Requiring that literal to disappear from every eligible report is a stronger,
separate policy; the engine does not remove or rewrite supported evidence to make
such a diagnostic pass. See [publication limits](publication-status.md).
