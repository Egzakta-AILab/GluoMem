# Same-event receipt coherence (explicit opt-in)

Enable only with the existing native source-bound assertion-scope producer:

```yaml
write:
  writer_contract: semantic-state-writer-v4
  semantic_producer_contract: observed-assertion-scope-v4
  source_capture_policy: attributed-interaction
  semantic_reuse_policy: same-event-corroboration-v1
```

Omitting `semantic_reuse_policy` preserves predecessor serialization, task leaves,
menus, cache inputs and behavior. This policy changes neither physical writer
version, database schema, I/P/B meanings, deduplication identity nor model output
contracts. It does not supply extraction, identity or consolidation decisions.

## What changes

A selected eligible ADD can exactly reuse a known equal independent I assertion
whose old receipts would be revoked by an earlier selected correction. Preparation
creates a typed `CurrentSourceSupport`, bound to that selected claim, its original
frame revision, source-only meaning, current literal witness, full configuration,
verdicts and immutable plan observations. The native selected ADD is not rewritten
to NOOP or CANDIDATE, and no new I identity or forward publication handle appears.

Inside the ordinary event transaction, after insertion of the actual source event
and before destructive commands, the writer independently reconstructs each
support command, validates the original binding and source fence, and inserts the
ordinary provenance row through the same production helper used by publication.
`SemanticCommitGuard.attach` mints the genuine receipt from this fresh source.
The normal gate still calls `SemanticStore.require_reuse`; it now sees that actual
surviving receipt. The original claim later consumes its preattached row rather
than duplicating provenance. Final checks require actual matching reuse and a
still-live fresh receipt. Any failure rolls back the entire publication.

A later explicit targeted choice can depend on an earlier selected corroboration
command. Its tokens remain the genuine **frozen observation tokens**, not forged
fresh receipt tokens: the equal meaning is conditionally preserved by the typed
source command. Commit must resolve that promise to the actual new surviving
receipt. No future claim is offered as a publication. Untargeted action feasibility
and explicit menus use the same selected-support and receipt-lifetime semantics;
CANDIDATE cannot borrow the ADD-only corroboration exception.

## Refusals and privacy

The prelude is not available for missing/forged quotes, changed role/scope/act,
unknown legacy meanings, dependent or quarantined frames, review-only input,
blocked sources or suppressed claims. The complete selected batch must preserve
the fresh envelope and supported identity. A later destructive target cannot use
a staged receipt and then revoke its own required source authority. Frozen plans,
configuration, source, command, revision and callback mutations still refuse.
Explicit source withdrawal and hard forgetting remain stronger, durable fences.

Old labels and receipts are removed normally when their source context is erased.
Nothing is reconstructed from an erased envelope or an old diagnostic snapshot.
Fresh support carries its actual source-observation time; withdrawing old support
cannot backdate the new receipt. Logical SQL erasure does not claim secure physical
deletion of WAL, backups, caller archives or provider-retained material.

## Verification boundary

The portable SQLite regressions exercise useful whole-event completion, real
preparation and atomic commit, FTS/evidence recall, idempotence, cross-connection
readback, source withdrawal/forgetting and refusal controls. Their content and
selections are authored fixtures. Embeddings are deterministic test vectors; no
provider generation, semantic ranking quality or live ingestion completion is
established. This opt-in does not grant permission to run a model or replay a
historical workload. See [publication limits](publication-status.md).
