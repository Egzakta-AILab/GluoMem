# Source-bound writes and usable recall

I is self-contained state, P refines its original base, B is a correction, and a
GOP groups a topic/session. An ordinary dated change is not automatically a
correction: two independent I frames can remain valid at different effective
times. A refinement is never silently rebased onto a newer root.

## Source capture and publication

`SourceEvent` and `SourceEventMessage` carry the scoped envelope. Select
`write.source_capture_policy` explicitly for strict attribution:

- `trusted-only` excludes assistant messages from claim support.
- `attributed-interaction` permits assistant evidence but preserves its assistant
  role; it does not turn it into a user statement or verified world fact.
- `None` preserves the legacy envelope protocol.

A strict extracted claim must name its actual source message and a literal
supporting span in that message. Unknown IDs, a span from another message and
ambiguous identities refuse; no singleton-message fallback grants authority.
Trusted observation time records when the source was learned. An absent explicit
claim date does not inherit the source clock as world-effective validity.

`prepare_source_event(event, store=..., cfg=..., extractors=...)` freezes scoped
inputs, candidate revisions, policy and selected operations without publishing.
`commit_source_event(plan, store=...)` validates those observations and publishes
the event, frames, provenance, lineage and indexes together. `ingest_source_event`
combines the two. Callbacks cannot silently refresh authority or replace an
unobserved target. A stale plan refuses instead of retargeting or falling back to
ADD. Duplicate responses are reloaded against current rows after erasure, not
served from a prepared plaintext snapshot. Reentrant forget/retraction remains
authoritative even when the attempted publication refuses; the worker commits
that outcome before recording its categorical job failure.

## Current and historical usable evidence

`HybridSearch.search` is raw ranked search. `HybridSearch.recall`,
`EmbeddedMemoryClient.recall`, and `/v1/memories:recall` return dependency-complete
usable evidence. Recall validates eligible I/P/B dependencies and typed,
reciprocal, scoped correction lineage. A superseded predecessor may identify a
correction relationship, but its scrubbed plaintext is never returned as current
support. `groups` and escaped single-line JSON evidence preserve the complete
family for downstream line-based packing; `k` limits matched seeds, not the
required expanded frame count.

Temporal metadata is managed explicitly by `TemporalStore`, including immutable
`TemporalObservation(observed_at=..., effective_at=...)` and explicit independent
state transitions. Chronology, not numeric ID order, determines the current state.
An unavailable latest version does not automatically revive an older value.

`as_of` selects world-effective history; `known_at` restricts what had been learned.
Both require host-supplied `TemporalPolicy(allow_historical=True)` on the embedded
client/search or service's injected search instance. Request fields alone do not
grant history access. The host clock controls authorization/eligibility; a logical
search clock cannot undo forgetting or expiry. Later refinements and provenance
are filtered too. Unmanaged timestamps remain unknown. Raw SQL/export is not an
authorized historical-evidence API.

Recall uses bounded lane continuation so early temporal misses need not consume
all usable slots. `candidate_budget` and `evidence_frame_budget` bound scanning
and expansion; `retrieval` reports quota satisfaction, lane exhaustion and budget
exhaustion separately. A capped scan is not proof that a lane is exhausted.

## Source withdrawal and hard forgetting

Ordinary supersession can retain an unchanged sibling's authenticated disjoint
supporting fragment, without making a second quote store. See
[source support](source-support.md) for offsets, binding checks and conservative
revocation. It cannot recreate erased semantic labels. Hard forget, scope forget,
dismissal and source withdrawal remain stronger than retained-fragment reuse.
Explicit B payload references and temporal/version dependencies participate in
erasure closure; unrelated independent roots survive. Compaction does not treat
different constituent sources as independent support for an entire aggregate or
promote mixed-policy fragments.

These are logical application/SQL guarantees, not secure deletion of WAL, storage
media, backups, caller archives or provider-retained material.

## Optional writer contracts

All require an explicit source-capture policy. Select capabilities deliberately;
none supplies generation-model selection or permission to send requests.

| Setting | Contract |
| --- | --- |
| `writer_contract: bounded-writer-v2` | Source-bound writer and legal evolving per-claim projection. |
| `writer_contract: action-target-writer-v3` | Explicit action/target pair in the native schema and parser. |
| `writer_contract: semantic-state-writer-v4` | Source-bound typed assertion/dependency proposals, semantic receipt guards, invocation-private cache. |
| `semantic_producer_contract: observed-assertion-date-v1` | Explicit observed identity and required-nullable claim date. |
| `semantic_producer_contract: observed-assertion-date-v2` | Also withholds destructive choices that revoke an earlier transition's authority. |
| `semantic_producer_contract: observed-property-identity-v3` | Separates property identity, fallible aliases and assertion relation from values/dates. |
| `semantic_producer_contract: observed-assertion-scope-v4` | Distinguishes direct assertion, prior-mistake report and unresolved assertion scope. |
| `semantic_reuse_policy: same-event-corroboration-v1` | Opt-in source-backed exact reuse that survives an earlier correction in the same atomic event. Requires assertion-scope V4. |

Start with [semantic writer configuration](semantic-writer-v4.md), then
[preparation V2](observed-preparation-v2.md), [property identity
V3](observed-property-identity-v3.md) and [receipt coherence](receipt-coherence.md).
Proposed labels and literal membership are consistency checks, not semantic
entailment or certified identity. Schema constraints do not guarantee model
completion, faithful extraction or benchmark quality.

SQLite's current schema is 6. Existing scoped versions 1–5 upgrade each missing
step without inventing historical clocks or semantic/fragment receipts. PostgreSQL
has one Alembic head, `0006_semantic_bindings`, after the inherited source-retraction
revision and revisions 0003–0005. Populated support/semantic tables block destructive
downgrade. See [publication limits](publication-status.md) for unverified gates.
