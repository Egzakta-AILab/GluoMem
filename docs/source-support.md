# Revocable source-support fragments

Ordinary supersession still scrubs the superseded frame, source envelope and queued
payload. Before that irreversible envelope redaction, the lifecycle may authorize
an unchanged sibling's **exact existing supporting quote**. It never stores a
second plaintext copy or restores a previously erased envelope.

## Admission and verification

`frames/source_support.py` is an internal lifecycle seam, not a model-facing writer
operation. It reads the original persisted envelope, checks its SHA-256 receipt
hash and scoped source/event identity, then uses `bind_source_claim` to revalidate
the existing provenance message, role, observation time and literal quote. A quote
must occur exactly once in its source message/document. Character offsets are
Python Unicode string offsets, half-open `[char_start, char_end)`, not UTF-8 bytes.
Null source timestamps remain null; the existing receipt-time display fallback is
not a new inferred world-effective date. Legacy unknown role/source attribution
cannot acquire a witness.

Only nonoverlapping validated sibling ranges survive ordinary supersession. A
full-message or overlapping quote containing the removed supporting span is
cleared, never trimmed, expanded, synthesized or marked verified by assertion.
An ambiguous removed span prevents retaining backing for that envelope. Ambiguous
surviving quotes lose backing themselves. Subsequent redaction of an already
scrubbed envelope may keep an existing exact witness, but cannot mint a new one.
Prepared duplicate input is not restoration authority.

`memory_source_support` contains only a provenance foreign key, scope, original
payload hash, exact offsets, nullable message timestamp and a binding digest. The
quote lives solely in the existing `memory_provenance.verbatim_span`. The digest
binds protocol, provenance/frame ID and frame content hash, source receipt ID,
identity/hash, message ID, role, observation/receipt times, offsets and quote hash.
The evidence resolver checks that binding against live scoped rows and source
withdrawal status. Retargeting a provenance row, changing content, role, source,
span, time, offsets or scope does not create valid retained backing.

This assumes trusted database/application writes. The digest is **not a keyed
signature against a malicious DBA**, and literal source membership does not prove
semantic entailment of an extracted claim. Callers/models cannot submit or mint
witnesses through SourceEvent/HTTP schemas.

## Lifecycle and erasure

- Ordinary SUPERSEDE and `correct_memory` retain only authorized disjoint siblings.
- Hard forget, scope forget and dismissal conservatively revoke all fragments and
  provenance quotes of affected envelopes, including unrelated same-envelope
  quotes. Independently authorized source envelopes remain available unless their
  frames are themselves in the existing lifecycle erasure closure.
- Source retraction deletes that source's provenance and cascading witnesses,
  scrubs its envelope/jobs and preserves the existing durable replay fence.
- The scrubbed frame always loses its witnesses, even when its source link is
  missing or damaged. SQL rollback rolls witnesses and redaction back together;
  existing reentrant-erasure replay applies normal lifecycle revocation again.
- Frame-content/compaction changes are not authorization to transfer or rebind a
  retained witness. A changed binding fails closed. This repair does not promise
  retained-proof transfer through compaction or semantic entailment validation.

These are logical application/SQL erasure guarantees, not secure deletion from
SQLite pages/WAL, backups, external providers or copies already returned to callers.

## API compatibility

Usable evidence retains the existing `span_verified` field and adds nullable
`verification_basis`: `raw-source` for current envelope membership or
`retained-fragment` for an authorized source-fragment binding. Unverified/legacy
objects omit the new field when it was absent; there is no new public restore or
fragment-write API. The same resolver supplies embedded, sync/async HTTP SDK and
rendered `[evidence]` context. Raw provenance/export is not a verification API.

Old evaluators that require the complete raw envelope will still reject retained
fragments. Any new evaluator mode must be explicit and source-bound; do not treat
the string `verification_basis` alone as authority or silently rescore old evidence.

## Schema seam

The source-support step advances SQLite `user_version` **4 -> 5** using
`storage/migrations/0005_source_support.sql`; fresh schema includes the same table.
The composed current SQLite version is **6**; schema 5 upgrades through the
semantic-bindings step. PostgreSQL Alembic revision is **0005_source_support**, directly after
`0004_source_attribution`, with matching `storage/pg_schema.py` metadata. Neither
migration backfills proofs or plaintext. PostgreSQL downgrade refuses a nonempty
witness table. Offline SQL generation is not live PostgreSQL qualification.

Both table deltas are composed: source-support revision 0005 precedes semantic
bindings revision 0006. Neither migration backfills authority. Never relabel a v4
schema as v5 or use an erased payload, diagnostic capture, model quote or incoming
duplicate to backfill supporting evidence.
