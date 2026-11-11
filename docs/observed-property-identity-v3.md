# Observed property identity producer v3

`write.semantic_producer_contract: observed-property-identity-v3` is an explicit
producer operating point. It requires the native source-bound
`semantic-state-writer-v4` writer and an explicit source capture policy, root
model, JSON adapter and `required` schema mode (or explicit `typed-text` local
validation without provider grammar). It does not select a model or grant
permission to make requests. The existing `identity_alignment` and `consolidate`
tasks, single-send controls, router and budgets are reused; there is no additional
canonicalization call or alias store.

## Slot identity and assertion relation

The identity stage receives unchanged source-only `ClaimMeaning`, source witness,
observed candidate meanings and exact references/receipt tokens. Its additional
`property_slot_context` separates proposed subject/property/perspective coordinates
and refinement anchors from values and dates. For stored observations it includes
only witnesses belonging to the exact surviving supplied receipt tokens. A prior
publication keeps its exact backward claim index and frozen content/meaning;
no stored receipt or source witness is invented for that conditional publication.

The native proposal retains `relation`, exact nullable `target_frame_id` /
`target_claim_index` and bounded rationale. Three additional required fields are:

- `assertion_relation`: `state_observation`, `mistaken_assertion`,
  `dependent_refinement`, `unrelated`, or `unresolved`.
- `subject_alignment` and `property_alignment`: `same_label`, `source_alias`,
  or `not_aligned`.

`same_label` means exact equality of the source-proposed labels, not certified
entity identity. `source_alias` is a fallible invocation-local relation proposal
between different labels. The host validates the declared basis and exact target,
not semantic equivalence. An incorrect model alias proposal can still be wrong;
the schema is not an entailment oracle. Refinement compares the proposed anchor
with the observed independent slot, not the detail's separate property label.

| Assertion relation | Native relation | Targeted action authority |
| --- | --- | --- |
| Ordinary state observation, possibly changed value/date | `same_assertion` | Structurally legal dated `TRANSITION`, or exact semantic `NOOP` |
| Explicit correction of a mistaken assertion | `same_assertion` | Structurally legal `SUPERSEDE` |
| Dependent detail on an observed independent anchor | `refines_assertion` | Exact legal `REFINE` |
| Unrelated assertion or independent error report | `independent` | None |
| Insufficient evidence or ambiguous targets | `unresolved` | None |

The producer must decide the assertion relation before action selection. An
ordinary changed value or later date is not a correction of the earlier assertion.
A negative report about an assertion is not automatically that assertion's
correction. Subject/topic similarity does not establish a shared property.
Unrelated/unresolved proposals keep their returned relation; the host never
retargets them or chooses the first candidate. Existing untargeted ADD/CANDIDATE
policy is retained for independent claims; dependent detail without a valid anchor
can only be held as CANDIDATE. This change does not introduce a stricter
uncertainty-to-review product policy.

## Boundaries and compatibility

The producer never rewrites extracted source meanings from candidate labels,
values, dates or relation proposals. No alias equivalence or proposal is persisted
as entity identity. Stored semantic meanings remain the original source proposals.
Role/perspective, source/message/span/scope, ordered reference, receipt,
revision/generation, source withdrawal, quarantine and erasure checks remain in
force. The final commit guard and frame lifecycle/temporal code are unchanged.

V3 inherits v2's preparation-only destructive-dependency constraint: a later
same-event SUPERSEDE is withheld when its projected source scrub would revoke an
earlier transition's only authority. Independent and current-event support remain
valid positives. Date extraction remains required-nullable; omission is rejected,
null does not invent world-effective time, and non-null dates must satisfy the
unchanged temporal authority checks.

The default and all predecessors preserve effective loaded signatures,
config serialization, compiled prefixes/demos/CoT controls, rendered messages and
cache keys. AppConfig's supported enum schema intentionally gains this literal;
the complete AppConfig type schema is therefore **not** byte-identical. The
private observed projection schema gains the separately versioned proposal type.
V3's private in-memory cache already includes the explicit producer literal in
its namespace; its lookup/parser now retain and revalidate the v3 proposal type.
Persistent semantic transcripts remain unsupported, and erasure clears the
process-local cache.

Portable authored schema and host/storage tests establish consistency contracts
only. They do not show that a real model will recover independently expected
relations. Historical outputs and grades are not reinterpreted. Provider/wire
qualification and prospective real-model evaluation remain separate gates; see
[publication limits](publication-status.md).
