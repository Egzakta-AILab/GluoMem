# Observed preparation policy v2

Select explicitly in the existing field; neither V4 nor producer v1 opts in:

```yaml
write:
  writer_contract: semantic-state-writer-v4
  source_capture_policy: attributed-interaction
  semantic_producer_contract: observed-assertion-date-v2
```

The existing native single-attempt, explicit source capture, disk-cache-disabled
and identity-capability requirements still apply. No identity-model instructions,
source-only meanings, temporal rules, admission, withdrawal or commit authority
are changed. There is no new default-serialized configuration field.

## Preparation-only rule

An invocation-local selected `TRANSITION` already freezes its predecessor in the
structural projection. V2 additionally protects that predecessor's **surviving
semantic authority**, including cross-frame/shared-envelope dependencies.
Before offering a later `SUPERSEDE`, the host evaluates that explicit hypothetical
action on an isolated copy of the existing semantic/admission projection. It uses
the same suppression, review, reuse, quarantine and named source-context
revocation calculations used when applying a real selected command. If the
hypothetical effect removes all surviving authority needed by any earlier
transition, only that destructive action is omitted. Frozen candidate ordering,
frame/temporal/semantic observations, receipt tokens and meanings are not updated.

Stored independent receipts, separately scoped source envelopes and genuine
current-event corroboration can preserve authority. Cumulative prior revocations
count. Non-destructive reuse/review and legal `NOOP`, `REFINE`, `ADD` and review
choices retain their existing contracts. This is an intersection, not a planner:
no claim is removed, reinterpreted, automatically retargeted or silently changed.

## Native and cache binding

The real factory and capability-specific transient cache preserve the selected
producer version. The private and rendered projections explicitly identify v2;
the action-dependent native schema is derived from its narrowed menu. Existing
cache key material already includes the producer version, projection and input
context, so v2 has its own namespace and cached selections are revalidated.
Producer v1 remains v1; absent-policy configuration, native schemas, signatures,
rendered messages and cache behavior remain unchanged.

The final `ObservedCommitGuard.require_target` predicate is unchanged. A retained
v1 plan can still lose authority during a later sibling correction and must
refuse atomically at its final transition recheck. V2 is not permission to replay
historical responses under a different menu, refresh their revision tokens,
restore erased context/labels or weaken source/erasure guards.

## Verification boundary

Preparation narrows the legal menu; it cannot guarantee that every source has a
faithful whole-event plan. Authored choices establish host/storage contracts, not
regenerated responses or model-quality scores. An unchanged v1 plan can still
refuse at its final recheck rather than publish a stale transition. Do not silently
apply this version to previously captured responses, configurations or grades.
See [semantic writer configuration](semantic-writer-v4.md) for the required root
model, source policy, adapter and cache settings, and [publication
limits](publication-status.md) before evaluating a real workload.
