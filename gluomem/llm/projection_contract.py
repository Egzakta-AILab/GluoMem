"""Exact host projection family shared by preparation and native validation."""


def semantic_projection_type(producer_contract, reuse_policy=None):
    """Resolve host authority, not its deliberately narrower native wire view."""
    from gluomem.llm.observed_identity import PRODUCER_CONTRACTS, ObservedSemanticProjection
    from gluomem.llm.semantic import SemanticProjection

    if reuse_policy is not None:
        from gluomem.write.receipt_coherence import POLICY, ReceiptCoherentProjection

        if reuse_policy != POLICY or producer_contract != "observed-assertion-scope-v4":
            raise ValueError("Unsupported semantic projection policy combination")
        return ReceiptCoherentProjection
    if producer_contract == "observed-assertion-scope-v4":
        from gluomem.llm.assertion_scope import AssertionScopedProjection

        return AssertionScopedProjection
    if producer_contract in PRODUCER_CONTRACTS:
        return ObservedSemanticProjection
    if producer_contract is None:
        return SemanticProjection
    raise ValueError("Unsupported semantic projection producer contract")
