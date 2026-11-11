"""Scoped semantic proposals bound to actual provenance, never a second truth store.

No label/value hashes are persisted. Opaque tokens are revisions, not identifiers
of sensitive content. Legacy provenance is not automatically converted.
"""

from __future__ import annotations

import json
from uuid import uuid4

from gluomem.llm.semantic import (
    SEMANTIC_WRITER_CONTRACT,
    SemanticReceipt,
    source_witness,
)
from gluomem.storage.scope import resolve_scope, scope_sql


class SemanticStore:
    def __init__(self, store):
        self.store = store
        self.db = store._backend

    def snapshot(self, frame_id, *, scope=None, producer_contract=None):
        from gluomem.llm.assertion_scope import (
            ASSERTION_SCOPE_CONTRACT,
            decode_binding,
            snapshot_type,
            spec_type,
        )

        s = resolve_scope(scope, default=self.store.scope)
        self.store.get_frame(frame_id, scope=s)
        rows = self.db.fetchall(
            "SELECT cs.*, p.source_event_id, p.message_id, p.message_role, p.observed_at, "
            "p.verbatim_span, p.source_ref, p.frame_id AS actual_frame, se.payload_json "
            "FROM memory_claim_semantics cs JOIN memory_provenance p ON p.id=cs.provenance_id "
            "JOIN source_events se ON se.id=p.source_event_id "
            f"WHERE {scope_sql('cs')} AND {scope_sql('p')} AND {scope_sql('se')} "
            "AND cs.frame_id=? ORDER BY cs.provenance_id",
            (*s.as_tuple(), *s.as_tuple(), *s.as_tuple(), frame_id),
        )
        if not rows:
            return None
        receipts = []
        proposed = None
        family_meanings = {}
        selected_family = (
            ASSERTION_SCOPE_CONTRACT if producer_contract == ASSERTION_SCOPE_CONTRACT else None
        )
        try:
            for row in rows:
                if row["contract"] != SEMANTIC_WRITER_CONTRACT or row["actual_frame"] != frame_id:
                    raise ValueError("invalid binding")
                # A redacted envelope no longer authenticates a source-derived proposal.
                if row["payload_json"] == "{}" or not row["verbatim_span"]:
                    continue
                from gluomem.write.events import SourceEvent

                event = SourceEvent.model_validate(json.loads(row["payload_json"]))
                if resolve_scope(event.scope) != s or event.source_ref != row["source_ref"]:
                    raise ValueError("invalid scope")
                family, meaning = decode_binding(row["meaning_json"])
                witness = source_witness(
                    event,
                    spec_type(family)(
                        text="binding",
                        meaning=meaning,
                        message_id=row["message_id"],
                        verbatim_span=row["verbatim_span"],
                    ),
                )
                if (
                    witness.source_role != row["message_role"]
                    or witness.observed_at != row["observed_at"]
                ):
                    raise ValueError("invalid witness")
                if family in family_meanings and family_meanings[family] != meaning:
                    raise ValueError("conflicting binding")
                family_meanings[family] = meaning
                proposed = meaning
                receipts.append(
                    SemanticReceipt(
                        provenance_id=row["provenance_id"],
                        source_event_row_id=row["source_event_id"],
                        source_context=row["payload_json"],
                        token=row["binding_token"],
                        witness=witness,
                    )
                )
        except (ValueError, TypeError, KeyError):
            raise ValueError("Invalid scoped semantic source binding") from None
        # Never discard a foreign receipt to keep a convenient known subset.
        # Every row above has already passed strict family and source/witness checks.
        if set(family_meanings) != {selected_family}:
            return None
        return (
            snapshot_type(producer_contract)(meaning=proposed, receipts=tuple(receipts))
            if receipts
            else None
        )

    def attach(self, provenance_id, frame_id, meaning, witness, *, scope=None):
        from gluomem.llm.assertion_scope import encode_meaning

        s = resolve_scope(scope, default=self.store.scope)
        row = self.db.fetchone(
            f"SELECT source_event_id, message_id, message_role, observed_at, verbatim_span "
            f"FROM memory_provenance WHERE {scope_sql()} AND id=? AND frame_id=?",
            (*s.as_tuple(), provenance_id, frame_id),
        )
        if row is None or (
            row["message_id"],
            row["message_role"],
            row["observed_at"],
            row["verbatim_span"],
        ) != (witness.message_id, witness.source_role, witness.observed_at, witness.verbatim_span):
            raise ValueError("Semantic attachment requires the exact admitted source witness")
        source_context = self.db.scalar(
            f"SELECT payload_json FROM source_events WHERE {scope_sql()} AND id=?",
            (*s.as_tuple(), row["source_event_id"]),
        )
        if not source_context or source_context == "{}":
            raise ValueError("Semantic attachment requires surviving source context")
        receipt = SemanticReceipt(
            provenance_id=provenance_id,
            source_event_row_id=row["source_event_id"],
            source_context=source_context,
            token=uuid4().hex,
            witness=witness,
        )
        self.db.execute(
            "INSERT INTO memory_claim_semantics (provenance_id,frame_id,tenant_id,principal_id,"
            "namespace,contract,meaning_json,binding_token) VALUES (?,?,?,?,?,?,?,?)",
            (
                provenance_id,
                frame_id,
                *s.as_tuple(),
                SEMANTIC_WRITER_CONTRACT,
                encode_meaning(meaning),
                receipt.token,
            ),
        )
        return receipt

    def require_reuse(self, frame_id, meaning, *, scope=None):
        from gluomem.llm.assertion_scope import meaning_contract

        snapshot = self.snapshot(frame_id, scope=scope, producer_contract=meaning_contract(meaning))
        if snapshot is None or snapshot.meaning != meaning:
            raise ValueError(
                "Semantic reuse requires review: unknown or incompatible assertion binding"
            )
        return snapshot


def erase_semantic_sources(backend, event_ids, *, scope):
    """Remove proposals whose complete bound extraction context was redacted."""
    from gluomem.llm.cache import clear_semantic_caches

    clear_semantic_caches()
    if event_ids:
        marks = ",".join("?" for _ in event_ids)
        backend.execute(
            f"DELETE FROM memory_claim_semantics WHERE {scope_sql()} AND provenance_id IN "
            f"(SELECT id FROM memory_provenance WHERE {scope_sql()} "
            f"AND source_event_id IN ({marks}))",
            (*scope.as_tuple(), *scope.as_tuple(), *event_ids),
        )


def erase_semantic_frames(backend, frame_ids, *, scope):
    from gluomem.llm.cache import clear_semantic_caches

    clear_semantic_caches()
    if frame_ids:
        marks = ",".join("?" for _ in frame_ids)
        backend.execute(
            f"DELETE FROM memory_claim_semantics WHERE {scope_sql()} AND frame_id IN ({marks})",
            (*scope.as_tuple(), *frame_ids),
        )
