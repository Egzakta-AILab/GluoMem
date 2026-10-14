"""Scoped provenance and replay suppression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from gluomem.storage.backend import DatabaseBackend
from gluomem.storage.scope import DEFAULT_SCOPE, MemoryScope, resolve_scope, scope_sql

TombstoneReason = Literal["dismissed", "forgotten", "retracted"]


def redact_event_payload(
    backend: DatabaseBackend, event_row_id: int, *, scope: MemoryScope, event_id: str, revision: int
) -> None:
    """Scrub an event and queued delivery payload without changing its receipt hash."""
    backend.execute(
        f"UPDATE source_events SET payload_json = '{{}}' WHERE id = ? AND {scope_sql()}",
        (event_row_id, *scope.as_tuple()),
    )
    backend.execute(
        f"UPDATE memory_jobs SET payload = '{{}}', last_error = NULL WHERE {scope_sql()} "
        "AND job_type = 'ingest_event' AND dedup_key = ?",
        (*scope.as_tuple(), f"{event_id}:{revision}"),
    )


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Link from a stored memory frame to its originating SourceEvent and span."""

    id: int
    scope: MemoryScope
    frame_id: int
    source_event_id: int | None
    source_type: str
    source_ref: str
    message_id: str | None
    verbatim_span: str | None
    created_at: str


def is_suppressed_by_tombstone(
    backend: DatabaseBackend,
    *,
    scope: MemoryScope,
    content_hash_val: str,
    suppression_key_val: str | None = None,
    event_id: str | None = None,
    force_reassert: bool = False,
) -> bool:
    """Return True if ``content_hash_val`` is suppressed in ``scope``."""
    eff_scope = resolve_scope(scope)
    rows = backend.fetchall(
        "SELECT t.reason, t.source_event_id FROM memory_tombstones t "
        "LEFT JOIN memory_tombstone_keys k ON k.tombstone_id = t.id "
        f"WHERE {scope_sql('t')} AND (t.content_hash = ? OR k.suppression_key = ?)",
        (
            *eff_scope.as_tuple(),
            content_hash_val,
            suppression_key_val,
        ),
    )
    for row in rows:
        reason = str(row["reason"])
        src_evt = str(row["source_event_id"]) if row["source_event_id"] is not None else None
        if reason in ("forgotten", "retracted") and not force_reassert:
            return True
        if reason == "dismissed" and (src_evt is None or event_id is None or src_evt == event_id):
            return True
    return False


def get_frame_provenance(
    backend: DatabaseBackend,
    frame_id: int,
    *,
    scope: MemoryScope = DEFAULT_SCOPE,
) -> list[ProvenanceRecord]:
    """Return all provenance records attached to ``frame_id`` in ``scope``."""
    return get_frames_provenance(backend, [frame_id], scope=scope)


def get_frames_provenance(
    backend: DatabaseBackend, frame_ids: list[int], *, scope: MemoryScope = DEFAULT_SCOPE
) -> list[ProvenanceRecord]:
    """Load provenance for a batch of frames with one query."""
    if not frame_ids:
        return []
    placeholders = ",".join("?" for _ in frame_ids)
    rows = backend.fetchall(
        "SELECT id, tenant_id, principal_id, namespace, frame_id, source_event_id, "
        "source_type, source_ref, message_id, verbatim_span, created_at "
        "FROM memory_provenance "
        f"WHERE {scope_sql()} AND frame_id IN ({placeholders}) "
        "ORDER BY id ASC",
        (
            *scope.as_tuple(),
            *frame_ids,
        ),
    )
    return [
        ProvenanceRecord(
            id=int(r["id"]),
            scope=MemoryScope(str(r["tenant_id"]), str(r["principal_id"]), str(r["namespace"])),
            frame_id=int(r["frame_id"]),
            source_event_id=(
                int(r["source_event_id"]) if r["source_event_id"] is not None else None
            ),
            source_type=str(r["source_type"]),
            source_ref=str(r["source_ref"]),
            message_id=str(r["message_id"]) if r["message_id"] is not None else None,
            verbatim_span=(str(r["verbatim_span"]) if r["verbatim_span"] is not None else None),
            created_at=str(r["created_at"]),
        )
        for r in rows
    ]
