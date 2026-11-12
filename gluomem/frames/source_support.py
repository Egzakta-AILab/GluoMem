"""Revocable, non-plaintext witnesses for exact surviving source fragments.

Only lifecycle redaction mints witnesses, from an authenticated still-present
original envelope. The original quote stays solely in memory_provenance. A
witness is a scoped binding digest, not a source archive or semantic entailment
claim. Database writes are trusted; this is not authentication against a DBA.
"""

from __future__ import annotations

import hashlib
import json

from gluomem.storage.backend import DatabaseBackend
from gluomem.storage.scope import MemoryScope, scope_sql

_PROTOCOL = "source-fragment-v1"
_BIND_FIELDS = (
    "id",
    "tenant_id",
    "principal_id",
    "namespace",
    "frame_id",
    "frame_content_hash",
    "source_event_id",
    "source_type",
    "source_ref",
    "message_id",
    "message_role",
    "observed_at",
    "created_at",
    "event_id",
    "revision",
    "source_created_at",
    "payload_hash",
)


def _binding_hash(row: dict, witness: dict) -> str:
    values = {key: row[key] for key in _BIND_FIELDS}
    values.update({key: witness[key] for key in ("char_start", "char_end", "message_time")})
    values["quote_hash"] = hashlib.sha256(row["verbatim_span"].encode("utf-8")).hexdigest()
    values["protocol"] = _PROTOCOL
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _valid_witness(row: dict, witness: dict | None) -> bool:
    if not witness or not row.get("verbatim_span") or row.get("withdrawn"):
        return False
    if tuple(witness[key] for key in ("tenant_id", "principal_id", "namespace")) != tuple(
        row[key] for key in ("tenant_id", "principal_id", "namespace")
    ):
        return False
    return bool(
        witness["event_payload_hash"] == row["payload_hash"]
        and witness["char_start"] >= 0
        and witness["char_end"] - witness["char_start"] == len(row["verbatim_span"])
        and witness["binding_hash"] == _binding_hash(row, witness)
    )


def retained_support(row: dict, witness: dict | None) -> tuple[str | None, str | None] | None:
    """Resolve an existing binding only while its live source/frame is authorized."""
    if row.get("frame_status") not in ("active", "candidate"):
        return None
    if witness is None or not _valid_witness(row, witness):
        return None
    return row["message_role"], witness["message_time"]


def _authenticated_event(row: dict, scope: MemoryScope):
    from gluomem.write.events import SourceEvent

    raw = row["payload_json"]
    if not isinstance(raw, str) or hashlib.sha256(raw.encode()).hexdigest() != row["payload_hash"]:
        return None
    try:
        event = SourceEvent.model_validate_json(raw)
    except ValueError:
        return None
    if (
        event.scope != scope
        or event.event_id != row["event_id"]
        or event.revision != row["revision"]
        or event.source_ref != row["source_ref"]
        or event.source_type != row["source_type"]
        or event.capture_policy is None
    ):
        return None
    return event


def _locate(row: dict, event) -> dict | None:
    from gluomem.write.source_binding import bind_source_claim

    quote = row["verbatim_span"]
    if not quote or event is None or row["created_at"] != row["source_created_at"]:
        return None
    try:
        binding = bind_source_claim(
            event, message_id=row["message_id"], quote=quote, capture_policy=event.capture_policy
        )
    except ValueError:
        return None
    if binding.role != row["message_role"] or binding.observed_at != row["observed_at"]:
        return None
    message = next((m for m in event.messages if m.message_id == row["message_id"]), None)
    text = message.content if message is not None else event.content
    start = text.find(quote)
    if start < 0 or text.find(quote, start + 1) >= 0:
        return None  # Repeated spans cannot establish a unique erasure boundary.
    return {
        "char_start": start,
        "char_end": start + len(quote),
        "message_time": message.timestamp if message is not None else None,
    }


def _overlaps(row: dict, location: dict, removed: list[tuple[dict, dict]]) -> bool:
    return any(
        row["message_id"] == other["message_id"]
        and location["char_start"] < span["char_end"]
        and span["char_start"] < location["char_end"]
        for other, span in removed
    )


def redact_frame_support(
    backend: DatabaseBackend,
    frame_ids: list[int],
    *,
    scope: MemoryScope,
    preserve_siblings: bool,
) -> None:
    """Called under the lifecycle generation lock, BEFORE envelope redaction.

    Ordinary supersession may retain exact nonoverlapping sibling support. Hard
    erasure/dismissal conservatively revoke every fragment of affected envelopes.
    Repeated ordinary redaction can only retain existing witnesses, never mint
    from a redacted source or use a supplied duplicate payload to restore proof.
    """
    ids = set(frame_ids)
    ph = ",".join("?" for _ in frame_ids)
    events = backend.fetchall(
        f"SELECT DISTINCT source_event_id FROM memory_provenance WHERE {scope_sql()} "
        f"AND frame_id IN ({ph}) AND source_event_id IS NOT NULL",
        (*scope.as_tuple(), *frame_ids),
    )
    for event_row in events:
        rows = backend.fetchall(
            "SELECT p.*, se.payload_json, se.payload_hash, se.event_id, se.revision, "
            "se.created_at AS source_created_at, m.status AS frame_status, "
            "m.content_hash AS frame_content_hash, r.source_ref AS withdrawn "
            "FROM memory_provenance p JOIN source_events se ON se.id = p.source_event_id "
            f"AND {scope_sql('se')} JOIN memory_frames m ON m.id = p.frame_id "
            f"AND {scope_sql('m')} LEFT JOIN source_retractions r ON {scope_sql('r')} "
            f"AND r.source_ref = p.source_ref WHERE {scope_sql('p')} AND p.source_event_id = ?",
            (
                *scope.as_tuple(),
                *scope.as_tuple(),
                *scope.as_tuple(),
                *scope.as_tuple(),
                event_row["source_event_id"],
            ),
        )
        records = [dict(row) for row in rows]
        witnesses = {
            int(w["provenance_id"]): dict(w)
            for w in backend.fetchall(
                f"SELECT s.* FROM memory_source_support s JOIN memory_provenance p "
                f"ON p.id = s.provenance_id WHERE {scope_sql('s')} AND {scope_sql('p')} "
                "AND p.source_event_id = ?",
                (*scope.as_tuple(), *scope.as_tuple(), event_row["source_event_id"]),
            )
        }
        locations = {}
        for row in records:
            event = _authenticated_event(row, scope) if not row["withdrawn"] else None
            location = _locate(row, event)
            if location is None and _valid_witness(row, witnesses.get(row["id"])):
                location = witnesses[row["id"]]
            if location is not None:
                locations[row["id"]] = location
        affected = [row for row in records if row["frame_id"] in ids]
        # A malformed/missing target binding cannot define safe surviving ranges.
        can_preserve = (
            preserve_siblings and bool(affected) and all(row["id"] in locations for row in affected)
        )
        removed = [(row, locations[row["id"]]) for row in affected if row["id"] in locations]
        for row in records:
            location = locations.get(row["id"])
            keep = (
                can_preserve
                and row["frame_id"] not in ids
                and location is not None
                and row["frame_status"] in ("active", "candidate")
                and not row["withdrawn"]
                and not _overlaps(row, location, removed)
            )
            if not keep:
                backend.execute(
                    f"DELETE FROM memory_source_support WHERE provenance_id = ? AND {scope_sql()}",
                    (row["id"], *scope.as_tuple()),
                )
                backend.execute(
                    "UPDATE memory_provenance SET verbatim_span = NULL WHERE id = ? "
                    f"AND {scope_sql()}",
                    (row["id"], *scope.as_tuple()),
                )
                continue
            # Existing witnesses are retained as-is; never upgraded from old quotes.
            if row["id"] in witnesses:
                continue
            backend.execute(
                "INSERT INTO memory_source_support (provenance_id, tenant_id, principal_id, "
                "namespace, event_payload_hash, char_start, char_end, message_time, binding_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row["id"],
                    *scope.as_tuple(),
                    row["payload_hash"],
                    location["char_start"],
                    location["char_end"],
                    location["message_time"],
                    _binding_hash(row, location),
                ),
            )
    # The frame itself loses all support, even if a damaged source link prevented
    # envelope discovery. Do this AFTER using valid old witnesses for overlap.
    backend.execute(
        f"DELETE FROM memory_source_support WHERE {scope_sql()} AND provenance_id IN "
        f"(SELECT id FROM memory_provenance WHERE {scope_sql()} AND frame_id IN ({ph}))",
        (*scope.as_tuple(), *scope.as_tuple(), *frame_ids),
    )
