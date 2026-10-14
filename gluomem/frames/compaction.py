"""gluomem.frames.compaction — GOP compaction.

``compact(store, gop, cfg)`` merges older P-frames into their base I-frame when
a base I accumulates MORE THAN ``write.compaction.p_frame_trigger`` P-frames.
The ``write.compaction.keep_recent`` most recent P-frames are retained; the
older P-frame bodies are merged INTO the base I-frame content, joined with
``write.compaction.separator``; the merged I-frame is RE-EMBEDDED (its vec +
FTS + chunk rows are replaced); and the merged P-frames are referentially
deleted along with their FTS/vec/chunk index rows (chunks cascade on the frame
delete; FTS and vec0 rows are not FK-cascaded and are removed explicitly).

Only P-frames that :func:`gluomem.frames.eligibility.can_merge_into` accepts
for the base are counted toward the trigger or merged: the merged I keeps the
base's trust and source, so a quarantined, lower-trust, other-source or
other-GOP delta is never folded into (and promoted by) the base.

Every constant (trigger, keep_recent, separator) is read from the passed
``AppConfig`` — nothing is hardcoded. The defaults (trigger=10, keep_recent=5,
separator=``"\\n---\\n"``) live in ``CompactionConfig``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gluomem.config.app_config import AppConfig
from gluomem.frames.content_hash import content_hash
from gluomem.frames.eligibility import can_merge_into

if TYPE_CHECKING:
    from gluomem.frames.store import FrameStore


def compact(store: FrameStore, gop: str, cfg: AppConfig) -> None:
    """Compact a GOP's P-frames into their base I-frame.

    For each base I-frame referenced by P-frames in ``gop``: if it accumulates
    MORE THAN ``cfg.write.compaction.p_frame_trigger`` P-frames, the oldest
    P-frames beyond the ``cfg.write.compaction.keep_recent`` most recent are
    merged into the base I content (joined by ``cfg.write.compaction.separator``),
    the merged I is re-indexed (re-embedded), and the merged P-frames + their
    index rows are deleted. Below or at the trigger, this is a no-op.
    """
    comp = cfg.write.compaction
    trigger = int(comp.p_frame_trigger)
    keep = int(comp.keep_recent)
    sep = comp.separator

    # Group the GOP's P-frames by their base I-frame id, ordered by id ASC
    # (id reflects write order, so the most recent P-frames have the greatest ids).
    rows = store._conn.execute(
        "SELECT id, base_frame_id FROM memory_frames "
        "WHERE gop_id=? AND frame_type='P' AND base_frame_id IS NOT NULL "
        "ORDER BY id ASC",
        (gop,),
    ).fetchall()
    by_base: dict[int, list[int]] = {}
    for pid, bid in rows:
        base_id = int(bid) if bid is not None else None
        if base_id is None:  # pragma: no cover — query filters NULLs
            continue
        by_base.setdefault(base_id, []).append(int(pid))

    if keep < 0:
        raise ValueError(
            f"write.compaction.keep_recent must be non-negative, got {keep}"
        )

    for base_id, p_ids in by_base.items():
        if len(p_ids) <= trigger:
            continue
        # The merged I keeps the base's trust/source, so only deltas that
        # cannot be promoted by the merge are candidates; the rest stay as
        # standalone P-frames with their own (possibly ineligible) trust.
        base = store.get_frame(base_id)
        compatible = [
            p for p in (store.get_frame(pid) for pid in p_ids)
            if can_merge_into(base, p)
        ]
        if len(compatible) <= trigger:
            continue
        # The most recent `keep` P-frames are retained; the rest are merged.
        older_frames = (
            compatible[: len(compatible) - keep] if keep > 0 else compatible
        )
        if not older_frames:
            continue
        older = [p.id for p in older_frames]

        # Merge the older P-frame bodies INTO the base I-frame content, joined
        # by the separator. P-frame content is plain text (B-frames are not
        # compacted here — they are corrections, not deltas on the base I).
        older_bodies = [p.content for p in older_frames]
        merged_content = base.content + sep + sep.join(older_bodies)

        # Update the I-frame content + content_hash (the merged content is a new
        # canonical dedup key).
        new_hash = content_hash(merged_content)
        store._conn.execute(
            "UPDATE memory_frames SET content=?, content_hash=? WHERE id=?",
            (merged_content, new_hash, base_id),
        )
        store._conn.commit()

        # Re-embed / re-index the merged I-frame (replaces its FTS + vec + chunks).
        merged_frame = store.get_frame(base_id)
        store._index(merged_frame)

        # Referentially delete the merged P-frames and their index rows. The
        # memory_frame_chunks rows cascade on the frame delete, but the FTS and
        # vec0 rows are not FK-cascaded — _delete_index_rows removes them
        # explicitly (including chunk-vec rows by chunk id).
        for pid in older:
            store._delete_index_rows(pid)
            store._conn.execute("DELETE FROM memory_frames WHERE id=?", (pid,))
        store._conn.commit()
