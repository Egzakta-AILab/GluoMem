"""gluomem.frames.store — FrameStore I/P/B writers.

The frame store is the core memory primitive. Three frame types:

- **I-frame** (intra / fact): a self-contained fact. ``base_frame_id`` is NULL.
  Dedup is by ``content_hash`` within a GOP: writing content that normalizes
  equal to an existing I-frame in the same GOP returns the existing frame
  instead of inserting (idempotent).
- **P-frame** (predicted / delta): a refinement or new observation about the
  same subject as a base I-frame. ``base_frame_id`` points at the base I.
- **B-frame** (bi-directional / correction): written by the contradiction
  detector when a later statement conflicts with a stored I/P. Its content is a
  JSON payload ``{"description": ..., "references": [...]}`` recoverable via
  ``get_b_correction``. When ``importance`` is None, the B-frame INHERITS the
  corrected base frame's importance.

Every write funnels through the internal ``_index(frame)`` hook, which performs
synchronous FTS + whole-frame vector + chunk row writes.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from gluomem.config.app_config import ChunkingConfig
from gluomem.embedding.base import Embedder
from gluomem.embedding.chunker import Chunk, chunk
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.content_hash import content_hash
from gluomem.frames.sessions import get_or_create_session
from gluomem.storage.db import MindDB
from gluomem.storage.sqlite_vec import SqliteVecStore

if TYPE_CHECKING:
    from gluomem.config.app_config import AppConfig

# --- typed enum aliases (mirror the schema CHECK constraints) ----------------

Importance = Literal[
    "critical", "important", "normal", "temporary", "deprecated"
]
FrameSource = Literal[
    "user_stated",
    "tool_verified",
    "agent_inferred",
    "import",
    "system",
    "personal",
    "workspace",
    "team_sync",
]
Trust = Literal["trusted", "normal", "low", "quarantined"]
FrameType = Literal["I", "P", "B"]


# --- dataclasses -------------------------------------------------------------


@dataclass
class Frame:
    """A typed memory frame."""

    id: int
    frame_type: FrameType
    gop_id: str
    t: int
    base_frame_id: int | None
    content: str
    importance: Importance
    source: FrameSource
    trust: Trust
    access_count: int
    created_at: str
    event_time: str | None
    last_accessed: str


@dataclass
class BCorrection:
    """The recoverable payload of a B-frame: a correction description + references."""

    description: str
    references: list[int]


# --- the store ---------------------------------------------------------------

# Column order returned by SELECT * on memory_frames (matches schema.sql).
_COLUMNS = (
    "id",
    "frame_type",
    "gop_id",
    "t",
    "base_frame_id",
    "content",
    "importance",
    "source",
    "trust",
    "access_count",
    "created_at",
    "event_time",
    "last_accessed",
    "content_hash",
)

# The B-frame correction payload is stored as the frame's `content` (JSON).
# This shape is the contract the B accessor reads back; the index hook indexes
# the *description* text (not the raw JSON) so FTS/vec match the human-readable
# correction.
_B_SOURCE: FrameSource = "agent_inferred"


class FrameStore:
    """The I/P/B frame store over a ``MindDB`` connection.

    Parameters
    ----------
    db:
        An open ``MindDB`` (WAL, FK, sqlite-vec loaded, schema applied). The
        store reuses its connection for all reads/writes.
    embedder:
        The embedder used by the ``_index`` write-through hook to produce
        whole-frame and chunk vectors at the embedder's native dim. Defaults to
        a deterministic ``MockEmbedder`` at ``db.dim`` so the store is fully
        usable offline with no explicit embedder. ``embedder.dim`` MUST equal
        ``db.dim`` (the vec0 tables were created at that dim).
    """

    def __init__(
        self,
        db: MindDB,
        *,
        embedder: Embedder | None = None,
        chunking: ChunkingConfig | None = None,
    ) -> None:
        self._db: MindDB = db
        self._conn: sqlite3.Connection = db.conn
        if embedder is None:
            embedder = MockEmbedder(dim=db.dim)
        if embedder.dim != db.dim:
            raise ValueError(
                f"embedder.dim ({embedder.dim}) must equal db.dim ({db.dim}); "
                f"the vec0 tables were created at db.dim"
            )
        self._embedder: Embedder = embedder
        # Chunking config drives the pure-function chunker. Defaults to the
        # documented ChunkingConfig defaults (max 2000 / overlap 200 /
        # whole-frame < 1500) so callers that don't pass a config still chunk.
        self._chunking: ChunkingConfig = chunking or ChunkingConfig()
        self._frames_vec: SqliteVecStore = SqliteVecStore(
            db.conn, table="memory_frames_vec", dim=db.dim
        )
        self._chunks_vec: SqliteVecStore = SqliteVecStore(
            db.conn, table="memory_frame_chunks_vec", dim=db.dim
        )

    @property
    def embedder(self) -> Embedder:
        """The frame embedder (read-only) — reused by the semantic deduper so a
        probe and the write-through index share one embedding provider."""
        return self._embedder

    # --- I-frames (facts) --------------------------------------------------

    def create_i_frame(
        self,
        gop: str,
        content: str,
        *,
        importance: Importance = "normal",
        source: FrameSource = "user_stated",
        event_time: str | None = None,
        trust: Trust = "normal",
        created_at: str | None = None,
    ) -> Frame:
        """Create an I-frame, deduping by ``content_hash`` within ``gop``.

        If an existing frame in ``gop`` already has the same content hash, that
        frame is returned unchanged (idempotent — no new row, no re-index).
        Otherwise a new I-frame is inserted and routed through ``_index``.
        """
        get_or_create_session(self._conn, gop)
        chash = content_hash(content)

        existing = self.find_frame_by_hash(gop, content)
        if existing is not None:
            return existing

        frame_id = self._insert(
            gop=gop,
            frame_type="I",
            base_frame_id=None,
            content=content,
            importance=importance,
            source=source,
            trust=trust,
            event_time=event_time,
            content_hash=chash,
            created_at=created_at,
        )
        frame = self.get_frame(frame_id)
        self._index(frame)
        return frame

    # --- P-frames (deltas) -------------------------------------------------

    def create_p_frame(
        self,
        gop: str,
        content: str,
        base_id: int,
        *,
        importance: Importance = "normal",
        source: FrameSource = "user_stated",
        event_time: str | None = None,
        trust: Trust = "normal",
        created_at: str | None = None,
    ) -> Frame:
        """Create a P-frame: a delta/refinement on the base I-frame ``base_id``."""
        get_or_create_session(self._conn, gop)
        frame_id = self._insert(
            gop=gop,
            frame_type="P",
            base_frame_id=base_id,
            content=content,
            importance=importance,
            source=source,
            trust=trust,
            event_time=event_time,
            content_hash=content_hash(content),
            created_at=created_at,
        )
        frame = self.get_frame(frame_id)
        self._index(frame)
        return frame

    # --- B-frames (corrections) -------------------------------------------

    def create_b_frame(
        self,
        gop: str,
        description: str,
        base_id: int,
        references: list[int],
        *,
        importance: Importance | None = None,
        event_time: str | None = None,
        trust: Trust = "normal",
        created_at: str | None = None,
    ) -> Frame:
        """Create a B-frame correction ``{description, references}``.

        The correction is persisted as JSON in the frame's ``content`` so the
        B accessor (``get_b_correction``) can read it back. When ``importance``
        is None, the B-frame INHERITS the corrected base frame's importance
        (e.g. correcting a ``critical`` frame yields a ``critical`` B-frame).
        """
        get_or_create_session(self._conn, gop)

        # Inherit importance from the corrected base frame when unspecified.
        if importance is None:
            base = self.get_frame(base_id)
            importance = base.importance

        payload = json.dumps(
            {"description": description, "references": list(references)},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        frame_id = self._insert(
            gop=gop,
            frame_type="B",
            base_frame_id=base_id,
            content=payload,
            importance=importance,
            source=_B_SOURCE,
            trust=trust,
            event_time=event_time,
            content_hash=content_hash(description),
            created_at=created_at,
        )
        frame = self.get_frame(frame_id)
        self._index(frame)
        return frame

    # --- read accessors ----------------------------------------------------

    def find_frame_by_hash(self, gop: str, content: str) -> Frame | None:
        """Return the earliest frame in ``gop`` whose content hash matches.

        This is the dedup probe: ``create_i_frame`` uses it internally, and the
        admission gate uses it to make P/B admission idempotent as well
        (the store itself only dedups I-frames).
        """
        row = self._conn.execute(
            "SELECT id FROM memory_frames WHERE gop_id=? AND content_hash=? "
            "ORDER BY id ASC LIMIT 1",
            (gop, content_hash(content)),
        ).fetchone()
        if row is None:
            return None
        return self.get_frame(int(row[0]))

    def frames_based_on(self, base_id: int) -> list[Frame]:
        """Return frames whose ``base_frame_id`` is ``base_id`` (id ascending).

        Read accessor for the ``overwrite`` demotion path (a superseded
        base I drags its P-refinements down with it).
        """
        rows = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM memory_frames "
            "WHERE base_frame_id=? ORDER BY id ASC",
            (base_id,),
        ).fetchall()
        return [self._row_to_frame(r) for r in rows]

    def set_importance(self, frame_id: int, importance: Importance) -> None:
        """Set a frame's importance in place (e.g. demote to ``deprecated``).

        Pure UPDATE — importance is not part of the FTS/vector index text, so
        no re-index is needed; scoring reads it from the row at query time.
        Raises ``KeyError`` if the frame does not exist.
        """
        cur = self._conn.execute(
            "UPDATE memory_frames SET importance=? WHERE id=?",
            (importance, frame_id),
        )
        if cur.rowcount == 0:
            raise KeyError(f"no memory frame with id={frame_id}")
        self._conn.commit()

    def get_frame(self, frame_id: int) -> Frame:
        """Return the typed ``Frame`` for ``frame_id``.

        Raises ``KeyError`` if no frame exists with that id.
        """
        row = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM memory_frames WHERE id=?",
            (frame_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"no memory frame with id={frame_id}")
        return self._row_to_frame(row)

    def get_b_correction(self, frame_id: int) -> BCorrection:
        """Return the ``{description, references}`` payload of a B-frame.

        Raises ``KeyError`` if the frame does not exist and ``ValueError`` if it
        is not a B-frame or its content is not a valid correction payload.
        """
        frame = self.get_frame(frame_id)
        if frame.frame_type != "B":
            raise ValueError(
                f"frame {frame_id} is type {frame.frame_type!r}, not 'B'"
            )
        try:
            payload = json.loads(frame.content)
        except json.JSONDecodeError as exc:  # pragma: no cover — defensive
            raise ValueError(
                f"B-frame {frame_id} content is not valid JSON: {exc}"
            ) from exc
        try:
            return BCorrection(
                description=str(payload["description"]),
                references=[int(r) for r in payload["references"]],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"B-frame {frame_id} content is not a valid correction "
                f"payload (need 'description' + 'references'): {exc!r}"
            ) from exc

    # --- state reconstruction ---------------------------------------------

    def reconstruct_state(self, gop: str) -> tuple[Frame | None, list[Frame]]:
        """Return ``(latest_I_frame | None, [P-frames written after that I])``.

        The latest I-frame is the I-frame with the greatest id in ``gop``. The
        P-frames are every P-frame in ``gop`` whose id is greater than the
        latest I's id (i.e. written after it), ordered by id ascending. If the
        GOP has no I-frames, returns ``(None, [])``.
        """
        latest_i_row = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM memory_frames "
            "WHERE gop_id=? AND frame_type='I' ORDER BY id DESC LIMIT 1",
            (gop,),
        ).fetchone()
        if latest_i_row is None:
            return None, []
        latest_i = self._row_to_frame(latest_i_row)

        p_rows = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM memory_frames "
            "WHERE gop_id=? AND frame_type='P' AND id > ? ORDER BY id ASC",
            (gop, latest_i.id),
        ).fetchall()
        return latest_i, [self._row_to_frame(r) for r in p_rows]

    # --- index hook (FTS + vec + chunk write-through) ----------------------

    def _index(self, frame: Frame) -> None:
        """Synchronous index write-through hook (FTS + vec + chunks).

        On every frame write this populates:

        - the FTS row (``memory_frames_fts``) with the indexable text, so a
          ``MATCH`` returns the frame id;
        - the whole-frame vector (``memory_frames_vec``) at the embedder's
          native dim, via ``SqliteVecStore``;
        - chunk row(s) (``memory_frame_chunks``) plus their chunk vectors
          (``memory_frame_chunks_vec``), produced by the pure-function chunker.

        For B-frames the *description* text is indexed (not the raw JSON
        payload) so FTS/vec match the human-readable correction.

        Failure atomicity: all embeddings are computed BEFORE any index row is
        touched (the embedder is the fallible step — network/model), and the
        FTS + vec + chunk writes then land in one transaction committed at the
        end. An embedder failure leaves existing index rows (compaction
        re-index) untouched; a SQL failure rolls the partial index write back.
        The frame row itself is committed by ``_insert`` before this hook runs,
        so a failed index leaves the frame present but unindexed — never a
        half-indexed frame.
        """
        text = self._indexable_text(frame)
        frame_vec = self._embedder.embed(text)
        chunks = chunk(text, self._chunking)
        chunk_vecs = self._embedder.embed_batch([c.content for c in chunks])
        if len(chunk_vecs) != len(chunks):
            raise RuntimeError(
                f"embed_batch returned {len(chunk_vecs)} vectors for "
                f"{len(chunks)} chunks (frame {frame.id}); index not written"
            )
        try:
            self._index_fts(frame.id, text)
            self._index_frame_vec(frame.id, frame_vec)
            self._index_chunks(frame.id, chunks, chunk_vecs)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # --- index helpers -----------------------------------------------------

    @staticmethod
    def _indexable_text(frame: Frame) -> str:
        """Return the text to index for a frame.

        For B-frames the content is a JSON correction payload
        ``{"description": ..., "references": [...]}``; the human-readable
        *description* is what should be searched/embedded, not the raw JSON.
        For I/P-frames the content itself is the indexable text.
        """
        if frame.frame_type == "B":
            try:
                payload = json.loads(frame.content)
                return str(payload.get("description", frame.content))
            except (json.JSONDecodeError, TypeError):
                return frame.content
        return frame.content

    def _index_fts(self, frame_id: int, text: str) -> None:
        """Insert (or replace) the FTS row for ``frame_id`` with ``text``.

        Runs inside ``_index``'s transaction — no commit here.
        """
        # Delete-then-insert so re-indexing (e.g. compaction re-embed) replaces
        # the row rather than creating a duplicate. FTS5 is not FK-cascaded.
        self._conn.execute(
            "DELETE FROM memory_frames_fts WHERE rowid=?", (frame_id,)
        )
        self._conn.execute(
            "INSERT INTO memory_frames_fts(rowid, content) VALUES (?, ?)",
            (frame_id, text),
        )

    def _index_frame_vec(self, frame_id: int, vec: np.ndarray) -> None:
        """Write (or replace) the whole-frame vector for ``frame_id``.

        Takes the precomputed vector (``_index`` embeds before any index row is
        touched) and runs inside ``_index``'s transaction — no commit here.
        """
        # Replace any existing vec row (compaction re-embeds the merged I).
        self._frames_vec.delete(frame_id, commit=False)
        self._frames_vec.add(frame_id, vec, commit=False)

    def _index_chunks(
        self, frame_id: int, chunks: list[Chunk], chunk_vecs: list[np.ndarray]
    ) -> None:
        """Write chunk row(s) + chunk vectors for ``frame_id``.

        Takes precomputed chunks (pure-function
        :func:`gluomem.embedding.chunker.chunk`) and their vectors — ``_index``
        embeds everything before any index row is touched. Each chunk gets a
        row in ``memory_frame_chunks`` (with ``char_start``/``char_end``) and a
        vector in ``memory_frame_chunks_vec`` at the embedder's native dim.
        Runs inside ``_index``'s transaction — no commit here.
        """
        # Remove any existing chunk rows + chunk vec rows for this frame
        # (compaction re-indexes the merged I-frame in place).
        self._delete_chunk_vecs(frame_id, commit=False)
        self._conn.execute(
            "DELETE FROM memory_frame_chunks WHERE frame_id=?", (frame_id,)
        )
        for idx, (ch, vec) in enumerate(zip(chunks, chunk_vecs, strict=True)):
            cur = self._conn.execute(
                "INSERT INTO memory_frame_chunks(frame_id, chunk_idx, content, "
                "char_start, char_end) VALUES (?, ?, ?, ?, ?)",
                (frame_id, idx, ch.content, ch.char_start, ch.char_end),
            )
            chunk_id = cur.lastrowid
            if chunk_id is None:  # pragma: no cover — INSERT always sets lastrowid
                raise RuntimeError("chunk INSERT did not return a lastrowid")
            self._chunks_vec.add(int(chunk_id), vec, commit=False)

    def _delete_chunk_vecs(self, frame_id: int, *, commit: bool = True) -> None:
        """Delete chunk-vector rows for every chunk belonging to ``frame_id``.

        The ``memory_frame_chunks_vec`` vec0 table cannot carry a FK to
        ``memory_frame_chunks``, so ON DELETE CASCADE does not reach it; callers
        that remove chunks must clean the chunk-vec rows explicitly.
        ``commit=False`` leaves the deletes inside the caller's transaction.
        """
        rows = self._conn.execute(
            "SELECT id FROM memory_frame_chunks WHERE frame_id=?", (frame_id,)
        ).fetchall()
        for (chunk_id,) in rows:
            self._chunks_vec.delete(int(chunk_id), commit=commit)

    def _delete_index_rows(self, frame_id: int) -> None:
        """Delete ALL index rows (FTS + frame vec + chunks + chunk vecs) for a frame.

        Used by compaction when referentially deleting a merged P-frame. The
        ``memory_frame_chunks`` rows cascade on the frame-row delete, but the
        FTS and vec0 rows are not FK-cascaded, so they are removed explicitly.
        All deletes land in one transaction (rolled back together on failure).
        """
        try:
            self._delete_chunk_vecs(frame_id, commit=False)
            self._conn.execute(
                "DELETE FROM memory_frames_fts WHERE rowid=?", (frame_id,)
            )
            self._frames_vec.delete(frame_id, commit=False)
            self._conn.execute(
                "DELETE FROM memory_frame_chunks WHERE frame_id=?", (frame_id,)
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # --- compaction (delegates to gluomem.frames.compaction) -----------------

    def compact(self, gop: str, cfg: AppConfig) -> None:
        """Compact a GOP's P-frames into its base I-frame.

        Delegates to ``gluomem.frames.compaction.compact``. See that module for
        the trigger/keep/merge/re-embed/referential-delete semantics. ``cfg`` is
        an ``AppConfig``; the compaction constants are read from
        ``cfg.write.compaction`` (nothing hardcoded).
        """
        from gluomem.frames.compaction import compact

        compact(self, gop, cfg)

    # --- internals ---------------------------------------------------------

    def _insert(
        self,
        *,
        gop: str,
        frame_type: FrameType,
        base_frame_id: int | None,
        content: str,
        importance: Importance,
        source: FrameSource,
        trust: Trust,
        event_time: str | None,
        content_hash: str,
        created_at: str | None = None,
    ) -> int:
        """Insert a frame row and return its assigned id.

        ``t`` is a per-GOP monotonic counter (max existing t in the GOP + 1, or
        0 for the first frame) so ``idx_frames_gop_t`` reflects write order.
        ``created_at=None`` keeps the schema default (ingest time = decay
        anchor); an explicit ISO string BACKDATES the row — benchmark ingest
        uses this so recency decay measures conversation time, not run time.
        """
        t_row = self._conn.execute(
            "SELECT COALESCE(MAX(t), -1) FROM memory_frames WHERE gop_id=?",
            (gop,),
        ).fetchone()
        t = int(t_row[0]) + 1
        if created_at is None:
            cur = self._conn.execute(
                "INSERT INTO memory_frames "
                "(frame_type, gop_id, t, base_frame_id, content, importance, "
                " source, trust, event_time, content_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    frame_type,
                    gop,
                    t,
                    base_frame_id,
                    content,
                    importance,
                    source,
                    trust,
                    event_time,
                    content_hash,
                ),
            )
        else:
            cur = self._conn.execute(
                "INSERT INTO memory_frames "
                "(frame_type, gop_id, t, base_frame_id, content, importance, "
                " source, trust, event_time, content_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    frame_type,
                    gop,
                    t,
                    base_frame_id,
                    content,
                    importance,
                    source,
                    trust,
                    event_time,
                    content_hash,
                    created_at,
                ),
            )
        self._conn.commit()
        lastrowid = cur.lastrowid
        if lastrowid is None:  # pragma: no cover — sqlite3 always sets it on INSERT
            raise RuntimeError("INSERT did not return a lastrowid")
        return int(lastrowid)

    def _row_to_frame(self, row: sqlite3.Row | tuple) -> Frame:
        """Build a typed ``Frame`` from a row ordered like ``_COLUMNS``."""
        values = tuple(row)
        return Frame(
            id=int(values[0]),
            frame_type=values[1],  # type: ignore[arg-type]
            gop_id=values[2],
            t=int(values[3]),
            base_frame_id=int(values[4]) if values[4] is not None else None,
            content=values[5],
            importance=values[6],  # type: ignore[arg-type]
            source=values[7],  # type: ignore[arg-type]
            trust=values[8],  # type: ignore[arg-type]
            access_count=int(values[9]),
            created_at=values[10],
            event_time=values[11],
            last_accessed=values[12],
        )
