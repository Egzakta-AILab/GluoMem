"""I/P/B frame store backed by MindDB."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from gluomem.config.app_config import AppConfig, ChunkingConfig
from gluomem.embedding.base import Embedder, assert_dim
from gluomem.embedding.chunker import Chunk, chunk
from gluomem.embedding.compat import bind_embedder, write_fingerprint
from gluomem.embedding.fingerprint import EmbedderFingerprint, fingerprint_of
from gluomem.frames.content_hash import content_hash
from gluomem.frames.sessions import get_or_create_session
from gluomem.storage.db import MindDB
from gluomem.storage.scope import (
    DEFAULT_SCOPE,
    FrameStatus,
    MemoryKind,
    MemoryScope,
    eligible_sql,
    scope_sql,
)
from gluomem.timeutil import now_utc_iso, to_storage, utc_key

if TYPE_CHECKING:
    from gluomem.storage.backend import DatabaseBackend, FtsStore, VectorStore

FrameType = Literal["I", "P", "B"]
Importance = Literal["critical", "important", "normal", "temporary", "deprecated"]
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

_DEFAULT_IMPORTANCE: Importance = "normal"
_DEFAULT_SOURCE: FrameSource = "user_stated"
_DEFAULT_TRUST: Trust = "normal"
_B_SOURCE: FrameSource = "agent_inferred"

_COLUMNS = (
    "id, frame_type, gop_id, t, base_frame_id, content, importance, "
    "source, trust, access_count, created_at, event_time, last_accessed, "
    "tenant_id, principal_id, namespace, kind, status, confidence, "
    "superseded_by, supersedes, expires_at, pinned, verified_by_user"
)


@dataclass
class Frame:
    """In-memory representation of a ``memory_frames`` row."""

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
    tenant_id: str = "local"
    principal_id: str = "default"
    namespace: str = "personal"
    kind: MemoryKind = "fact"
    status: FrameStatus = "active"
    confidence: float = 1.0
    superseded_by: int | None = None
    supersedes: int | None = None
    expires_at: str | None = None
    pinned: bool = False
    verified_by_user: bool = False

    @property
    def scope(self) -> MemoryScope:
        """Return the ``MemoryScope`` owning this frame."""
        return MemoryScope(self.tenant_id, self.principal_id, self.namespace)

    @property
    def indexable_text(self) -> str:
        """Human-readable text for FTS, vector, and reranker indexing."""
        if self.frame_type == "B":
            try:
                payload = json.loads(self.content)
                if isinstance(payload, dict) and "description" in payload:
                    return str(payload["description"])
            except (json.JSONDecodeError, TypeError):
                pass
        return self.content


@dataclass(frozen=True)
class BCorrection:
    """Decoded payload of a B-frame (contradiction / correction)."""

    description: str
    references: list[int]


@dataclass(frozen=True)
class IndexState:
    """Readiness of a single frame's FTS, vector, and chunk projections."""

    frame_id: int
    fts_current: bool
    frame_vec: int
    chunks: int
    expected_chunks: int
    chunk_vecs: int

    @property
    def ready(self) -> bool:
        return (
            self.fts_current
            and self.frame_vec == 1
            and self.chunks == self.expected_chunks
            and self.chunk_vecs == self.expected_chunks
        )


@dataclass(frozen=True)
class LineageEntry:
    """Audit row for a frame constituent folded during compaction."""

    derived_frame_id: int
    source_frame_id: int
    source_frame_type: FrameType
    source_content_hash: str | None
    source_source: FrameSource
    source_trust: Trust
    source_created_at: str
    source_event_time: str | None
    operation: str


@dataclass(frozen=True)
class _PreparedIndex:
    """Embeddings computed before opening a database transaction."""

    text: str
    frame_vec: np.ndarray
    chunks: list[Chunk]
    chunk_vecs: list[np.ndarray]


class FrameStore:
    """Read/write store for I/P/B frames and their search projections."""

    def __init__(
        self,
        db: MindDB,
        embedder: Embedder | None = None,
        *,
        chunking: ChunkingConfig | None = None,
        scope: MemoryScope = DEFAULT_SCOPE,
    ) -> None:
        if embedder is None:
            raise TypeError("FrameStore requires an embedder")
        if embedder.dim != db.dim:
            raise ValueError(
                f"embedder.dim ({embedder.dim}) != db.dim ({db.dim})"
            )
        self._fingerprint: EmbedderFingerprint = bind_embedder(db, embedder)
        self._db = db
        self._backend: DatabaseBackend = db.backend
        self._embedder = embedder
        self._chunking = chunking or ChunkingConfig()
        self._scope: MemoryScope = scope
        self._fts: FtsStore = db.fts
        self._frames_vec: VectorStore = db.frame_vec
        self._chunks_vec: VectorStore = db.chunk_vec

    @property
    def scope(self) -> MemoryScope:
        return self._scope

    def _eff_scope(self, scope: MemoryScope | None = None) -> MemoryScope:
        return scope if scope is not None else self._scope

    @property
    def db(self) -> MindDB:
        return self._db

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._db.conn

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def chunking(self) -> ChunkingConfig:
        return self._chunking

    @property
    def fingerprint(self) -> EmbedderFingerprint:
        return self._fingerprint

    @classmethod
    def rebuild_index(
        cls,
        db: MindDB,
        embedder: Embedder,
        *,
        chunking: ChunkingConfig | None = None,
    ) -> FrameStore:
        """Re-embed every stored frame with ``embedder`` and swap the index atomically."""
        if embedder.dim != db.dim:
            raise ValueError(
                f"embedder.dim ({embedder.dim}) != db.dim ({db.dim}); "
                "dimension changes require a new MindDB"
            )
        cfg = chunking or ChunkingConfig()
        rows = db.backend.fetchall(
            f"SELECT {_COLUMNS} FROM memory_frames ORDER BY id ASC"
        )
        frames = [cls._row_to_frame(r) for r in rows]

        prepared: list[tuple[int, _PreparedIndex]] = []
        for frame in frames:
            text = frame.indexable_text
            fvec = embedder.embed(text)
            assert_dim(fvec, embedder.dim)
            chunks = chunk(text, cfg)
            cvecs = embedder.embed_batch([c.content for c in chunks])
            if len(cvecs) != len(chunks):
                raise RuntimeError(
                    f"embed_batch returned {len(cvecs)} vectors for "
                    f"{len(chunks)} chunks; rebuild aborted"
                )
            for cv in cvecs:
                assert_dim(cv, embedder.dim)
            prepared.append(
                (
                    frame.id,
                    _PreparedIndex(
                        text=text, frame_vec=fvec, chunks=chunks, chunk_vecs=cvecs
                    ),
                )
            )

        fp = fingerprint_of(embedder)
        with db.transaction():
            for fid, _ in prepared:
                chunk_rows = db.backend.fetchall(
                    "SELECT id FROM memory_frame_chunks WHERE frame_id = ?",
                    (fid,),
                )
                for (cid,) in chunk_rows:
                    db.chunk_vec.delete(int(cid), commit=False)
                db.frame_vec.delete(fid, commit=False)
                db.fts.delete(fid, commit=False)
                db.backend.execute(
                    "DELETE FROM memory_frame_chunks WHERE frame_id = ?",
                    (fid,),
                )
            for fid, prep in prepared:
                db.fts.upsert(fid, prep.text, commit=False)
                db.frame_vec.add(fid, prep.frame_vec, commit=False)
                for idx, (ch, cvec) in enumerate(
                    zip(prep.chunks, prep.chunk_vecs, strict=True)
                ):
                    cid = db.backend.insert_returning_id(
                        "INSERT INTO memory_frame_chunks"
                        "(frame_id, chunk_idx, content, char_start, char_end) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (fid, idx, ch.content, ch.char_start, ch.char_end),
                    )
                    db.chunk_vec.add(cid, cvec, commit=False)
            write_fingerprint(db, fp)
        return cls(db, embedder=embedder, chunking=cfg)

    def create_i_frame(
        self,
        gop: str,
        content: str,
        importance: Importance = _DEFAULT_IMPORTANCE,
        source: FrameSource = _DEFAULT_SOURCE,
        event_time: str | None = None,
        *,
        trust: Trust = _DEFAULT_TRUST,
        created_at: str | None = None,
        scope: MemoryScope | None = None,
        kind: MemoryKind = "fact",
        status: FrameStatus = "active",
        confidence: float = 1.0,
        expires_at: str | None = None,
        pinned: bool = False,
        verified_by_user: bool = False,
    ) -> Frame:
        """Create an I-frame (idempotent per ``(scope, gop, content_hash)``)."""
        eff_scope = self._eff_scope(scope)
        h = content_hash(content)
        existing = self._find_by_hash(gop, h, frame_type="I", scope=eff_scope)
        if existing is not None:
            self.ensure_indexed(existing)
            return existing

        return self._insert_indexed(
            index_text=content,
            scope=eff_scope,
            gop=gop,
            frame_type="I",
            base_frame_id=None,
            content=content,
            kind=kind,
            status=status,
            importance=importance,
            source=source,
            trust=trust,
            confidence=confidence,
            expires_at=expires_at,
            pinned=pinned,
            verified_by_user=verified_by_user,
            event_time=event_time,
            content_hash=h,
            created_at=created_at,
        )

    def create_p_frame(
        self,
        gop: str,
        content: str,
        base_id: int,
        importance: Importance = _DEFAULT_IMPORTANCE,
        source: FrameSource = _DEFAULT_SOURCE,
        *,
        trust: Trust = _DEFAULT_TRUST,
        event_time: str | None = None,
        created_at: str | None = None,
        scope: MemoryScope | None = None,
        kind: MemoryKind = "fact",
        status: FrameStatus = "active",
        confidence: float = 1.0,
        expires_at: str | None = None,
        pinned: bool = False,
        verified_by_user: bool = False,
    ) -> Frame:
        """Create a P-frame (delta attached to ``base_id`` in the same scope)."""
        eff_scope = self._eff_scope(scope)
        self.get_frame(base_id, scope=eff_scope)
        return self._insert_indexed(
            index_text=content,
            scope=eff_scope,
            gop=gop,
            frame_type="P",
            base_frame_id=base_id,
            content=content,
            kind=kind,
            status=status,
            importance=importance,
            source=source,
            trust=trust,
            confidence=confidence,
            expires_at=expires_at,
            pinned=pinned,
            verified_by_user=verified_by_user,
            event_time=event_time,
            content_hash=content_hash(content),
            created_at=created_at,
        )

    def create_b_frame(
        self,
        gop: str,
        description: str,
        base_id: int,
        references: list[int],
        importance: Importance | None = None,
        *,
        source: FrameSource = _B_SOURCE,
        trust: Trust = _DEFAULT_TRUST,
        event_time: str | None = None,
        created_at: str | None = None,
        scope: MemoryScope | None = None,
        kind: MemoryKind = "correction",
        status: FrameStatus = "active",
        confidence: float = 1.0,
        supersedes: int | None = None,
        expires_at: str | None = None,
        pinned: bool = False,
        verified_by_user: bool = False,
    ) -> Frame:
        """Create a B-frame (correction payload attached to ``base_id`` in the same scope)."""
        eff_scope = self._eff_scope(scope)
        base = self.get_frame(base_id, scope=eff_scope)
        if importance is None:
            resolved_importance: Importance = base.importance
        else:
            resolved_importance = importance

        payload = json.dumps(
            {"description": description, "references": list(references)},
            separators=(",", ":"),
        )
        return self._insert_indexed(
            index_text=description,
            scope=eff_scope,
            gop=gop,
            frame_type="B",
            base_frame_id=base_id,
            content=payload,
            kind=kind,
            status=status,
            importance=resolved_importance,
            source=source,
            trust=trust,
            confidence=confidence,
            supersedes=supersedes,
            expires_at=expires_at,
            pinned=pinned,
            verified_by_user=verified_by_user,
            event_time=event_time,
            content_hash=content_hash(description),
            created_at=created_at,
        )

    def get_frame(
        self, frame_id: int, *, scope: MemoryScope | None = None
    ) -> Frame:
        """Load a single frame by id within ``scope``, or raise ``KeyError``."""
        eff_scope = self._eff_scope(scope)
        row = self._backend.fetchone(
            f"SELECT {_COLUMNS} FROM memory_frames "
            "WHERE id = ? AND tenant_id = ? AND principal_id = ? AND namespace = ?",
            (
                frame_id,
                eff_scope.tenant_id,
                eff_scope.principal_id,
                eff_scope.namespace,
            ),
        )
        if row is None:
            raise KeyError(f"Frame {frame_id} not found")
        return self._row_to_frame(row)

    def find_frame_by_hash(
        self,
        gop: str,
        content: str,
        *,
        scope: MemoryScope | None = None,
    ) -> Frame | None:
        """Return the earliest frame in ``(scope, gop)`` matching ``content_hash(content)``."""
        return self._find_by_hash(
            gop, content_hash(content), scope=self._eff_scope(scope)
        )

    def frames_based_on(
        self, base_id: int, *, scope: MemoryScope | None = None
    ) -> list[Frame]:
        """Return all frames (P or B) whose ``base_frame_id == base_id`` in ``scope``."""
        eff_scope = self._eff_scope(scope)
        rows = self._backend.fetchall(
            f"SELECT {_COLUMNS} FROM memory_frames "
            "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? "
            "AND base_frame_id = ? ORDER BY id ASC",
            (
                eff_scope.tenant_id,
                eff_scope.principal_id,
                eff_scope.namespace,
                base_id,
            ),
        )
        return [self._row_to_frame(r) for r in rows]

    def set_importance(
        self,
        frame_id: int,
        importance: Importance,
        *,
        scope: MemoryScope | None = None,
    ) -> None:
        """Update ``importance`` for ``frame_id`` in ``scope`` (raises ``KeyError`` if missing)."""
        eff_scope = self._eff_scope(scope)
        self.get_frame(frame_id, scope=eff_scope)
        with self._db.transaction():
            self._backend.execute(
                "UPDATE memory_frames SET importance = ? "
                "WHERE id = ? AND tenant_id = ? AND principal_id = ? AND namespace = ?",
                (
                    importance,
                    frame_id,
                    eff_scope.tenant_id,
                    eff_scope.principal_id,
                    eff_scope.namespace,
                ),
            )

    def set_frame_importance(self, frame_id: int, importance: Importance) -> None:
        """Alias for :meth:`set_importance`."""
        self.set_importance(frame_id, importance)

    def set_status(
        self,
        frame_id: int,
        status: FrameStatus,
        *,
        superseded_by: int | None = None,
        scope: MemoryScope | None = None,
    ) -> None:
        """Update ``status`` (and optional ``superseded_by``) for ``frame_id`` in ``scope``."""
        eff_scope = self._eff_scope(scope)
        self.get_frame(frame_id, scope=eff_scope)
        with self._db.transaction():
            self._backend.execute(
                "UPDATE memory_frames SET status = ?, superseded_by = ? "
                "WHERE id = ? AND tenant_id = ? AND principal_id = ? AND namespace = ?",
                (
                    status,
                    superseded_by,
                    frame_id,
                    eff_scope.tenant_id,
                    eff_scope.principal_id,
                    eff_scope.namespace,
                ),
            )

    def get_b_correction(self, frame_id: int) -> BCorrection:
        """Decode and return the ``BCorrection`` payload of a B-frame."""
        frame = self.get_frame(frame_id)
        if frame.frame_type != "B":
            raise ValueError(
                f"Frame {frame_id} is of type {frame.frame_type!r}, not 'B'"
            )
        try:
            data = json.loads(frame.content)
            return BCorrection(
                description=str(data["description"]),
                references=[int(r) for r in data["references"]],
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Frame {frame_id} B content is not a valid correction payload: "
                f"{frame.content!r}"
            ) from exc

    def get_frames(
        self,
        frame_ids: list[int] | set[int],
        *,
        scope: MemoryScope | None = None,
    ) -> dict[int, Frame]:
        """Batch-load frames by id in ``scope`` in a single query."""
        ids = list(frame_ids)
        if not ids:
            return {}
        eff_scope = self._eff_scope(scope)
        marks = ",".join("?" * len(ids))
        rows = self._backend.fetchall(
            f"SELECT {_COLUMNS} FROM memory_frames "
            f"WHERE id IN ({marks}) "
            f"AND tenant_id = ? AND principal_id = ? AND namespace = ?",
            [
                *ids,
                eff_scope.tenant_id,
                eff_scope.principal_id,
                eff_scope.namespace,
            ],
        )
        return {int(r["id"]): self._row_to_frame(r) for r in rows}

    def resolve_chunk_frames(
        self,
        chunk_ids: list[int],
        *,
        scope: MemoryScope | None = None,
    ) -> dict[int, int]:
        """Batch-resolve ``chunk_id -> frame_id`` in ``scope`` in a single query."""
        if not chunk_ids:
            return {}
        eff_scope = self._eff_scope(scope)
        marks = ",".join("?" * len(chunk_ids))
        rows = self._backend.fetchall(
            f"SELECT c.id AS id, c.frame_id AS frame_id "
            f"FROM memory_frame_chunks c "
            f"JOIN memory_frames f ON f.id = c.frame_id "
            f"WHERE c.id IN ({marks}) "
            f"AND f.tenant_id = ? AND f.principal_id = ? AND f.namespace = ?",
            [
                *chunk_ids,
                eff_scope.tenant_id,
                eff_scope.principal_id,
                eff_scope.namespace,
            ],
        )
        return {int(r["id"]): int(r["frame_id"]) for r in rows}

    def filter_eligible_ids(
        self,
        frame_ids: list[int] | set[int],
        *,
        scope: MemoryScope | None = None,
        now_iso: str | None = None,
    ) -> set[int]:
        """Return the subset of ``frame_ids`` in ``scope`` that are recall-eligible."""
        ids = list(frame_ids)
        if not ids:
            return set()
        eff_scope = self._eff_scope(scope)
        eff_now = utc_key(now_iso) if now_iso is not None else now_utc_iso()
        id_marks = ",".join("?" * len(ids))
        rows = self._backend.fetchall(
            f"SELECT id FROM memory_frames WHERE id IN ({id_marks}) "
            f"AND {scope_sql()} AND {eligible_sql()}",
            [*ids, *eff_scope.as_tuple(), eff_now],
        )
        return {int(r["id"]) for r in rows}

    def list_gop_frames(
        self,
        gop: str,
        *,
        frame_type: FrameType | None = None,
        scope: MemoryScope | None = None,
    ) -> list[Frame]:
        """List frames in ``(scope, gop)`` ordered by ``id`` ascending."""
        eff_scope = self._eff_scope(scope)
        if frame_type is None:
            rows = self._backend.fetchall(
                f"SELECT {_COLUMNS} FROM memory_frames "
                "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? "
                "AND gop_id = ? ORDER BY id ASC",
                (
                    eff_scope.tenant_id,
                    eff_scope.principal_id,
                    eff_scope.namespace,
                    gop,
                ),
            )
        else:
            rows = self._backend.fetchall(
                f"SELECT {_COLUMNS} FROM memory_frames "
                "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? "
                "AND gop_id = ? AND frame_type = ? ORDER BY id ASC",
                (
                    eff_scope.tenant_id,
                    eff_scope.principal_id,
                    eff_scope.namespace,
                    gop,
                    frame_type,
                ),
            )
        return [self._row_to_frame(r) for r in rows]

    def all_frames(self, *, scope: MemoryScope | None = None) -> list[Frame]:
        """Return every frame in ``scope`` ordered by ``id`` ascending."""
        eff_scope = self._eff_scope(scope)
        rows = self._backend.fetchall(
            f"SELECT {_COLUMNS} FROM memory_frames "
            "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? "
            "ORDER BY id ASC",
            (
                eff_scope.tenant_id,
                eff_scope.principal_id,
                eff_scope.namespace,
            ),
        )
        return [self._row_to_frame(r) for r in rows]

    def count_frames(self, *, scope: MemoryScope | None = None) -> int:
        """Return total number of frames in ``scope``."""
        eff_scope = self._eff_scope(scope)
        val = self._backend.scalar(
            "SELECT COUNT(*) FROM memory_frames "
            "WHERE tenant_id = ? AND principal_id = ? AND namespace = ?",
            (
                eff_scope.tenant_id,
                eff_scope.principal_id,
                eff_scope.namespace,
            ),
        )
        return int(val) if val is not None else 0

    def get_frame_hashes(
        self,
        frame_ids: list[int],
        *,
        scope: MemoryScope | None = None,
    ) -> dict[int, str | None]:
        """Return ``frame_id -> content_hash`` for ``frame_ids`` in ``scope``."""
        if not frame_ids:
            return {}
        eff_scope = self._eff_scope(scope)
        marks = ",".join("?" * len(frame_ids))
        rows = self._backend.fetchall(
            f"SELECT id, content_hash FROM memory_frames "
            f"WHERE id IN ({marks}) "
            f"AND tenant_id = ? AND principal_id = ? AND namespace = ?",
            [
                *frame_ids,
                eff_scope.tenant_id,
                eff_scope.principal_id,
                eff_scope.namespace,
            ],
        )
        return {
            int(r["id"]): (
                str(r["content_hash"]) if r["content_hash"] is not None else None
            )
            for r in rows
        }

    def touch_frames(
        self,
        frame_ids: list[int],
        *,
        scope: MemoryScope | None = None,
    ) -> None:
        """Increment ``access_count`` and refresh ``last_accessed`` in ``scope``."""
        if not frame_ids:
            return
        eff_scope = self._eff_scope(scope)
        marks = ",".join("?" * len(frame_ids))
        now_expr = (
            "CURRENT_TIMESTAMP::text"
            if self._backend.dialect == "postgres"
            else "datetime('now')"
        )
        with self._db.transaction():
            self._backend.execute(
                f"UPDATE memory_frames "
                f"SET access_count = access_count + 1, last_accessed = {now_expr} "
                f"WHERE id IN ({marks}) "
                f"AND tenant_id = ? AND principal_id = ? AND namespace = ?",
                [
                    *frame_ids,
                    eff_scope.tenant_id,
                    eff_scope.principal_id,
                    eff_scope.namespace,
                ],
            )

    def reconstruct_state(
        self,
        gop: str,
        *,
        include_ineligible: bool = False,
        scope: MemoryScope | None = None,
        now_iso: str | None = None,
    ) -> tuple[Frame | None, list[Frame]]:
        """Return ``(latest_i_frame, p_frames_since_latest_i)`` for ``(scope, gop)``."""
        eff_scope = self._eff_scope(scope)
        eff_now = utc_key(now_iso) if now_iso is not None else now_utc_iso()
        elig_clause = "" if include_ineligible else f" AND {eligible_sql()}"
        elig_params: tuple[str, ...] = () if include_ineligible else (eff_now,)
        i_row = self._backend.fetchone(
            f"SELECT {_COLUMNS} FROM memory_frames "
            f"WHERE tenant_id = ? AND principal_id = ? AND namespace = ? "
            f"AND gop_id = ? AND frame_type = 'I'{elig_clause} "
            "ORDER BY id DESC LIMIT 1",
            (
                eff_scope.tenant_id,
                eff_scope.principal_id,
                eff_scope.namespace,
                gop,
                *elig_params,
            ),
        )
        if i_row is None:
            return None, []
        latest_i = self._row_to_frame(i_row)
        p_rows = self._backend.fetchall(
            f"SELECT {_COLUMNS} FROM memory_frames "
            f"WHERE tenant_id = ? AND principal_id = ? AND namespace = ? "
            f"AND gop_id = ? AND frame_type = 'P' AND id > ?{elig_clause} "
            "ORDER BY id ASC",
            (
                eff_scope.tenant_id,
                eff_scope.principal_id,
                eff_scope.namespace,
                gop,
                latest_i.id,
                *elig_params,
            ),
        )
        return latest_i, [self._row_to_frame(r) for r in p_rows]

    def compact(
        self,
        gop: str,
        cfg: AppConfig,
        *,
        scope: MemoryScope | None = None,
    ) -> None:
        """Compact eligible P-frames in ``(scope, gop)`` into their base I-frame."""
        from gluomem.frames.compaction import compact

        compact(self, gop, cfg, scope=self._eff_scope(scope))

    def lineage(self, derived_frame_id: int) -> list[LineageEntry]:
        """Return lineage audit entries for ``derived_frame_id`` ordered by id."""
        rows = self._backend.fetchall(
            "SELECT derived_frame_id, source_frame_id, source_frame_type, "
            "source_content_hash, source_source, source_trust, "
            "source_created_at, source_event_time, operation "
            "FROM memory_frame_lineage WHERE derived_frame_id = ? ORDER BY id ASC",
            (derived_frame_id,),
        )
        return [
            LineageEntry(
                derived_frame_id=int(r["derived_frame_id"]),
                source_frame_id=int(r["source_frame_id"]),
                source_frame_type=r["source_frame_type"],
                source_content_hash=r["source_content_hash"],
                source_source=r["source_source"],
                source_trust=r["source_trust"],
                source_created_at=str(r["source_created_at"]),
                source_event_time=(
                    str(r["source_event_time"])
                    if r["source_event_time"] is not None
                    else None
                ),
                operation=str(r["operation"]),
            )
            for r in rows
        ]

    def _record_lineage(
        self,
        derived_frame_id: int,
        source: Frame,
        source_hash: str | None,
        operation: str,
    ) -> None:
        """Insert one lineage row inside the caller's transaction."""
        self._backend.execute(
            "INSERT INTO memory_frame_lineage "
            "(derived_frame_id, source_frame_id, source_frame_type, "
            " source_content_hash, source_source, source_trust, "
            " source_created_at, source_event_time, operation) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                derived_frame_id,
                source.id,
                source.frame_type,
                source_hash,
                source.source,
                source.trust,
                source.created_at,
                source.event_time,
                operation,
            ),
        )

    def _content_hash_of(self, frame_id: int) -> str | None:
        val = self._backend.scalar(
            "SELECT content_hash FROM memory_frames WHERE id = ?", (frame_id,)
        )
        return None if val is None else str(val)

    def index_state(self, frame_id: int) -> IndexState:
        """Inspect whether ``frame_id`` has complete, current search projections."""
        frame = self.get_frame(frame_id)
        expected_text = frame.indexable_text
        expected_chunks = len(chunk(expected_text, self._chunking))

        fts_text = self._fts.get_text(frame_id)
        fts_current = fts_text == expected_text
        frame_vec_count = self._frames_vec.count_by_ids([frame_id])

        chunk_rows = self._backend.fetchall(
            "SELECT id FROM memory_frame_chunks WHERE frame_id = ? ORDER BY chunk_idx ASC",
            (frame_id,),
        )
        chunk_ids = [int(r["id"]) for r in chunk_rows]
        chunk_vec_count = self._chunks_vec.count_by_ids(chunk_ids)

        return IndexState(
            frame_id=frame_id,
            fts_current=fts_current,
            frame_vec=frame_vec_count,
            chunks=len(chunk_ids),
            expected_chunks=expected_chunks,
            chunk_vecs=chunk_vec_count,
        )

    def ensure_indexed(self, frame: Frame) -> bool:
        """Rebuild ``frame``'s projections if missing or stale."""
        state = self.index_state(frame.id)
        if state.ready:
            return False
        self._index(frame)
        return True

    def repair_indexes(self) -> list[int]:
        """Check all frames and re-index any with missing or stale projections."""
        repaired: list[int] = []
        for frame in self.all_frames():
            if self.ensure_indexed(frame):
                repaired.append(frame.id)
        return repaired

    @staticmethod
    def _indexable_text(frame: Frame) -> str:
        return frame.indexable_text

    def _index(self, frame: Frame) -> None:
        """Synchronously index ``frame`` into FTS, whole-frame vec, and chunk vecs."""
        prepared = self._prepare_index(frame.indexable_text)
        with self._db.transaction():
            self._write_index(frame.id, prepared)

    def _prepare_index(self, text: str) -> _PreparedIndex:
        """Compute embeddings outside any DB transaction."""
        frame_vec = self._embedder.embed(text)
        assert_dim(frame_vec, self._embedder.dim)
        chunks = chunk(text, self._chunking)
        chunk_vecs = self._embedder.embed_batch([c.content for c in chunks])
        if len(chunk_vecs) != len(chunks):
            raise RuntimeError(
                f"embed_batch returned {len(chunk_vecs)} vectors for "
                f"{len(chunks)} chunks; index not written"
            )
        for cv in chunk_vecs:
            assert_dim(cv, self._embedder.dim)
        return _PreparedIndex(
            text=text,
            frame_vec=frame_vec,
            chunks=chunks,
            chunk_vecs=chunk_vecs,
        )

    def _write_index(self, frame_id: int, prepared: _PreparedIndex) -> None:
        """Replace FTS, frame vec, and chunk rows inside the active transaction."""
        self._index_fts(frame_id, prepared.text)
        self._index_frame_vec(frame_id, prepared.frame_vec)
        self._index_chunks(frame_id, prepared.chunks, prepared.chunk_vecs)

    def _index_fts(self, frame_id: int, text: str) -> None:
        self._fts.upsert(frame_id, text, commit=False)

    def _index_frame_vec(self, frame_id: int, vec: np.ndarray) -> None:
        self._frames_vec.delete(frame_id, commit=False)
        self._frames_vec.add(frame_id, vec, commit=False)

    def _index_chunks(
        self, frame_id: int, chunks: list[Chunk], chunk_vecs: list[np.ndarray]
    ) -> None:
        self._delete_chunk_vecs(frame_id, commit=False)
        self._backend.execute(
            "DELETE FROM memory_frame_chunks WHERE frame_id = ?",
            (frame_id,),
        )
        for idx, (ch, vec) in enumerate(zip(chunks, chunk_vecs, strict=True)):
            chunk_id = self._backend.insert_returning_id(
                "INSERT INTO memory_frame_chunks"
                "(frame_id, chunk_idx, content, char_start, char_end) "
                "VALUES (?, ?, ?, ?, ?)",
                (frame_id, idx, ch.content, ch.char_start, ch.char_end),
            )
            self._chunks_vec.add(chunk_id, vec, commit=False)

    def _delete_chunk_vecs(self, frame_id: int, *, commit: bool = True) -> None:
        chunk_rows = self._backend.fetchall(
            "SELECT id FROM memory_frame_chunks WHERE frame_id = ?",
            (frame_id,),
        )
        for (cid,) in chunk_rows:
            self._chunks_vec.delete(int(cid), commit=commit)

    def _delete_index_rows(self, frame_id: int, *, commit: bool = True) -> None:
        """Delete all FTS, frame vec, chunk, and chunk vec rows for ``frame_id``."""
        if commit:
            with self._db.transaction():
                self._delete_index_rows(frame_id, commit=False)
            return
        self._delete_chunk_vecs(frame_id, commit=False)
        self._fts.delete(frame_id, commit=False)
        self._frames_vec.delete(frame_id, commit=False)
        self._backend.execute(
            "DELETE FROM memory_frame_chunks WHERE frame_id = ?",
            (frame_id,),
        )

    def _find_by_hash(
        self,
        gop: str,
        h: str,
        *,
        frame_type: FrameType | None = None,
        scope: MemoryScope | None = None,
    ) -> Frame | None:
        eff_scope = self._eff_scope(scope)
        if frame_type is None:
            row = self._backend.fetchone(
                f"SELECT {_COLUMNS} FROM memory_frames "
                "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? "
                "AND gop_id = ? AND content_hash = ? "
                "AND status IN ('active', 'candidate') "
                "ORDER BY id ASC LIMIT 1",
                (
                    eff_scope.tenant_id,
                    eff_scope.principal_id,
                    eff_scope.namespace,
                    gop,
                    h,
                ),
            )
        else:
            row = self._backend.fetchone(
                f"SELECT {_COLUMNS} FROM memory_frames "
                "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? "
                "AND gop_id = ? AND frame_type = ? AND content_hash = ? "
                "AND status IN ('active', 'candidate') "
                "ORDER BY id ASC LIMIT 1",
                (
                    eff_scope.tenant_id,
                    eff_scope.principal_id,
                    eff_scope.namespace,
                    gop,
                    frame_type,
                    h,
                ),
            )
        return self._row_to_frame(row) if row is not None else None

    def _insert(
        self,
        *,
        scope: MemoryScope | None = None,
        gop: str,
        frame_type: FrameType,
        base_frame_id: int | None,
        content: str,
        kind: MemoryKind = "fact",
        status: FrameStatus = "active",
        importance: Importance,
        source: FrameSource,
        trust: Trust,
        confidence: float = 1.0,
        superseded_by: int | None = None,
        supersedes: int | None = None,
        expires_at: str | None = None,
        pinned: bool = False,
        verified_by_user: bool = False,
        event_time: str | None,
        content_hash: str,
        created_at: str | None = None,
    ) -> int:
        """Insert a frame row (uncommitted) and return its assigned id."""
        eff_scope = self._eff_scope(scope)
        created_at_s = to_storage(created_at)
        event_time_s = to_storage(event_time)
        expires_at_s = None if pinned else to_storage(expires_at)
        if kind == "episode" and expires_at_s is None and not pinned:
            from datetime import UTC, datetime, timedelta

            from gluomem.timeutil import parse_utc

            anchor_str = created_at_s or event_time_s
            anchor_dt = (
                parse_utc(anchor_str)
                if anchor_str is not None
                else datetime.now(UTC).replace(tzinfo=None)
            )
            expires_at_s = to_storage(anchor_dt + timedelta(days=90))
        max_t = self._backend.scalar(
            "SELECT COALESCE(MAX(t), -1) FROM memory_frames "
            "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? AND gop_id = ?",
            (eff_scope.tenant_id, eff_scope.principal_id, eff_scope.namespace, gop),
        )
        t = (int(max_t) if max_t is not None else -1) + 1
        common_cols = (
            "tenant_id, principal_id, namespace, frame_type, gop_id, t, "
            "base_frame_id, content, kind, status, importance, source, trust, "
            "confidence, superseded_by, supersedes, expires_at, pinned, "
            "verified_by_user, event_time, content_hash"
        )
        common_vals = (
            eff_scope.tenant_id,
            eff_scope.principal_id,
            eff_scope.namespace,
            frame_type,
            gop,
            t,
            base_frame_id,
            content,
            kind,
            status,
            importance,
            source,
            trust,
            float(confidence),
            superseded_by,
            supersedes,
            expires_at_s,
            1 if pinned else 0,
            1 if verified_by_user else 0,
            event_time_s,
            content_hash,
        )
        if created_at_s is None:
            placeholders = ", ".join("?" * len(common_vals))
            return self._backend.insert_returning_id(
                f"INSERT INTO memory_frames ({common_cols}) VALUES ({placeholders})",
                common_vals,
            )
        vals_with_created = (*common_vals, created_at_s)
        placeholders = ", ".join("?" * len(vals_with_created))
        return self._backend.insert_returning_id(
            f"INSERT INTO memory_frames ({common_cols}, created_at) VALUES ({placeholders})",
            vals_with_created,
        )

    def _insert_indexed(self, *, index_text: str, **row: Any) -> Frame:
        """Embed ``index_text``, then insert the row and its index in one transaction."""
        prepared = self._prepare_index(index_text)
        eff_scope: MemoryScope = row["scope"]
        with self._db.transaction():
            get_or_create_session(
                self._backend, row["gop"], scope=eff_scope, commit=False
            )
            frame_id = self._insert(**row)
            self._write_index(frame_id, prepared)
        return self.get_frame(frame_id, scope=eff_scope)

    @staticmethod
    def _row_to_frame(row: Any) -> Frame:
        if hasattr(row, "keys"):
            keys = set(row.keys())
            return Frame(
                id=int(row["id"]),
                frame_type=row["frame_type"],
                gop_id=str(row["gop_id"]),
                t=int(row["t"]),
                base_frame_id=(
                    int(row["base_frame_id"])
                    if row["base_frame_id"] is not None
                    else None
                ),
                content=str(row["content"]),
                importance=row["importance"],
                source=row["source"],
                trust=row["trust"],
                access_count=int(row["access_count"]),
                created_at=str(row["created_at"]),
                event_time=(
                    str(row["event_time"]) if row["event_time"] is not None else None
                ),
                last_accessed=str(row["last_accessed"]),
                tenant_id=str(row["tenant_id"]) if "tenant_id" in keys else "local",
                principal_id=(
                    str(row["principal_id"]) if "principal_id" in keys else "default"
                ),
                namespace=str(row["namespace"]) if "namespace" in keys else "personal",
                kind=row["kind"] if "kind" in keys and row["kind"] else "fact",
                status=(
                    row["status"] if "status" in keys and row["status"] else "active"
                ),
                confidence=(
                    float(row["confidence"])
                    if "confidence" in keys and row["confidence"] is not None
                    else 1.0
                ),
                superseded_by=(
                    int(row["superseded_by"])
                    if "superseded_by" in keys and row["superseded_by"] is not None
                    else None
                ),
                supersedes=(
                    int(row["supersedes"])
                    if "supersedes" in keys and row["supersedes"] is not None
                    else None
                ),
                expires_at=(
                    str(row["expires_at"])
                    if "expires_at" in keys and row["expires_at"] is not None
                    else None
                ),
                pinned=bool(row["pinned"]) if "pinned" in keys else False,
                verified_by_user=(
                    bool(row["verified_by_user"])
                    if "verified_by_user" in keys
                    else False
                ),
            )
        values = tuple(row)
        return Frame(
            id=int(values[0]),
            frame_type=values[1],
            gop_id=str(values[2]),
            t=int(values[3]),
            base_frame_id=int(values[4]) if values[4] is not None else None,
            content=str(values[5]),
            importance=values[6],
            source=values[7],
            trust=values[8],
            access_count=int(values[9]),
            created_at=str(values[10]),
            event_time=str(values[11]) if values[11] is not None else None,
            last_accessed=str(values[12]),
            tenant_id=str(values[13]) if len(values) > 13 else "local",
            principal_id=str(values[14]) if len(values) > 14 else "default",
            namespace=str(values[15]) if len(values) > 15 else "personal",
            kind=values[16] if len(values) > 16 and values[16] else "fact",
            status=values[17] if len(values) > 17 and values[17] else "active",
            confidence=(
                float(values[18])
                if len(values) > 18 and values[18] is not None
                else 1.0
            ),
            superseded_by=(
                int(values[19])
                if len(values) > 19 and values[19] is not None
                else None
            ),
            supersedes=(
                int(values[20])
                if len(values) > 20 and values[20] is not None
                else None
            ),
            expires_at=(
                str(values[21])
                if len(values) > 21 and values[21] is not None
                else None
            ),
            pinned=bool(values[22]) if len(values) > 22 else False,
            verified_by_user=bool(values[23]) if len(values) > 23 else False,
        )
