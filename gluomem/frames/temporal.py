"""Metadata-only chronology for independent I-frame state versions.

An ordinary dated change links two self-contained I frames. It does not create
P refinement, B correction, or change GOP/session identity. ``observed_at`` is
the trusted source knowledge time; ``recorded_at`` is the local storage clock.
Neither is inferred from a frame's event time, ID, or extraction-list position.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

from gluomem.frames.eligibility import is_recall_eligible
from gluomem.frames.generation import (
    bump_scope_generation,
    current_scope_generation,
    lock_scope_generation,
)
from gluomem.frames.store import FrameStore
from gluomem.storage.scope import MemoryScope, resolve_scope, scope_sql
from gluomem.timeutil import now_utc_iso, parse_utc


def _timestamp(value: str) -> str:
    return parse_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class TemporalObservation:
    """Trusted source time, separate from optional world-effective time."""

    observed_at: str
    effective_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_at", _timestamp(self.observed_at))
        if self.effective_at is not None:
            object.__setattr__(self, "effective_at", _timestamp(self.effective_at))


@dataclass(frozen=True)
class TemporalMetadata:
    frame_id: int
    scope: MemoryScope
    operation: Literal["observation", "transition"]
    previous_frame_id: int | None
    observation: TemporalObservation
    recorded_at: str


class TemporalConflict(ValueError):
    """An immutable observation or prepared state transition is no longer valid."""


@dataclass(frozen=True)
class TemporalRevision:
    """Opaque scoped freshness token; never retains frame plaintext."""

    frame_id: int
    scope: MemoryScope
    generation: int
    digest: str
    metadata: TemporalMetadata | None


@dataclass(frozen=True)
class TemporalQuery:
    as_of: str | None = None
    known_at: str | None = None
    now: str = field(default_factory=now_utc_iso)

    def __post_init__(self) -> None:
        for name in ("as_of", "known_at", "now"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _timestamp(value))


@dataclass(frozen=True)
class TemporalPolicy:
    """Authorization supplied by host policy, never by model/extracted metadata."""

    allow_historical: bool = False


@dataclass(frozen=True)
class TemporalResolution:
    """Identity-only result; callers must still fetch eligible live payloads."""

    status: Literal["resolved", "unknown", "unavailable", "unmanaged"]
    frame_id: int | None
    path: tuple[int, ...]
    metadata: TemporalMetadata | None = None


class TemporalStore:
    """Use existing frame, backend and nested-transaction interfaces."""

    def __init__(self, frames: FrameStore) -> None:
        self.frames = frames
        self.db = frames.db
        self.backend = self.db.backend

    def metadata(
        self,
        frame_id: int,
        *,
        scope: MemoryScope | None = None,
    ) -> TemporalMetadata | None:
        scope = resolve_scope(scope, default=self.frames.scope)
        self.frames.get_frame(frame_id, scope=scope)
        row = self.backend.fetchone(
            f"SELECT * FROM memory_frame_temporal WHERE frame_id = ? AND {scope_sql()}",
            (frame_id, *scope.as_tuple()),
        )
        if row is None:
            return None
        return TemporalMetadata(
            frame_id=int(row["frame_id"]),
            scope=scope,
            operation=row["operation"],
            previous_frame_id=(
                int(row["previous_frame_id"]) if row["previous_frame_id"] is not None else None
            ),
            observation=TemporalObservation(
                observed_at=str(row["observed_at"]),
                effective_at=row["effective_at"],
            ),
            recorded_at=str(row["recorded_at"]),
        )

    def record_observation(
        self,
        frame_id: int,
        observation: TemporalObservation,
        *,
        scope: MemoryScope | None = None,
    ) -> TemporalMetadata:
        scope = resolve_scope(scope, default=self.frames.scope)
        with self.db.transaction():
            lock_scope_generation(self.backend, scope)
            frame = self.frames.get_frame(frame_id, scope=scope)
            if frame.frame_type != "I" or not is_recall_eligible(frame):
                raise TemporalConflict("Observation requires an authorized independent I frame")
            existing = self.metadata(frame_id, scope=scope)
            if existing is not None:
                if existing.operation == "observation" and existing.observation == observation:
                    return existing
                raise TemporalConflict("Temporal observation metadata is immutable")
            self.backend.execute(
                "INSERT INTO memory_frame_temporal "
                "(frame_id, tenant_id, principal_id, namespace, operation, previous_frame_id, "
                "effective_at, observed_at, recorded_at) VALUES (?, ?, ?, ?, 'observation', "
                "NULL, ?, ?, ?)",
                (
                    frame_id,
                    *scope.as_tuple(),
                    observation.effective_at,
                    observation.observed_at,
                    _timestamp(now_utc_iso()),
                ),
            )
            bump_scope_generation(self.backend, scope)
        result = self.metadata(frame_id, scope=scope)
        assert result is not None
        return result

    def _path(self, frame_id: int, scope: MemoryScope) -> list[TemporalMetadata]:
        current = self.metadata(frame_id, scope=scope)
        if current is None:
            return []
        seen: set[int] = set()
        while current.previous_frame_id is not None:
            if current.frame_id in seen:
                raise TemporalConflict("Cyclic temporal metadata")
            seen.add(current.frame_id)
            parent = self.metadata(current.previous_frame_id, scope=scope)
            if parent is None:
                raise TemporalConflict("Missing predecessor observation")
            current = parent
        path: list[TemporalMetadata] = []
        seen.clear()
        while True:
            if current.frame_id in seen:
                raise TemporalConflict("Cyclic temporal metadata")
            seen.add(current.frame_id)
            path.append(current)
            row = self.backend.fetchone(
                f"SELECT frame_id FROM memory_frame_temporal WHERE previous_frame_id = ? "
                f"AND {scope_sql()}",
                (current.frame_id, *scope.as_tuple()),
            )
            if row is None:
                return path
            successor = self.metadata(int(row["frame_id"]), scope=scope)
            assert successor is not None
            current = successor

    def related_frame_ids(
        self,
        frame_id: int,
        *,
        scope: MemoryScope | None = None,
    ) -> tuple[int, ...]:
        """Privacy closure hook: state-version identities, never retained text.

        Explicit hard forgetting may combine these with existing P/B/compaction
        closure. Source withdrawal must evaluate independent support instead of
        blindly erasing an entire ordinary-change component.
        """
        scope = resolve_scope(scope, default=self.frames.scope)
        with self.db.transaction():
            lock_scope_generation(self.backend, scope)
            return tuple(item.frame_id for item in self._path(frame_id, scope)) or (frame_id,)

    def resolve(
        self,
        frame_id: int,
        *,
        policy: TemporalPolicy,
        query: TemporalQuery | None = None,
        scope: MemoryScope | None = None,
    ) -> TemporalResolution:
        """Select by explicit chain/effective/observed time, then check eligibility.

        A selected but withdrawn version yields uncertainty, never an older
        current value. Historical views do not override live hard lifecycle rules.
        """
        scope = resolve_scope(scope, default=self.frames.scope)
        query = query or TemporalQuery()
        if (query.as_of is not None or query.known_at is not None) and not policy.allow_historical:
            raise PermissionError("Historical temporal views require host policy authorization")
        with self.db.transaction():
            lock_scope_generation(self.backend, scope)
            path = self._path(frame_id, scope)
            ids = tuple(item.frame_id for item in path)
            if not path:
                return TemporalResolution("unmanaged", None, ())
            effective_at = query.as_of or query.now
            known_at = query.known_at or query.now
            selected = None
            for item in path:
                observation = item.observation
                if observation.observed_at > known_at:
                    continue
                if query.as_of is not None and observation.effective_at is None:
                    continue
                if observation.effective_at is not None and observation.effective_at > effective_at:
                    continue
                selected = item
            if selected is None:
                return TemporalResolution("unknown", None, ids)
            frame = self.frames.get_frame(selected.frame_id, scope=scope)
            if not is_recall_eligible(frame, now_iso=query.now):
                return TemporalResolution("unavailable", None, ids)
            if query.known_at is not None:
                support = self.backend.fetchall(
                    f"SELECT observed_at FROM memory_provenance WHERE frame_id = ? "
                    f"AND {scope_sql()}",
                    (frame.id, *scope.as_tuple()),
                )
                # A surviving later source supports current truth, not knowledge
                # at an earlier date from a now-withdrawn receipt. Unknown legacy
                # timestamps cannot authorize a known-at claim. Direct manual I
                # observations without provenance use their explicit trusted time.
                if support and not any(
                    row["observed_at"] is not None
                    and _timestamp(str(row["observed_at"])) <= query.known_at
                    for row in support
                ):
                    return TemporalResolution("unknown", None, ids)
            return TemporalResolution("resolved", frame.id, ids, selected)

    def revision(
        self,
        frame_id: int,
        *,
        scope: MemoryScope | None = None,
    ) -> TemporalRevision:
        scope = resolve_scope(scope, default=self.frames.scope)
        with self.db.transaction():
            lock_scope_generation(self.backend, scope)
            data = asdict(self.frames.get_frame(frame_id, scope=scope))
            for key in ("access_count", "last_accessed"):
                data.pop(key)
            digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
            return TemporalRevision(
                frame_id,
                scope,
                current_scope_generation(self.backend, scope),
                digest,
                self.metadata(frame_id, scope=scope),
            )

    def record_transition(
        self,
        expected: TemporalRevision,
        frame_id: int,
        observation: TemporalObservation,
        *,
        scope: MemoryScope | None = None,
    ) -> TemporalMetadata:
        """Link a fresh independent I without retiring or mutating the prior I.

        Call inside the ordinary writer's publication transaction so new frame,
        provenance and this edge commit or roll back together.
        """
        scope = resolve_scope(scope, default=self.frames.scope)
        with self.db.transaction():
            lock_scope_generation(self.backend, scope)
            if expected.scope != scope or expected != self.revision(expected.frame_id, scope=scope):
                raise TemporalConflict("Stale or cross-scope temporal revision")
            previous = expected.metadata
            if previous is None:
                raise TemporalConflict("Transition requires an explicit predecessor observation")
            if observation.effective_at is None:
                raise TemporalConflict("Transition requires world-effective time")
            if (
                previous.observation.effective_at is not None
                and observation.effective_at <= previous.observation.effective_at
            ):
                raise TemporalConflict("Transition must advance explicit world-effective time")
            for ident in (expected.frame_id, frame_id):
                frame = self.frames.get_frame(ident, scope=scope)
                if frame.frame_type != "I" or not is_recall_eligible(frame):
                    raise TemporalConflict("Transition requires authorized independent I frames")
            if self.metadata(frame_id, scope=scope) is not None:
                raise TemporalConflict("New state already has immutable temporal metadata")
            if (
                self.backend.fetchone(
                    f"SELECT frame_id FROM memory_frame_temporal WHERE previous_frame_id = ? "
                    f"AND {scope_sql()}",
                    (expected.frame_id, *scope.as_tuple()),
                )
                is not None
            ):
                raise TemporalConflict("Predecessor already has an explicit successor")
            self.backend.execute(
                "INSERT INTO memory_frame_temporal "
                "(frame_id, tenant_id, principal_id, namespace, operation, previous_frame_id, "
                "effective_at, observed_at, recorded_at) "
                "VALUES (?, ?, ?, ?, 'transition', ?, ?, ?, ?)",
                (
                    frame_id,
                    *scope.as_tuple(),
                    expected.frame_id,
                    observation.effective_at,
                    observation.observed_at,
                    _timestamp(now_utc_iso()),
                ),
            )
            bump_scope_generation(self.backend, scope)
        result = self.metadata(frame_id, scope=scope)
        assert result is not None
        return result
