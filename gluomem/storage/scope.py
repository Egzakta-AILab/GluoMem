"""Multi-principal isolation scope and frame lifecycle domain types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

MemoryKind = Literal[
    "fact",
    "profile",
    "preference",
    "decision",
    "procedure",
    "episode",
    "correction",
    "summary",
]
MEMORY_KINDS: tuple[str, ...] = get_args(MemoryKind)

FrameStatus = Literal[
    "candidate",
    "active",
    "superseded",
    "expired",
    "forgotten",
    "tombstoned",
]


@dataclass(frozen=True, slots=True)
class MemoryScope:
    """Three-part isolation boundary for all memory frames, sessions, and KG rows."""

    tenant_id: str = "local"
    principal_id: str = "default"
    namespace: str = "personal"

    def __post_init__(self) -> None:
        for field_name in ("tenant_id", "principal_id", "namespace"):
            val = getattr(self, field_name)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(
                    f"MemoryScope.{field_name} must be a non-empty string, got {val!r}"
                )

    def as_tuple(self) -> tuple[str, str, str]:
        """Return `(tenant_id, principal_id, namespace)` for SQL parameter binding."""
        return (self.tenant_id, self.principal_id, self.namespace)

    def as_dict(self) -> dict[str, str]:
        """Return dictionary representation of `(tenant_id, principal_id, namespace)`."""
        return {
            "tenant_id": self.tenant_id,
            "principal_id": self.principal_id,
            "namespace": self.namespace,
        }


DEFAULT_SCOPE = MemoryScope("local", "default", "personal")


def scope_sql(alias: str = "") -> str:
    """Predicate matching one scope; bind with ``*scope.as_tuple()``."""
    p = f"{alias}." if alias else ""
    return f"{p}tenant_id = ? AND {p}principal_id = ? AND {p}namespace = ?"


def eligible_sql(alias: str = "") -> str:
    """SQL form of ``is_recall_eligible``; binds one ``utc_key`` "now" parameter."""
    p = f"{alias}." if alias else ""
    return (
        f"{p}status = 'active' AND {p}trust != 'quarantined' "
        f"AND ({p}pinned = 1 OR {p}expires_at IS NULL "
        f"OR REPLACE({p}expires_at, ' ', 'T') > ?)"
    )


def resolve_scope(
    scope: MemoryScope | None = None,
    *,
    default: MemoryScope = DEFAULT_SCOPE,
) -> MemoryScope:
    """Return `scope` if provided, otherwise `default` (`DEFAULT_SCOPE`)."""
    return scope if scope is not None else default
