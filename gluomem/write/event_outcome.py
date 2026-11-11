"""Explicit event commit result for the outermost transaction owner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gluomem.write.events import IngestEventResult


@dataclass(frozen=True, slots=True)
class EventCommitOutcome:
    """A publication result or its original exception, never an implicit success.

    An owner must exit its write transaction normally before calling
    ``raise_if_refused``. A refusal is not a receipt or permission to complete a
    job. Neither payloads nor arbitrary exception messages appear in the repr.
    This object does not independently commit a caller-owned transaction.
    """

    result: IngestEventResult | None = field(default=None, repr=False)
    error: Exception | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if (self.result is None) == (self.error is None):
            raise ValueError("Event commit outcome requires exactly one result or error")

    @property
    def refused(self) -> bool:
        return self.error is not None

    def raise_if_refused(self) -> None:
        """Raise the same exception with its original cause and traceback intact."""
        if self.error is not None:
            raise self.error
