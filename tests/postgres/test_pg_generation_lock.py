"""Exercise generation initialization, rollback and counts on live PostgreSQL."""

import pytest

from tests.contracts.test_generation_lock import (
    test_empty_lifecycle_counts_and_scoped_lock_preserve_semantics,
    test_generation_lock_and_bump_rollback_together,
    test_source_lock_initializes_zero_without_bumping,
)

pytestmark = pytest.mark.postgres

__all__ = [
    "test_empty_lifecycle_counts_and_scoped_lock_preserve_semantics",
    "test_generation_lock_and_bump_rollback_together",
    "test_source_lock_initializes_zero_without_bumping",
]
