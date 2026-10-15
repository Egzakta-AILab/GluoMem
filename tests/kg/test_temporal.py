"""UTC normalization tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from gluomem.timeutil import parse_utc, to_storage, utc_key


@pytest.mark.parametrize(
    "value",
    [
        "2023-05-08 08:00:00",
        "2023-05-08T08:00:00",
        "2023-05-08T08:00:00Z",
        "2023-05-08T08:00:00+00:00",
        "2023-05-08T10:00:00+02:00",
        "2023-05-08T03:00:00-05:00",
        "2023-05-08T08:00:00.750000",
        datetime(2023, 5, 8, 8, 0, tzinfo=UTC),
        datetime(2023, 5, 8, 8, 0),
        datetime(2023, 5, 8, 9, 0, tzinfo=timezone(timedelta(hours=1))),
    ],
)
def test_equivalent_instants_share_one_key(value: object) -> None:
    assert utc_key(value) == "2023-05-08T08:00:00"  # type: ignore[arg-type]


def test_utc_storage_and_ordering() -> None:
    assert utc_key("2023-05-08") == "2023-05-08T00:00:00"
    assert parse_utc("2023-05-08") == datetime(2023, 5, 8, tzinfo=UTC)
    values = ["2023-05-08T10:00:00+02:00", "2023-05-08 07:59:59", "2023-05-08T08:00:01Z"]
    assert sorted(values, key=utc_key) == [values[1], values[0], values[2]]

    assert to_storage(None) is None
    assert to_storage("2023-05-08T13:56:00") == "2023-05-08T13:56:00"
    assert to_storage("2023-05-08") == "2023-05-08"
    assert to_storage("2023-05-08T15:56:00+02:00") == "2023-05-08T13:56:00"
    assert to_storage(datetime(2023, 5, 8, 13, 56)) == "2023-05-08T13:56:00"
    assert utc_key("not-a-date") == "not-a-date"
    with pytest.raises(ValueError):
        to_storage("last tuesday")
