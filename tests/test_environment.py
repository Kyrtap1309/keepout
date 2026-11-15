"""Tests of what the command line and the dashboard both read."""

import pytest

from keepout._environment import parse_time


@pytest.mark.parametrize(
    ("text", "isot"),
    [
        ("2026-10-06", "2026-10-06T00:00:00.000"),
        ("2026-10-06T12:00:00", "2026-10-06T12:00:00.000"),
        ("2026-10-06 12:00", "2026-10-06T12:00:00.000"),
    ],
)
def test_parse_time_reads_iso_utc(text: str, isot: str) -> None:
    time = parse_time(text)
    assert time.isot == isot
    assert time.scale == "utc"


@pytest.mark.parametrize("text", ["J2026", "B2026", "2026:279:12:00", "soon"])
def test_parse_time_rejects_other_formats(text: str) -> None:
    with pytest.raises(ValueError, match=f"not an ISO 8601 UTC time: {text!r}"):
        parse_time(text)
