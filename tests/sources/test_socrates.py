import dataclasses
from pathlib import Path

import astropy.units as u
import pytest
from astropy.time import Time
from celestrak_builders import SAMPLE, SOCRATES_HEADER, socrates_row

from keepout.errors import SourceFormatError
from keepout.sources.celestrak import Retrieval, SocratesReport
from keepout.sources.socrates import COLUMNS, epoch_matches, read_socrates

CALSPHERE, ISS, _, _, _, _, STARLINK, DRAGON = SAMPLE

DOCKED = socrates_row(ISS, DRAGON, ("1.000", "1.000"), "0.004", "0.000")
"""The ISS and Crew Dragon, docked, with the sample's epochs."""
FAST = socrates_row(CALSPHERE, STARLINK, ("0.838", "0.764"), "2.500", "7.500")
"""Calsphere 1 and Starlink-38128 at 7.5 km/s, the sample's epochs (DSE to
43 s)."""
REPORT = SOCRATES_HEADER + DOCKED + FAST


def test_the_columns_are_the_reports() -> None:
    assert ",".join(COLUMNS) + "\r\n" == SOCRATES_HEADER


def test_a_report_is_read_in_si_units() -> None:
    docked, fast = read_socrates(REPORT)
    assert (docked.primary, docked.secondary) == (25544, 100882)
    assert docked.since_epoch_primary == docked.since_epoch_secondary == 86400.0
    assert docked.since_epoch_tolerance == pytest.approx(43.2005)
    assert docked.tca == Time("2026-10-07T00:20:47.861", scale="utc")
    assert docked.miss_distance == pytest.approx(4.0)
    assert docked.relative_speed == 0.0
    assert docked.line == DOCKED
    assert fast.relative_speed == 7500.0
    assert fast.line == FAST


def test_a_report_without_approaches_is_empty() -> None:
    assert read_socrates(SOCRATES_HEADER) == ()


def test_a_byte_order_mark_is_skipped() -> None:
    assert read_socrates("﻿" + REPORT) == read_socrates(REPORT)


@pytest.mark.parametrize(
    "text",
    [
        "",
        REPORT.replace(",DILUTION", ",OTHER"),
        REPORT.replace("NORAD_CAT_ID_1,OBJECT_NAME_1,", "OBJECT_NAME_1,"),
    ],
)
def test_the_header_is_checked(text: str) -> None:
    with pytest.raises(SourceFormatError, match="not a SOCRATES report header"):
        read_socrates(text)


def test_every_row_has_eleven_values() -> None:
    with pytest.raises(SourceFormatError, match="line 3: 10 values, not 11"):
        read_socrates(SOCRATES_HEADER + DOCKED + DOCKED.replace(",0.010", ""))


@pytest.mark.parametrize(
    ("old", "new"), [("25544", "ISS"), ("0.004", "near"), ("00:20:47.861", "noon")]
)
def test_a_value_that_cannot_be_read_is_refused(old: str, new: str) -> None:
    with pytest.raises(SourceFormatError, match="cannot read a SOCRATES report"):
        read_socrates(SOCRATES_HEADER + DOCKED.replace(old, new, 1))


def test_epoch_matches_within_the_dse_digits() -> None:
    docked, fast = read_socrates(REPORT)
    assert epoch_matches(docked, ISS, DRAGON)
    assert epoch_matches(fast, CALSPHERE, STARLINK)
    later = dataclasses.replace(DRAGON, epoch=DRAGON.epoch + 50.0 * u.s)
    assert not epoch_matches(docked, ISS, later)
    assert not epoch_matches(docked, later, DRAGON)


def test_a_fetched_report_gives_its_approaches(tmp_path: Path) -> None:
    now = Time("2026-10-07T00:00:00", scale="utc")
    report = SocratesReport(REPORT, Retrieval("url", tmp_path, now, now))
    assert report.approaches() == read_socrates(REPORT)
