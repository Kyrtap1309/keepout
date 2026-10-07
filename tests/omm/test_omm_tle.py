from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sgp4
from hypothesis import given, settings
from omm_builders import (
    GOES_9_TLE,
    GOES_9_TLE_VALUES,
    MINUTES,
    assert_same_satrec,
    element_sets,
    omm_from,
    sgp4_record,
    with_checksum,
)
from sgp4.api import Satrec
from sgp4.io import compute_checksum

from keepout.errors import OmmParseError
from keepout.omm import tle

GOES_LINE_1, GOES_LINE_2 = GOES_9_TLE.splitlines()[1:]

VERIFICATION_TLES = Path(sgp4.__file__).with_name("SGP4-VER.TLE")
"""Vallado et al. (2006) verification TLEs, shipped with sgp4; columns past
69 hold test times. Three element sets (33333 to 33335, SGP4's error cases)
have line-1 checksums that do not match their edited columns."""

EPOCH_TOLERANCE = 1e-6
"""Largest epoch difference [s] to twoline2rv's record: sgp4.omm.initialize
computes the epoch in days since 1949-12-31 in one double (measured: at
most 0.39 us over all 16 626 objects of CelesTrak's active group)."""

POSITION_TOLERANCE = 0.01
"""Largest position difference [m] to twoline2rv's record over a week either
side (measured: 0.14 mm on CelesTrak's active group, 4.1 mm on object 23333,
eccentricity 0.97, of the verification set)."""


def verification_pairs() -> list[tuple[str, str]]:
    lines = [
        line[:69]
        for line in VERIFICATION_TLES.read_text().splitlines()
        if line[:2] in ("1 ", "2 ")
    ]
    return list(zip(lines[::2], lines[1::2], strict=True))


def assert_close_to_twoline2rv(record: Any, line_1: str, line_2: str) -> None:
    """Initialised alike to rounding; positions within the tolerance."""
    expected = Satrec.twoline2rv(line_1, line_2)
    assert record.satnum == expected.satnum
    epoch = (record.jdsatepoch - expected.jdsatepoch) + (
        record.jdsatepochF - expected.jdsatepochF
    )
    assert abs(epoch * 86400.0) < EPOCH_TOLERANCE
    for minutes in MINUTES:
        error, r, _ = record.sgp4_tsince(minutes)
        expected_error, expected_r, _ = expected.sgp4_tsince(minutes)
        assert error == expected_error, minutes
        if error == 0:
            difference = np.linalg.norm(np.subtract(r, expected_r)) * 1e3
            assert difference < POSITION_TOLERANCE, minutes


def test_the_ccsds_tle_is_read() -> None:
    (goes,) = tle.loads(GOES_9_TLE)
    assert goes == omm_from(GOES_9_TLE_VALUES)
    assert goes.object_name == "GOES 9 [P]"
    assert goes.epoch.isot == "2007-03-05T10:34:41.426400"


def test_the_ccsds_tle_propagates_like_twoline2rv() -> None:
    (goes,) = tle.loads(GOES_9_TLE)
    assert_same_satrec(goes.satrec(), sgp4_record(GOES_9_TLE_VALUES))
    assert_close_to_twoline2rv(goes.satrec(), GOES_LINE_1, GOES_LINE_2)


@pytest.mark.parametrize(("line_1", "line_2"), verification_pairs())
def test_the_verification_tles_propagate_like_twoline2rv(
    line_1: str, line_2: str
) -> None:
    if line_1[68] != str(compute_checksum(line_1[:68])):
        with pytest.raises(OmmParseError, match="TLE line 1 checksum"):
            tle.loads(f"{line_1}\n{line_2}\n")
        return
    (omm,) = tle.loads(f"{line_1}\n{line_2}\n")
    assert omm.object_name == "UNKNOWN"
    assert_close_to_twoline2rv(omm.satrec(), line_1, line_2)


def test_blank_fields_take_their_defaults() -> None:
    # Verification object 11801: no designator, classification U, blank
    # ephemeris type.
    line_1 = "1 11801U          80230.29629788  .01431103  00000-0  14311-1      13"
    line_2 = "2 11801  46.7916 230.4354 7318036  47.4722  10.4117  2.28537848    13"
    blank = with_checksum(line_1[:7] + " " + line_1[8:68])
    (omm,) = tle.loads(f"{blank}\n{line_2}\n")
    assert omm.international_designator == "UNKNOWN"
    assert omm.classification == "U"
    assert omm.element_set_no == 1


@pytest.mark.parametrize(
    ("satnum", "number"),
    [("A0000", 100_000), ("H9999", 179_999), ("J0001", 180_001), ("Z9999", 339_999)],
)
def test_alpha_5_catalog_numbers_are_read(satnum: str, number: int) -> None:
    line_1 = with_checksum(f"1 {satnum}{GOES_LINE_1[7:68]}")
    line_2 = with_checksum(f"2 {satnum}{GOES_LINE_2[7:68]}")
    (omm,) = tle.loads(f"{line_1}\n{line_2}\n")
    assert omm.catalog_number == number
    assert_close_to_twoline2rv(omm.satrec(), line_1, line_2)


def test_names_two_and_three_line_forms_are_read_together() -> None:
    two_line = "\n".join([GOES_LINE_1, GOES_LINE_2])
    text = f"{two_line}\n\n0 GOES 9\n{two_line}\n{GOES_9_TLE}"
    names = [omm.object_name for omm in tle.loads(text)]
    assert names == ["UNKNOWN", "GOES 9", "GOES 9 [P]"]


@pytest.mark.parametrize("name", ["0 0", "1 0", "2 X"])
def test_a_name_that_starts_like_a_line_is_read_after_the_3le_prefix(
    name: str,
) -> None:
    # Bare, such a name line would read as a 3LE prefix or a TLE line.
    (omm,) = tle.loads(f"0 {name}\n{GOES_LINE_1}\n{GOES_LINE_2}\n")
    assert omm.object_name == name


def test_trailing_blanks_and_crlf_are_ignored() -> None:
    text = GOES_9_TLE.replace("\n", "   \r\n")
    assert tle.loads(text) == tle.loads(GOES_9_TLE)


def test_blank_text_holds_no_element_sets() -> None:
    assert tle.loads("") == tle.loads("\n \n") == ()


def test_the_checksum_is_sgp4s() -> None:
    for line in (GOES_LINE_1, GOES_LINE_2):
        assert tle.checksum(line) == compute_checksum(line[:68]) == int(line[68])


def test_a_file_is_read(tmp_path: Path) -> None:
    path = tmp_path / "goes.tle"
    path.write_text(GOES_9_TLE, encoding="utf-8")
    assert tle.load(path) == tle.loads(GOES_9_TLE)


def _goes(line_1: str = GOES_LINE_1, line_2: str = GOES_LINE_2) -> str:
    return f"GOES 9\n{line_1}\n{line_2}\n"


def _fixed(line: str, column: int, text: str) -> str:
    """``line`` with ``text`` at 1-based ``column``, checksum recomputed."""
    return with_checksum(line[: column - 1] + text + line[column - 1 + len(text) : 68])


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("GOES 9\n", r"expected TLE line 1 \(record 1, line 2\)"),
        (_goes(line_2=""), r"expected TLE line 2 \(record 1, line 3\)"),
        (
            f"{GOES_LINE_2}\n{GOES_LINE_1}\n",
            r"expected TLE line 1 \(record 1, line 1\)",
        ),
        (
            _goes(GOES_LINE_1[:-1] + "1"),
            r"TLE line 1 checksum is '1', its columns give 0 \(record 1, line 2\)",
        ),
        (
            _goes(line_2=GOES_LINE_2 + "0"),
            r"TLE line 2 must have 69 columns laid out as .* \(record 1, line 3\)",
        ),
        (
            _goes(line_2=_fixed(GOES_LINE_2, 3, "23582")),
            r"line 2 is of object '23582', line 1 of '23581' "
            r"\(record 1, line 3, keyword NORAD_CAT_ID\)",
        ),
        (
            _goes(_fixed(GOES_LINE_1, 3, "I0000")),
            r"invalid TLE field 'I0000' \(record 1, line 2, keyword NORAD_CAT_ID\)",
        ),
        (
            _goes(_fixed(GOES_LINE_1, 19, "07366")),
            r"invalid TLE field '07366.44075725': no day 366 in 2007 "
            r"\(record 1, line 2, keyword EPOCH\)",
        ),
        (
            _goes(_fixed(GOES_LINE_1, 34, "-0.0000011")),
            r"invalid TLE field '-0.0000011' .*keyword MEAN_MOTION_DOT\)",
        ),
        (
            _goes(_fixed(GOES_LINE_1, 54, " 10000 3")),
            r"invalid TLE field ' 10000 3' .*keyword BSTAR\)",
        ),
        (
            _goes(_fixed(GOES_LINE_1, 63, "4")),
            r"ephemeris type 4 is not SGP4 \(0\) .*line 2, keyword EPHEMERIS_TYPE\)",
        ),
        (
            _goes(line_2=_fixed(GOES_LINE_2, 27, " 005013")),
            r"invalid TLE field ' 005013' \(record 1, line 3, keyword ECCENTRICITY\)",
        ),
        (
            _goes(line_2=_fixed(GOES_LINE_2, 9, "181.0000")),
            r"inclination must be in \[0, pi\].* \(record 1, line 3, keyword "
            r"INCLINATION\)",
        ),
    ],
)
def test_invalid_tles_are_refused_with_their_place(text: str, message: str) -> None:
    with pytest.raises(OmmParseError, match=message):
        tle.loads(text)


@settings(deadline=None)
@given(element_sets())
def test_satrec_is_sgp4s_initialisation_of_the_tle_values(
    element_set: tuple[str, dict[str, str]],
) -> None:
    text, values = element_set
    (omm,) = tle.loads(text)
    assert omm == omm_from(values)
    assert_same_satrec(omm.satrec(), sgp4_record(values))


def test_an_epoch_on_a_day_that_ends_in_a_utc_step_is_read_exactly() -> None:
    # Day 366.00000501 of 1964 is 00:00:00.432864, on a day that ended in a
    # UTC step of 0.1 s.
    line_1 = with_checksum(GOES_LINE_1[:18] + "64366.00000501" + GOES_LINE_1[32:68])
    (omm,) = tle.loads(f"{line_1}\n{GOES_LINE_2}\n")
    values = {
        **GOES_9_TLE_VALUES,
        "OBJECT_NAME": "UNKNOWN",
        "EPOCH": "1964-12-31T00:00:00.432864",
    }
    assert omm == omm_from(values)
    assert_same_satrec(omm.satrec(), sgp4_record(values))
