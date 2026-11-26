"""Records, Hypothesis strategies and SGP4 comparisons for OMM tests."""

import csv
import datetime
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from hypothesis import strategies as st
from sgp4 import omm as sgp4_omm
from sgp4.alpha5 import to_alpha5
from sgp4.api import Satrec
from sgp4.io import compute_checksum

from keepout.omm import Omm
from keepout.omm.celestrak import loads_csv
from keepout.omm.keywords import KEYWORDS
from keepout.propagation import MAX_SATREC_NUMBER

OMM_DATA = Path(__file__).parent / "data"
SAMPLE_CSV = OMM_DATA / "celestrak_sample.csv"
SAMPLE_JSON = OMM_DATA / "celestrak_sample.json"
CCSDS_KVN = tuple(OMM_DATA / f"ccsds_omm_g{figure}.kvn" for figure in (7, 8, 9))
"""CCSDS 502.0-B-3 figures G-7 to G-9: GOES 9 in KVN (G-8 with a covariance,
G-9 with units and a user-defined parameter)."""
CCSDS_XML = OMM_DATA / "ccsds_omm_g10.xml"
"""CCSDS 502.0-B-3 figure G-10: GOES-9 in XML."""

GOES_9 = {
    "OBJECT_NAME": "GOES 9",
    "OBJECT_ID": "1995-025A",
    "EPOCH": "2020-03-04T10:34:41.426400",
    "MEAN_MOTION": "1.00273272",
    "ECCENTRICITY": "0.0005013",
    "INCLINATION": "3.0539",
    "RA_OF_ASC_NODE": "81.7939",
    "ARG_OF_PERICENTER": "249.2363",
    "MEAN_ANOMALY": "150.1602",
    "EPHEMERIS_TYPE": "0",
    "CLASSIFICATION_TYPE": "U",
    "NORAD_CAT_ID": "23581",
    "ELEMENT_SET_NO": "0925",
    "REV_AT_EPOCH": "4316",
    "BSTAR": "0.0001",
    "MEAN_MOTION_DOT": "-0.00000113",
    "MEAN_MOTION_DDOT": "0.0",
}
"""The CCSDS examples' values in CelesTrak's form (epoch to microseconds)."""

ISS = {
    "OBJECT_NAME": "ISS (ZARYA)",
    "OBJECT_ID": "1998-067A",
    "EPOCH": "2026-10-06T00:20:47.860512",
    "MEAN_MOTION": "15.48747543",
    "ECCENTRICITY": ".00068585",
    "INCLINATION": "51.6314",
    "RA_OF_ASC_NODE": "111.6262",
    "ARG_OF_PERICENTER": "227.8987",
    "MEAN_ANOMALY": "132.1419",
    "EPHEMERIS_TYPE": "0",
    "CLASSIFICATION_TYPE": "U",
    "NORAD_CAT_ID": "25544",
    "ELEMENT_SET_NO": "999",
    "REV_AT_EPOCH": "58893",
    "BSTAR": ".10208237E-3",
    "MEAN_MOTION_DOT": ".5131E-4",
    "MEAN_MOTION_DDOT": "0",
}
"""The ISS row of the sample CSV, as text."""


def csv_text(records: Sequence[Mapping[str, str]]) -> str:
    """CelesTrak CSV of text records, columns in the first record's order."""
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(records[0] if records else KEYWORDS)
    writer.writerows(record.values() for record in records)
    return stream.getvalue()


def omm_from(values: Mapping[str, str]) -> Omm:
    """The element set of one text record, read as CelesTrak CSV."""
    (omm,) = loads_csv(csv_text([values]))
    return omm


def json_number(text: str) -> str:
    """A CSV number as a JSON number: JSON needs a digit before the point."""
    sign = "-" if text.startswith("-") else ""
    body = text.removeprefix("-")
    return sign + ("0" + body if body.startswith(".") else body)


def json_text(records: Sequence[Mapping[str, str]]) -> str:
    """CelesTrak JSON of text records: numbers as JSON numbers, as written."""
    objects = []
    for record in records:
        items = [
            f"{json.dumps(key)}: "
            + (
                json.dumps(value)
                if KEYWORDS[key].kind in ("str", "time")
                else json_number(value)
            )
            for key, value in record.items()
        ]
        objects.append("{" + ", ".join(items) + "}")
    return "[" + ",\n".join(objects) + "]"


def _decimal(low: float, high: float, places: int) -> st.SearchStrategy[str]:
    """Fixed-point text, with or without a leading zero (``.5`` or ``0.5``)."""
    return st.builds(
        lambda value, bare: (
            f"{value:.{places}f}".replace("0.", ".", 1)
            if bare and value < 1.0
            else f"{value:.{places}f}"
        ),
        st.floats(low, high, allow_nan=False),
        st.booleans(),
    )


# CelesTrak's B*, ndot and nddot: ".464504E-3", "-.37E-5", "0".
_EXPONENTIAL = st.one_of(
    st.just("0"),
    st.builds(
        lambda sign, digits, exponent: f"{sign}.{digits}E{exponent}",
        st.sampled_from(["", "-"]),
        st.integers(1, 99_999_999).map(str),
        st.integers(-9, 0),
    ),
)

_NAME = st.text(st.sampled_from('ABCXYZ0189 -()/,"'), min_size=1, max_size=24).filter(
    lambda name: name.strip() == name
)

# UTC epochs from 1972 (the first read) to 2149 on CelesTrak's microsecond
# grid, within and past ERFA's leap-second table.
_EPOCH_START = datetime.datetime(1972, 1, 1, tzinfo=datetime.UTC)
_EPOCH_END = datetime.datetime(2150, 1, 1, tzinfo=datetime.UTC)
_EPOCH = st.integers(
    0, (_EPOCH_END - _EPOCH_START) // datetime.timedelta(microseconds=1) - 1
).map(
    lambda microseconds: (
        _EPOCH_START + datetime.timedelta(microseconds=microseconds)
    ).strftime("%Y-%m-%dT%H:%M:%S.%f")
)


@st.composite
def records(draw: st.DrawFn) -> dict[str, str]:
    """One object's CelesTrak values, as text in CelesTrak's styles."""
    return {
        "OBJECT_NAME": draw(_NAME),
        "OBJECT_ID": f"{draw(st.integers(1957, 2028))}-"
        f"{draw(st.integers(1, 999)):03d}{draw(st.sampled_from(['A', 'BZ', 'CYA']))}",
        "EPOCH": draw(_EPOCH),
        "MEAN_MOTION": draw(_decimal(0.1, 17.0, 8)),
        "ECCENTRICITY": draw(_decimal(0.0, 0.99, 8)),
        "INCLINATION": draw(_decimal(0.0, 180.0, 4)),
        "RA_OF_ASC_NODE": draw(_decimal(0.0, 359.9999, 4)),
        "ARG_OF_PERICENTER": draw(_decimal(0.0, 359.9999, 4)),
        "MEAN_ANOMALY": draw(_decimal(0.0, 359.9999, 4)),
        "EPHEMERIS_TYPE": "0",
        "CLASSIFICATION_TYPE": draw(st.sampled_from("UCS")),
        "NORAD_CAT_ID": str(draw(st.integers(1, 339_999))),
        "ELEMENT_SET_NO": str(draw(st.integers(0, 9999))),
        "REV_AT_EPOCH": str(draw(st.integers(0, 99_999))),
        "BSTAR": draw(_EXPONENTIAL),
        "MEAN_MOTION_DOT": draw(_EXPONENTIAL),
        "MEAN_MOTION_DDOT": draw(_EXPONENTIAL),
    }


SATREC_ATTRIBUTES = (
    "satnum",
    "classification",
    "intldesg",
    "ephtype",
    "elnum",
    "revnum",
    "epochyr",
    "epochdays",
    "jdsatepoch",
    "jdsatepochF",
    "no_kozai",
    "ecco",
    "inclo",
    "nodeo",
    "argpo",
    "mo",
    "bstar",
    "ndot",
    "nddot",
    "error",
)
"""Everything a record is initialised with, and its initialisation error."""

MINUTES = (0.0, 97.0, 1440.0, 10080.0, -4320.0)
"""Times from epoch [min] at which records are compared."""


def sgp4_record(values: Mapping[str, str]) -> Satrec:
    """A new record initialised by :func:`sgp4.omm.initialize` itself."""
    record = Satrec()
    sgp4_omm.initialize(record, values)
    return record


def assert_same_satrec(actual: Satrec, expected: Satrec) -> None:
    """Two new SGP4 records are initialised and propagate bit for bit alike.

    Propagating changes a record's ``error``: compare records never used.
    """
    for name in SATREC_ATTRIBUTES:
        assert getattr(actual, name) == getattr(expected, name), name
    for minutes in MINUTES:
        # NaN (a decayed orbit) compares equal to NaN here.
        np.testing.assert_array_equal(
            np.hstack(actual.sgp4_tsince(minutes)[1:]),
            np.hstack(expected.sgp4_tsince(minutes)[1:]),
            strict=True,
        )
        assert actual.sgp4_tsince(minutes)[0] == expected.sgp4_tsince(minutes)[0]


GOES_9_TLE = (
    "GOES 9 [P]\n"
    "1 23581U 95025A   07064.44075725 -.00000113  00000-0  10000-3 0  9250\n"
    "2 23581   3.0539  81.7939 0005013 249.2363 150.1602  1.00273272 43169\n"
)
"""CCSDS 502.0-B-3 figure G-6: the TLE the OMM examples are based on."""

GOES_9_TLE_VALUES = {
    **GOES_9,
    "OBJECT_NAME": "GOES 9 [P]",
    "EPOCH": "2007-03-05T10:34:41.426400",
    "ELEMENT_SET_NO": "925",
    "BSTAR": ".10000E-3",
    "MEAN_MOTION_DOT": "-.00000113",
    "MEAN_MOTION_DDOT": ".00000E-0",
    "ECCENTRICITY": ".0005013",
}
"""Figure G-6 in CelesTrak's form."""


def with_checksum(line: str) -> str:
    """A 68-column TLE line with its checksum, computed by :mod:`sgp4`."""
    return line + str(compute_checksum(line))


@st.composite
def element_sets(draw: st.DrawFn) -> tuple[str, dict[str, str]]:
    """A TLE (two lines, a name line or not) and its values in CelesTrak's form."""
    number = draw(st.integers(0, MAX_SATREC_NUMBER))
    satnum = to_alpha5(number)
    classification = draw(st.sampled_from("UCS "))
    # Two-digit years from 1972 on: UTC epochs before are not read.
    year = draw(st.one_of(st.integers(72, 99), st.integers(0, 56)))
    full_year = year + (1900 if year >= 57 else 2000)
    leap = full_year % 4 == 0 and (full_year % 100 != 0 or full_year % 400 == 0)
    day = draw(st.integers(1, 366 if leap else 365))
    fraction = draw(st.integers(0, 99_999_999))
    epoch = datetime.datetime(
        full_year, 1, 1, tzinfo=datetime.UTC
    ) + datetime.timedelta(days=day - 1, microseconds=fraction * 864)
    launch_year, launch = draw(st.integers(0, 99)), draw(st.integers(1, 999))
    piece = draw(st.sampled_from(["A", "BZ", "CYA"]))
    sign = st.sampled_from(" -")
    dot_sign, dot = draw(sign), draw(st.integers(0, 99_999_999))
    exponents = [
        (
            draw(sign),
            draw(st.integers(0, 99_999)),
            draw(st.sampled_from("-+")),
            draw(st.integers(0, 9)),
        )
        for _ in range(2)
    ]
    element_set_no = draw(st.integers(0, 9999))
    angles = [draw(st.integers(0, 3_599_999)) for _ in range(3)]
    inclination = draw(st.integers(0, 1_800_000))
    eccentricity = draw(st.integers(0, 9_999_999))
    mean_motion = draw(st.integers(10_000_000, 1_700_000_000))
    rev = draw(st.integers(0, 99_999))
    nddot, bstar = (f"{s}{m:05d}{e}{x}" for s, m, e, x in exponents)
    line_1 = with_checksum(
        f"1 {satnum}{classification} {launch_year:02d}{launch:03d}{piece:<3} "
        f"{year:02d}{day:03d}.{fraction:08d} {dot_sign}.{dot:08d} {nddot} {bstar} "
        f"0 {element_set_no:4d}"
    )
    line_2 = with_checksum(
        f"2 {satnum} {inclination / 1e4:8.4f} {angles[0] / 1e4:8.4f} "
        f"{eccentricity:07d} {angles[1] / 1e4:8.4f} {angles[2] / 1e4:8.4f} "
        f"{mean_motion / 1e8:11.8f}{rev:5d}"
    )
    name = draw(st.one_of(st.none(), _NAME))
    full_launch_year = launch_year + (1900 if launch_year >= 57 else 2000)
    values = {
        "OBJECT_NAME": "UNKNOWN" if name is None else name,
        "OBJECT_ID": f"{full_launch_year}-{launch:03d}{piece}",
        "EPOCH": epoch.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        "MEAN_MOTION": f"{mean_motion / 1e8:.8f}",
        "ECCENTRICITY": f".{eccentricity:07d}",
        "INCLINATION": f"{inclination / 1e4:.4f}",
        "RA_OF_ASC_NODE": f"{angles[0] / 1e4:.4f}",
        "ARG_OF_PERICENTER": f"{angles[1] / 1e4:.4f}",
        "MEAN_ANOMALY": f"{angles[2] / 1e4:.4f}",
        "EPHEMERIS_TYPE": "0",
        "CLASSIFICATION_TYPE": classification.strip() or "U",
        "NORAD_CAT_ID": str(number),
        "ELEMENT_SET_NO": str(element_set_no),
        "REV_AT_EPOCH": str(rev),
        "BSTAR": "{}.{:05d}E{}{}".format(exponents[1][0].strip(), *exponents[1][1:]),
        "MEAN_MOTION_DOT": f"{dot_sign.strip()}.{dot:08d}",
        "MEAN_MOTION_DDOT": "{}.{:05d}E{}{}".format(
            exponents[0][0].strip(), *exponents[0][1:]
        ),
    }
    # A bare name line starting "0 ", "1 " or "2 " reads as a 3LE prefix or
    # a TLE line: such a name is only ever written with the "0 " prefix.
    bare = name is not None and name[:2] not in ("0 ", "1 ", "2 ")
    prefix = draw(st.sampled_from(["", "0 "])) if bare else "0 "
    title = "" if name is None else prefix + name + "\n"
    return f"{title}{line_1}\n{line_2}\n", values
