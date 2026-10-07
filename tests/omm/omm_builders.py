"""Records, Hypothesis strategies and SGP4 comparisons for OMM tests."""

import csv
import datetime
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from hypothesis import strategies as st
from sgp4 import omm as sgp4_omm
from sgp4.api import Satrec

from keepout.omm.keywords import KEYWORDS

OMM_DATA = Path(__file__).parent / "data"
SAMPLE_CSV = OMM_DATA / "celestrak_sample.csv"
SAMPLE_JSON = OMM_DATA / "celestrak_sample.json"

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

# UTC epochs from 1957 to 2149 on CelesTrak's microsecond grid, on both sides
# of ERFA's leap-second table (1960 to a few years past its last leap second).
_EPOCH_START = datetime.datetime(1957, 1, 1, tzinfo=datetime.UTC)
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


def sgp4_record(values: Mapping[str, str]) -> Any:
    """A new record initialised by :func:`sgp4.omm.initialize` itself."""
    record = Satrec()
    sgp4_omm.initialize(record, values)
    return record


def assert_same_satrec(actual: Any, expected: Any) -> None:
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
