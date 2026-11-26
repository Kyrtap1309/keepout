import dataclasses
import math
from typing import Any

import numpy as np
import pytest
from astropy.time import Time
from omm_builders import ISS, assert_same_satrec, omm_from, sgp4_record
from screening_data import load_fixture

from keepout.ccsds.epochs import parse_epoch
from keepout.errors import UnsupportedElements
from keepout.omm.keywords import KEYWORDS
from keepout.omm.model import (
    MAX_CATALOG_NUMBER,
    MAX_ELEMENT_SET_NO,
    SGP4_EPOCH_DIGITS,
    InvalidElement,
)
from keepout.propagation import MAX_SATREC_NUMBER, satrec_of

ISS_OMM = omm_from(ISS)


def test_the_elements_are_held_in_si_units() -> None:
    assert ISS_OMM.object_name == "ISS (ZARYA)"
    assert ISS_OMM.international_designator == "1998-067A"
    assert ISS_OMM.catalog_number == 25544
    assert ISS_OMM.epoch.scale == "utc"
    assert ISS_OMM.epoch.isot == "2026-10-06T00:20:47.860512"
    assert ISS_OMM.mean_motion == 15.48747543 * (2.0 * math.pi / 86400.0)
    assert ISS_OMM.eccentricity == 0.00068585
    assert ISS_OMM.inclination == 51.6314 * (math.pi / 180.0)
    assert ISS_OMM.ra_of_asc_node == 111.6262 * (math.pi / 180.0)
    assert ISS_OMM.arg_of_pericenter == 227.8987 * (math.pi / 180.0)
    assert ISS_OMM.mean_anomaly == 132.1419 * (math.pi / 180.0)
    assert ISS_OMM.bstar == 0.10208237e-3
    assert ISS_OMM.mean_motion_dot == 0.5131e-4 * (2.0 * math.pi / 86400.0 / 86400.0)
    assert ISS_OMM.mean_motion_ddot == 0.0
    assert ISS_OMM.classification == "U"
    assert ISS_OMM.element_set_no == 999
    assert ISS_OMM.rev_at_epoch == 58893


def test_keywords_give_back_the_message_values() -> None:
    values = ISS_OMM.keyword_texts()
    assert list(values) == list(KEYWORDS)
    assert values == {
        **ISS,
        "ECCENTRICITY": "0.00068585",
        "BSTAR": "0.00010208237",
        "MEAN_MOTION_DOT": "5.131E-05",
        "MEAN_MOTION_DDOT": "0.0",
    }


def test_satrec_is_the_record_sgp4_makes_from_the_message() -> None:
    assert_same_satrec(satrec_of(ISS_OMM), sgp4_record(ISS))


def test_an_epoch_past_the_leap_second_table_is_read_and_used() -> None:
    values = {**ISS, "EPOCH": "2100-02-28T23:59:59.999999"}
    omm = omm_from(values)
    assert omm.keyword_texts() == {**ISS_OMM.keyword_texts(), "EPOCH": values["EPOCH"]}
    assert_same_satrec(satrec_of(omm), sgp4_record(values))


def test_satrec_holds_a_six_digit_catalog_number() -> None:
    record = satrec_of(dataclasses.replace(ISS_OMM, catalog_number=MAX_SATREC_NUMBER))
    assert record.satnum == MAX_SATREC_NUMBER
    assert record.satnum_str == "Z9999"


def test_satrec_refuses_a_catalog_number_sgp4_cannot_hold() -> None:
    above = dataclasses.replace(ISS_OMM, catalog_number=MAX_SATREC_NUMBER + 1)
    with pytest.raises(UnsupportedElements, match="340000 is above 339999"):
        satrec_of(above)


def test_satrec_rounds_the_epoch_to_microseconds() -> None:
    # SGP4 reads microseconds; a finer epoch is rounded, not truncated.
    finer = dataclasses.replace(
        ISS_OMM,
        epoch=Time("2026-10-06T00:20:47.8605118", scale="utc", precision=7),
    )
    assert_same_satrec(satrec_of(finer), satrec_of(ISS_OMM))
    coarse = dataclasses.replace(
        ISS_OMM, epoch=Time("2026-10-06T00:20:48", scale="utc", precision=0)
    )
    assert satrec_of(coarse).jdsatepochF == pytest.approx(
        (20 * 60 + 48) / 86400.0, abs=1e-12
    )


def test_satrec_is_the_record_sgp4_makes_from_the_keywords() -> None:
    for omm in load_fixture():
        values = omm.keyword_texts(epoch_digits=SGP4_EPOCH_DIGITS)
        assert_same_satrec(satrec_of(omm), sgp4_record(values))


@pytest.mark.parametrize(
    ("epoch", "text"),
    [
        ("2016-12-31T23:59:60.5", "2016-12-31T23:59:60.500000"),
        ("2016-12-31T23:59:59.9999996", "2016-12-31T23:59:60.000000"),
    ],
)
def test_satrec_refuses_an_epoch_in_a_leap_second(epoch: str, text: str) -> None:
    leap = dataclasses.replace(ISS_OMM, epoch=parse_epoch(epoch))
    with pytest.raises(UnsupportedElements, match=f"epoch {text} falls in a leap"):
        satrec_of(leap)


def test_the_epoch_is_stored_in_utc() -> None:
    tt = ISS_OMM.epoch.tt
    omm = dataclasses.replace(ISS_OMM, epoch=tt)
    assert omm.epoch.scale == "utc"
    assert abs((omm.epoch - ISS_OMM.epoch).sec) < 1e-9


def test_numpy_numbers_are_stored_as_python_numbers() -> None:
    changes: dict[str, Any] = {
        "catalog_number": np.int64(25544),
        "eccentricity": np.float32(0.5),
    }
    omm = dataclasses.replace(ISS_OMM, **changes)
    assert type(omm.catalog_number) is int
    assert type(omm.eccentricity) is float
    assert omm == dataclasses.replace(ISS_OMM, eccentricity=0.5)


def test_equal_elements_are_equal() -> None:
    assert omm_from(ISS) == ISS_OMM
    assert dataclasses.replace(ISS_OMM, rev_at_epoch=1) != ISS_OMM


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"object_name": 1}, "object_name must be a string"),
        ({"international_designator": None}, "international_designator must be"),
        ({"classification": "UU"}, "classification must be one character"),
        ({"classification": 85}, "classification must be a string"),
        ({"catalog_number": 1.0}, "catalog_number must be an integer"),
        ({"catalog_number": True}, "catalog_number must be an integer"),
        ({"catalog_number": -1}, r"catalog_number must be in \[0, 999999999\]"),
        (
            {"catalog_number": MAX_CATALOG_NUMBER + 1},
            r"catalog_number must be in \[0, 999999999\]",
        ),
        ({"element_set_no": -1}, r"element_set_no must be in \[0, 9999\]"),
        ({"element_set_no": 10_000}, r"element_set_no must be in \[0, 9999\]"),
        ({"rev_at_epoch": -1}, "rev_at_epoch must be >= 0"),
        ({"epoch": "2026-10-06T00:20:47"}, "epoch must be a scalar astropy Time"),
        (
            {"epoch": Time(["2026-10-06T00:20:47"] * 2, scale="utc")},
            "epoch must be a scalar",
        ),
        ({"mean_motion": "15"}, "mean_motion must be a real number"),
        ({"bstar": False}, "bstar must be a real number"),
        ({"mean_anomaly": math.nan}, "mean_anomaly must be finite"),
        ({"mean_motion_ddot": math.inf}, "mean_motion_ddot must be finite"),
        ({"mean_motion": 0.0}, "mean_motion must be > 0"),
        ({"eccentricity": 1.0}, r"eccentricity must be in \[0, 1\)"),
        ({"eccentricity": -1e-9}, r"eccentricity must be in \[0, 1\)"),
        ({"inclination": -1e-9}, r"inclination must be in \[0, pi\]"),
        ({"inclination": math.pi + 1e-9}, r"inclination must be in \[0, pi\]"),
    ],
)
def test_invalid_elements_are_refused(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(InvalidElement, match=message) as raised:
        dataclasses.replace(ISS_OMM, **changes)
    assert raised.value.attribute == next(iter(changes))


def test_the_extreme_valid_values_are_accepted() -> None:
    omm = dataclasses.replace(
        ISS_OMM,
        catalog_number=MAX_CATALOG_NUMBER,
        eccentricity=0.0,
        inclination=math.pi,
        element_set_no=MAX_ELEMENT_SET_NO,
        rev_at_epoch=0,
        object_name="",
    )
    assert omm.catalog_number == MAX_CATALOG_NUMBER


@pytest.mark.parametrize("precision", [0, 3, 9])
def test_the_written_epoch_does_not_depend_on_the_time_precision(
    precision: int,
) -> None:
    omm = dataclasses.replace(ISS_OMM, epoch=Time(ISS_OMM.epoch, precision=precision))
    assert omm.keyword_texts() == ISS_OMM.keyword_texts()
    assert omm.keyword_texts()["EPOCH"] == ISS["EPOCH"]
    assert_same_satrec(satrec_of(omm), satrec_of(ISS_OMM))


@pytest.mark.parametrize(
    ("text", "written"),
    [
        ("2026-10-06T00:20:47.86", "2026-10-06T00:20:47.860000"),
        ("2026-10-06T00:20:47.8605123", "2026-10-06T00:20:47.8605123"),
        ("2026-10-06T00:20:47.860512345", "2026-10-06T00:20:47.860512345"),
    ],
)
def test_the_written_epoch_keeps_every_digit_to_the_nanosecond(
    text: str, written: str
) -> None:
    omm = dataclasses.replace(ISS_OMM, epoch=Time(text, scale="utc", precision=3))
    assert omm.keyword_texts()["EPOCH"] == written
    assert parse_epoch(written) == omm.epoch


def test_an_epoch_finer_than_nanoseconds_is_written_to_the_nanosecond() -> None:
    # One double of Julian date: no decimal of up to nine digits holds it.
    epoch = Time(ISS_OMM.epoch.jd, format="jd", scale="utc")
    written = dataclasses.replace(ISS_OMM, epoch=epoch).keyword_texts()["EPOCH"]
    assert len(written.partition(".")[2]) == 9
    assert abs((parse_epoch(written) - epoch).sec) <= 0.5e-9


def test_keywords_round_the_epoch_to_the_requested_digits() -> None:
    omm = dataclasses.replace(
        ISS_OMM, epoch=Time("2026-10-06T00:20:47.8605126", scale="utc", precision=9)
    )
    assert omm.keyword_texts(epoch_digits=6)["EPOCH"] == "2026-10-06T00:20:47.860513"
    assert omm.keyword_texts(epoch_digits=0)["EPOCH"] == "2026-10-06T00:20:48"
