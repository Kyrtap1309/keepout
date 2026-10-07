import dataclasses
import math
from typing import Any

import numpy as np
import pytest
from astropy.time import Time
from omm_builders import ISS, assert_same_satrec, sgp4_record

from keepout.omm._io import parse_record
from keepout.omm.keywords import DEGREE, KEYWORDS, REVOLUTION_PER_DAY
from keepout.omm.model import MAX_CATALOG_NUMBER, MAX_SATREC_NUMBER, InvalidElement

ISS_OMM = parse_record(ISS)


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


def test_the_angle_unit_is_sgp4s() -> None:
    # sgp4.omm multiplies degrees by pi / 180.0: the same double.
    assert math.pi / 180.0 == DEGREE
    assert 2.0 * math.pi / 86400.0 == REVOLUTION_PER_DAY


def test_keywords_give_back_the_message_values() -> None:
    values = ISS_OMM.keywords()
    assert list(values) == list(KEYWORDS)
    assert values == {
        **ISS,
        "ECCENTRICITY": "0.00068585",
        "BSTAR": "0.00010208237",
        "MEAN_MOTION_DOT": "5.131E-05",
        "MEAN_MOTION_DDOT": "0.0",
    }


def test_satrec_is_the_record_sgp4_makes_from_the_message() -> None:
    assert_same_satrec(ISS_OMM.satrec(), sgp4_record(ISS))


@pytest.mark.parametrize(
    "epoch", ["1958-03-17T12:00:00.000000", "2100-02-28T23:59:59.999999"]
)
def test_epochs_outside_the_leap_second_table_are_read_and_used(epoch: str) -> None:
    values = {**ISS, "EPOCH": epoch}
    omm = parse_record(values)
    assert omm.keywords() == {**ISS_OMM.keywords(), "EPOCH": epoch}
    assert_same_satrec(omm.satrec(), sgp4_record(values))


def test_satrec_holds_a_six_digit_catalog_number() -> None:
    record = dataclasses.replace(ISS_OMM, catalog_number=MAX_SATREC_NUMBER).satrec()
    assert record.satnum == MAX_SATREC_NUMBER
    assert record.satnum_str == "Z9999"


def test_satrec_refuses_a_catalog_number_sgp4_cannot_hold() -> None:
    above = dataclasses.replace(ISS_OMM, catalog_number=MAX_SATREC_NUMBER + 1)
    with pytest.raises(ValueError, match="340000 is above 339999"):
        above.satrec()


def test_satrec_rounds_the_epoch_to_microseconds() -> None:
    # SGP4 reads microseconds; a finer epoch is rounded, not truncated.
    finer = dataclasses.replace(
        ISS_OMM,
        epoch=Time("2026-10-06T00:20:47.8605118", scale="utc", precision=7),
    )
    assert_same_satrec(finer.satrec(), ISS_OMM.satrec())
    coarse = dataclasses.replace(
        ISS_OMM, epoch=Time("2026-10-06T00:20:48", scale="utc", precision=0)
    )
    assert coarse.satrec().jdsatepochF == pytest.approx(
        (20 * 60 + 48) / 86400.0, abs=1e-12
    )


def test_satrec_refuses_an_epoch_in_a_leap_second() -> None:
    leap = dataclasses.replace(
        ISS_OMM, epoch=Time("2016-12-31T23:59:60.5", scale="utc")
    )
    with pytest.raises(ValueError, match="leap second"):
        leap.satrec()


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
    assert parse_record(ISS) == ISS_OMM
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
        ({"element_set_no": -1}, "element_set_no must be >= 0"),
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
        element_set_no=0,
        rev_at_epoch=0,
        object_name="",
    )
    assert omm.catalog_number == MAX_CATALOG_NUMBER
