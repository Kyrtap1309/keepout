import dataclasses
from pathlib import Path

import pytest
from astropy.time import Time
from hypothesis import given, settings
from hypothesis import strategies as st
from omm_builders import (
    CCSDS_KVN,
    GOES_9,
    ISS,
    SAMPLE_CSV,
    assert_same_satrec,
    csv_text,
    records,
    sgp4_record,
)

from keepout.errors import OmmParseError
from keepout.omm import kvn
from keepout.omm.celestrak import load_csv, loads_csv

CREATED = Time("2026-10-06T12:00:00", scale="utc", precision=0)

ISS_KVN = """\
CCSDS_OMM_VERS      = 3.0
CREATION_DATE       = 2026-10-06T12:00:00
ORIGINATOR          = UNKNOWN

OBJECT_NAME         = ISS (ZARYA)
OBJECT_ID           = 1998-067A
CENTER_NAME         = EARTH
REF_FRAME           = TEME
TIME_SYSTEM         = UTC
MEAN_ELEMENT_THEORY = SGP4

EPOCH               = 2026-10-06T00:20:47.860512
MEAN_MOTION         = 15.48747543 [rev/day]
ECCENTRICITY        = 0.00068585
INCLINATION         = 51.6314 [deg]
RA_OF_ASC_NODE      = 111.6262 [deg]
ARG_OF_PERICENTER   = 227.8987 [deg]
MEAN_ANOMALY        = 132.1419 [deg]

EPHEMERIS_TYPE      = 0
CLASSIFICATION_TYPE = U
NORAD_CAT_ID        = 25544
ELEMENT_SET_NO      = 999
REV_AT_EPOCH        = 58893
BSTAR               = 0.00010208237 [1/ER]
MEAN_MOTION_DOT     = 5.131E-05 [rev/day**2]
MEAN_MOTION_DDOT    = 0.0 [rev/day**3]
"""
"""The ISS sample row as written by keepout."""


def iss_kvn(old: str = "", new: str = "") -> str:
    """:data:`ISS_KVN` with ``old`` replaced by ``new`` (``old`` must occur)."""
    assert ISS_KVN.count(old) >= 1
    return ISS_KVN.replace(old, new, 1)


def test_the_ccsds_examples_are_read() -> None:
    (goes,) = loads_csv(csv_text([GOES_9]))
    for path in CCSDS_KVN:
        assert kvn.load(path) == (goes,), path.name


def test_the_ccsds_example_initialises_sgp4_like_its_values() -> None:
    (goes,) = kvn.load(CCSDS_KVN[0])
    assert_same_satrec(goes.satrec(), sgp4_record(GOES_9))


def test_writer_layout_and_units() -> None:
    (iss,) = loads_csv(csv_text([ISS]))
    assert kvn.dumps([iss], creation_date=CREATED) == ISS_KVN
    assert kvn.loads(ISS_KVN) == (iss,)


def test_the_writer_names_the_originator_and_defaults_to_now(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_now(cls: type[Time]) -> Time:
        return cls("2026-10-06T12:34:56.25", scale="utc")

    monkeypatch.setattr(Time, "now", classmethod(fake_now))
    (iss,) = kvn.loads(ISS_KVN)
    text = kvn.dumps([iss], originator="18 SDS")
    assert "ORIGINATOR          = 18 SDS\n" in text
    created = text.splitlines()[1].split("=")[1].strip()
    assert created == "2026-10-06T12:34:56"  # the clock's time, whole seconds


def test_several_messages_are_read_in_order() -> None:
    omms = load_csv(SAMPLE_CSV)
    text = kvn.dumps(omms, creation_date=CREATED)
    assert text.count("CCSDS_OMM_VERS") == len(omms)
    assert kvn.loads(text) == omms
    joined = "\n".join(path.read_text() for path in CCSDS_KVN)
    assert len(kvn.loads(joined)) == len(CCSDS_KVN)


def test_blank_text_holds_no_messages() -> None:
    assert kvn.loads("") == ()
    assert kvn.loads("\n  \n") == ()
    assert kvn.dumps([]) == ""


def test_version_2_is_read() -> None:
    assert kvn.loads(iss_kvn("= 3.0", "= 2.0")) == kvn.loads(ISS_KVN)


def test_omitted_tle_parameters_take_the_ccsds_defaults() -> None:
    text = iss_kvn("EPHEMERIS_TYPE      = 0\n").replace("CLASSIFICATION_TYPE = U\n", "")
    assert kvn.loads(text) == kvn.loads(ISS_KVN)


def test_comments_and_other_keywords_are_read_past() -> None:
    text = iss_kvn(
        "OBJECT_NAME",
        "COMMENT a remark\nCOMMENT\nGM = 398600.8 [km**3/s**2]\nOBJECT_NAME",
    )
    assert kvn.loads(text) == kvn.loads(ISS_KVN)


def test_text_values_keep_brackets() -> None:
    (omm,) = kvn.loads(iss_kvn("= ISS (ZARYA)", "= GOES 9 [P]"))
    assert omm.object_name == "GOES 9 [P]"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            "OBJECT_NAME = ISS\n",
            r"CCSDS_OMM_VERS must be the first keyword \(line 1, keyword OBJECT_NAME\)",
        ),
        (
            "COMMENT first\n" + ISS_KVN,
            r"CCSDS_OMM_VERS must be the first keyword \(line 1\)",
        ),
        (
            "\nCCSDS_OMM_VERS = 1.0\n",
            r"unsupported OMM version '1.0', expected 2.0 or 3.0 "
            r"\(record 1, line 2, keyword CCSDS_OMM_VERS\)",
        ),
        (
            ISS_KVN + "\nCCSDS_OMM_VERS = 3.0\nmean_motion = 1\n",
            r"expected 'KEYWORD = value', got 'mean_motion = 1' "
            r"\(record 2, line 30\)",
        ),
        (
            iss_kvn("= 999\n", "= 999\nELEMENT_SET_NO = 1\n"),
            r"duplicate keyword \(record 1, line 24, keyword ELEMENT_SET_NO\)",
        ),
        (
            ISS_KVN + "\n" + iss_kvn("BSTAR               = 0.00010208237 [1/ER]\n"),
            r"missing keyword \(record 2, line 29, keyword BSTAR\)",
        ),
        (
            iss_kvn("[rev/day]", "[rev/s]"),
            r"unit \[rev/s\] does not match expected \[rev/day\] "
            r"\(record 1, line 13, keyword MEAN_MOTION\)",
        ),
        (
            iss_kvn("= 0.00068585", "= 0.000685.85"),
            r"invalid number '0.000685.85' \(record 1, line 14, keyword ECCENTRICITY\)",
        ),
        (
            iss_kvn("= 51.6314", "= 181"),
            r"inclination must be in \[0, pi\].* \(record 1, line 15, keyword "
            r"INCLINATION\)",
        ),
        (
            iss_kvn("= TEME", "= GCRF"),
            r"must be TEME, got 'GCRF' \(record 1, line 8, keyword REF_FRAME\)",
        ),
        (
            iss_kvn("= SGP4", "= SGP4-XP"),
            r"must be SGP4 or SGP/SGP4, got 'SGP4-XP' "
            r"\(record 1, line 10, keyword MEAN_ELEMENT_THEORY\)",
        ),
        (
            iss_kvn("EPHEMERIS_TYPE      = 0", "EPHEMERIS_TYPE = 4"),
            r"ephemeris type 4 is not SGP4 \(0\) \(record 1, line 20, keyword "
            r"EPHEMERIS_TYPE\)",
        ),
    ],
)
def test_invalid_kvn_is_refused_with_its_place(text: str, message: str) -> None:
    with pytest.raises(OmmParseError, match=message):
        kvn.loads(text)


@pytest.mark.parametrize("name", ["A\nB", "A\rB", "A\u2028B"])
def test_a_line_break_cannot_be_written(name: str) -> None:
    (iss,) = kvn.loads(ISS_KVN)
    with pytest.raises(ValueError, match="OBJECT_NAME: KVN cannot hold a line break"):
        kvn.dumps([dataclasses.replace(iss, object_name=name)])
    with pytest.raises(ValueError, match="ORIGINATOR: KVN cannot hold a line break"):
        kvn.dumps([iss], originator=name)


def test_a_file_is_read(tmp_path: Path) -> None:
    path = tmp_path / "omm.kvn"
    path.write_text(ISS_KVN, encoding="utf-8")
    assert kvn.load(path) == kvn.loads(ISS_KVN)


@settings(deadline=None)
@given(st.lists(records(), max_size=4))
def test_written_elements_read_back_equal(values: list[dict[str, str]]) -> None:
    omms = loads_csv(csv_text(values))
    assert kvn.loads(kvn.dumps(omms, creation_date=CREATED)) == omms


@pytest.mark.parametrize("precision", [3, 9])
def test_written_epochs_do_not_depend_on_the_time_precision(precision: int) -> None:
    (iss,) = loads_csv(csv_text([ISS]))
    omm = dataclasses.replace(iss, epoch=Time(iss.epoch, precision=precision))
    text = kvn.dumps([omm], creation_date=CREATED)
    assert text == kvn.dumps([iss], creation_date=CREATED)
    assert kvn.loads(text) == (omm,)


def test_an_epoch_finer_than_microseconds_is_written_and_read_exactly() -> None:
    (iss,) = loads_csv(csv_text([ISS]))
    epoch = "2026-10-06T00:20:47.860512345"
    omm = dataclasses.replace(iss, epoch=Time(epoch, scale="utc", precision=3))
    text = kvn.dumps([omm], creation_date=CREATED)
    assert f"EPOCH               = {epoch}\n" in text
    assert kvn.loads(text) == (omm,)
