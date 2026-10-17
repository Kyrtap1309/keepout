import dataclasses
import io
import json
from pathlib import Path

import pytest
from astropy.time import Time
from hypothesis import given, settings
from hypothesis import strategies as st
from omm_builders import (
    ISS,
    SAMPLE_CSV,
    SAMPLE_JSON,
    assert_same_satrec,
    csv_text,
    json_text,
    records,
    sgp4_record,
)
from sgp4 import omm as sgp4_omm

from keepout.ccsds.keywords import parse_epoch
from keepout.errors import OmmParseError
from keepout.omm import Omm, celestrak
from keepout.omm.celestrak import (
    dumps_csv,
    dumps_json,
    load_csv,
    load_json,
    loads_csv,
    loads_json,
)


def test_the_sample_csv_is_read_in_file_order() -> None:
    omms = load_csv(SAMPLE_CSV)
    assert [omm.catalog_number for omm in omms] == [
        900,
        25544,
        51850,
        30798,
        38745,
        100000,
        100001,
        100882,
    ]
    assert omms[1].object_name == "ISS (ZARYA)"
    assert omms[6].bstar == -0.14182159e-2
    assert omms[4].mean_motion_ddot > 0.0


def test_the_sample_json_holds_the_same_objects_as_the_csv() -> None:
    assert load_json(SAMPLE_JSON) == load_csv(SAMPLE_CSV)


def test_every_sample_record_is_the_one_sgp4_makes_from_the_file() -> None:
    with SAMPLE_CSV.open(newline="") as stream:
        fields = list(sgp4_omm.parse_csv(stream))
    omms = load_csv(SAMPLE_CSV)
    assert len(omms) == len(fields) == 8
    for omm, values in zip(omms, fields, strict=True):
        assert_same_satrec(omm.satrec(), sgp4_record(values))


def test_a_file_with_a_byte_order_mark_is_read(tmp_path: Path) -> None:
    path = tmp_path / "gp.csv"
    path.write_bytes(b"\xef\xbb\xbf" + SAMPLE_CSV.read_bytes())
    assert load_csv(path) == load_csv(SAMPLE_CSV)
    path.write_bytes(b"\xef\xbb\xbf" + SAMPLE_JSON.read_bytes())
    assert load_json(path) == load_json(SAMPLE_JSON)


def test_csv_columns_may_come_in_any_order_with_extra_ones() -> None:
    reordered = {"EXTRA": "x", **dict(reversed(ISS.items()))}
    assert loads_csv(csv_text([reordered])) == loads_csv(csv_text([ISS]))


def test_blank_csv_lines_are_skipped_and_quoted_names_read() -> None:
    named = {**ISS, "OBJECT_NAME": 'A, "B"'}
    text = csv_text([named]).replace("\n", "\r\n") + "\r\n\r\n"
    (omm,) = loads_csv(text)
    assert omm.object_name == 'A, "B"'


def test_a_header_without_rows_gives_no_objects() -> None:
    assert loads_csv(csv_text([])) == ()
    assert loads_json("[]") == ()


def test_space_track_json_is_read() -> None:
    # Every value a string, more keywords, nulls among them.
    record = {
        "CCSDS_OMM_VERS": "3.0",
        "CENTER_NAME": "EARTH",
        "REF_FRAME": "TEME",
        "TIME_SYSTEM": "UTC",
        "MEAN_ELEMENT_THEORY": "SGP4",
        "DECAY_DATE": None,
        "TLE_LINE1": "1 25544U ...",
        **ISS,
    }
    assert loads_json(json.dumps([record])) == loads_csv(csv_text([ISS]))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", r"no CSV header \(line 1\)"),
        (
            "OBJECT_NAME,OBJECT_NAME\n",
            r"repeated column \(line 1, keyword OBJECT_NAME\)",
        ),
        ("No GP data found\n", r"missing column \(line 1, keyword OBJECT_NAME\)"),
        (
            csv_text([ISS]) + "ISS,1998-067A\n",
            r"2 values for 17 columns \(record 2, line 3\)",
        ),
        (
            csv_text([{**ISS, "MEAN_MOTION": "fast"}]),
            r"invalid number 'fast' \(record 1, line 2, keyword MEAN_MOTION\)",
        ),
        (
            csv_text([{**ISS, "ECCENTRICITY": "1.2"}]),
            (
                r"eccentricity must be in \[0, 1\), got 1.2 "
                r"\(record 1, line 2, keyword ECCENTRICITY\)"
            ),
        ),
        (
            csv_text([{**ISS, "EPOCH": "2026-02-30T00:00:00.000000"}]),
            r"(?s)invalid CCSDS epoch .* \(record 1, line 2, keyword EPOCH\)",
        ),
        (
            csv_text([{**ISS, "NORAD_CAT_ID": "25544.0"}]),
            r"invalid integer '25544.0' .*keyword NORAD_CAT_ID",
        ),
        (
            csv_text([{**ISS, "EPHEMERIS_TYPE": "4"}]),
            r"ephemeris type 4 is not SGP4 \(0\) .*keyword EPHEMERIS_TYPE",
        ),
        (
            csv_text([{**ISS, "REF_FRAME": "GCRF"}]),
            r"must be TEME, got 'GCRF' \(record 1, line 2, keyword REF_FRAME\)",
        ),
        (
            csv_text([ISS]) + '"unterminated\n',
            r"invalid CSV: unexpected end of data \(line 3\)",
        ),
        # A row before broken quoting is read, and refused, first.
        (
            csv_text([ISS, {**ISS, "BSTAR": "x"}]) + '"unterminated\n',
            r"invalid number 'x' \(record 2, line 3, keyword BSTAR\)",
        ),
        # An invalid epoch is reported in its row, after an earlier row's error.
        (
            csv_text(
                [
                    {**ISS, "EPOCH": "2026-01-01T00:00:00.000000"},
                    {**ISS, "INCLINATION": "x"},
                    {**ISS, "EPOCH": "2026-02-30T00:00:00.000000"},
                ]
            ),
            r"invalid number 'x' \(record 2, line 3, keyword INCLINATION\)",
        ),
    ],
)
def test_invalid_csv_is_refused_with_its_place(text: str, message: str) -> None:
    with pytest.raises(OmmParseError, match=message):
        loads_csv(text)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("No GP data found", r"invalid JSON: Expecting value \(line 1\)"),
        ("{}", "JSON must be an array of OMM objects"),
        ("[[]]", r"not a JSON object \(record 1\)"),
        (
            json.dumps([ISS, {key: ISS[key] for key in list(ISS)[:-1]}]),
            r"missing keyword \(record 2, keyword MEAN_MOTION_DDOT\)",
        ),
        (
            json.dumps([{**ISS, "BSTAR": None}]),
            r"must be text, got None \(record 1, keyword BSTAR\)",
        ),
        (
            json.dumps([{**ISS, "BSTAR": True}]),
            r"must be text, got True \(record 1, keyword BSTAR\)",
        ),
        (
            json_text([{**ISS, "BSTAR": "NaN"}]),
            r"invalid number 'NaN' \(record 1, keyword BSTAR\)",
        ),
        (
            json_text([{**ISS, "MEAN_MOTION": "1e400"}]),
            r"number out of range '1e400' \(record 1, keyword MEAN_MOTION\)",
        ),
    ],
)
def test_invalid_json_is_refused_with_its_place(text: str, message: str) -> None:
    with pytest.raises(OmmParseError, match=message):
        loads_json(text)


def test_an_invalid_epoch_is_refused_with_its_place_among_valid_ones() -> None:
    rows = [{**ISS, "EPOCH": f"2026-02-{day:02d}T00:00:00.000000"} for day in (1, 30)]
    message = r"(?s)invalid CCSDS epoch .* \(record 2, keyword EPOCH\)"
    with pytest.raises(OmmParseError, match=message):
        loads_json(json_text(rows))
    with pytest.raises(
        OmmParseError, match=message.replace("record 2", "record 2, line 3")
    ):
        loads_csv(csv_text(rows))


@settings(deadline=None, max_examples=50)
@given(st.lists(records(), min_size=1, max_size=30))
def test_epochs_read_from_a_catalog_are_those_parsed_one_by_one(
    values: list[dict[str, str]],
) -> None:
    rows = [
        {**row, "EPOCH": row["EPOCH"][: 20 + index % 8].removesuffix(".")}
        for index, row in enumerate(values)
    ]
    for omms in (loads_csv(csv_text(rows)), loads_json(json_text(rows))):
        for omm, row in zip(omms, rows, strict=True):
            expected = parse_epoch(row["EPOCH"])
            epoch = omm.epoch
            assert (epoch.jd1, epoch.jd2) == (expected.jd1, expected.jd2)
            assert (epoch.precision, epoch.format) == (
                expected.precision,
                expected.format,
            )


def test_writers_give_back_the_sample() -> None:
    omms = load_csv(SAMPLE_CSV)
    assert loads_csv(dumps_csv(omms)) == omms
    assert loads_json(dumps_json(omms)) == omms
    assert dumps_csv(omms).splitlines()[0] == SAMPLE_CSV.read_text().splitlines()[0]
    assert json.loads(dumps_json(omms)) == json.loads(SAMPLE_JSON.read_text())


def test_the_module_is_exported() -> None:
    assert celestrak.loads_csv is loads_csv


@settings(deadline=None)
@given(st.lists(records(), max_size=4))
def test_csv_and_json_of_the_same_values_give_equal_elements(
    values: list[dict[str, str]],
) -> None:
    assert loads_csv(csv_text(values)) == loads_json(json_text(values))


@settings(deadline=None)
@given(records())
def test_satrec_is_bit_identical_to_sgp4s_own_initialisation(
    values: dict[str, str],
) -> None:
    (omm,) = loads_csv(csv_text([values]))
    assert_same_satrec(omm.satrec(), sgp4_record(values))
    (from_json,) = loads_json(json_text([values]))
    assert_same_satrec(from_json.satrec(), sgp4_record(values))


@settings(deadline=None)
@given(st.lists(records(), max_size=4))
def test_written_elements_read_back_equal(values: list[dict[str, str]]) -> None:
    omms: tuple[Omm, ...] = loads_csv(csv_text(values))
    assert loads_csv(dumps_csv(omms)) == omms
    assert loads_json(dumps_json(omms)) == omms


@pytest.mark.parametrize("precision", [3, 9])
def test_written_epochs_do_not_depend_on_the_time_precision(precision: int) -> None:
    (iss,) = loads_csv(csv_text([ISS]))
    omm = dataclasses.replace(iss, epoch=Time(iss.epoch, precision=precision))
    text = dumps_csv([omm])
    assert text == dumps_csv([iss])
    assert dumps_json([omm]) == dumps_json([iss])
    assert loads_csv(text) == (omm,)
    assert loads_json(dumps_json([omm])) == (omm,)
    # sgp4 reads the written CSV, and makes the same record of it.
    (fields,) = sgp4_omm.parse_csv(io.StringIO(text))
    assert_same_satrec(omm.satrec(), sgp4_record(fields))


def test_an_epoch_finer_than_microseconds_is_written_rounded() -> None:
    (iss,) = loads_csv(csv_text([ISS]))
    finer = dataclasses.replace(
        iss, epoch=Time("2026-10-06T00:20:47.8605118", scale="utc", precision=7)
    )
    assert dumps_csv([finer]) == dumps_csv([iss])
    assert dumps_json([finer]) == dumps_json([iss])
