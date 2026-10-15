import gzip
from pathlib import Path

import pytest
from astropy.time import Time
from omm_builders import (
    CCSDS_KVN,
    CCSDS_XML,
    GOES_9_TLE,
    SAMPLE_CSV,
    SAMPLE_JSON,
)

from keepout.errors import OmmParseError
from keepout.omm import celestrak, kvn, load, loads, tle, xml

CREATED = Time("2026-10-06T12:00:00", scale="utc", precision=0)

SAMPLE = celestrak.load_csv(SAMPLE_CSV)


@pytest.mark.parametrize(
    "text",
    [
        SAMPLE_CSV.read_text(),
        SAMPLE_JSON.read_text(),
        kvn.dumps(SAMPLE, creation_date=CREATED),
        xml.dumps(SAMPLE, creation_date=CREATED),
    ],
    ids=["csv", "json", "kvn", "xml"],
)
def test_every_format_is_recognised(text: str) -> None:
    assert loads(text) == SAMPLE
    assert loads("\ufeff\n  \n" + text) == SAMPLE


@pytest.mark.parametrize(
    "path", [*CCSDS_KVN, CCSDS_XML], ids=lambda path: Path(path).name
)
def test_the_ccsds_examples_are_recognised(path: Path) -> None:
    assert load(path) == (kvn.load(path) if path.suffix == ".kvn" else xml.load(path))


def test_two_and_three_line_tles_are_recognised() -> None:
    assert loads(GOES_9_TLE) == tle.loads(GOES_9_TLE)
    two_lines = "".join(GOES_9_TLE.splitlines(keepends=True)[1:])
    assert loads(two_lines) == tle.loads(two_lines)


def test_line_numbers_count_from_the_first_line_of_kvn_and_tle() -> None:
    with pytest.raises(OmmParseError, match=r"line 4\)"):
        loads("\n\n" + GOES_9_TLE.replace("9250", "9251"))


def test_a_gzip_file_is_read_whatever_its_name(tmp_path: Path) -> None:
    path = tmp_path / "catalog.csv"
    path.write_bytes(gzip.compress(SAMPLE_CSV.read_bytes()))
    assert load(path) == SAMPLE
    assert celestrak.load_csv(path) == SAMPLE
    named = tmp_path / "catalog.csv.gz"
    named.write_bytes(gzip.compress(GOES_9_TLE.encode()))
    assert load(named) == tle.loads(GOES_9_TLE)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (gzip.compress(b"OBJECT_NAME", mtime=0)[:-4], "invalid gzip data"),
        (b"\x1f\x8bnot gzip", "invalid gzip data"),
        (b"OBJECT_NAME,\xff\n", "not UTF-8 text"),
    ],
    ids=["truncated gzip", "not gzip", "not UTF-8"],
)
def test_broken_files_are_refused(tmp_path: Path, data: bytes, message: str) -> None:
    path = tmp_path / "broken"
    path.write_bytes(data)
    with pytest.raises(OmmParseError, match=f"broken: {message}"):
        load(path)


@pytest.mark.parametrize("text", ["", "No GP data found", "OBJECT_NAME = ISS"])
def test_unknown_data_is_refused(text: str) -> None:
    with pytest.raises(OmmParseError, match="not OMM data"):
        loads(text)
