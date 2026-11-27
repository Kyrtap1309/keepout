"""The Iridium-Cosmos data tool: reading Space-Track's bulk TLE file and
choosing the excerpt, on an invented bulk file of real 2009 element sets."""

import math
import zipfile
from pathlib import Path

import iridium_cosmos as study
import iridium_cosmos_data as data_tool
import pytest
from astropy.time import Time
from sgp4.io import compute_checksum

MEMBER = "tle2009.txt"

IRIDIUM = (
    "1 24946U 97051C   09039.66818788  .00000197  00000-0  63273-4 0  4717",
    "2 24946 086.3996 122.1691 0002268 085.1934 274.9529 14.34219784597178",
)
COSMOS = (
    "1 22675U 93036A   09039.51970769 -.00000003 +00000-0 +87983-5 0 07417",
    "2 22675 074.0356 021.2473 0016045 100.8058 259.4899 14.31135543817277",
)
GEO = (
    "1 26719U 01011A   09038.00348938  .00000142  00000-0  10000-3 0  4539",
    "2 26719 000.0739 343.4591 0003765 334.3957 209.0996 01.00269866 29026",
)
START = Time("2009-01-09T00:00:00", scale="utc")
END = study.TCA_REFERENCE


def with_checksum(line: str) -> str:
    return line[:68] + str(compute_checksum(line))


def edited(line: str, column: int, text: str) -> str:
    """``line`` with ``text`` written from ``column`` (0-based), checksum redone."""
    return with_checksum(line[:column] + text + line[column + len(text) :])


def bulk(tmp_path: Path, *sets: tuple[str, str], extra: str = "") -> Path:
    """A zip like Space-Track's: line 1 of each set ends with a backslash."""
    text = "".join(f"{one}\\\n{two}\n" for one, two in sets) + extra
    path = tmp_path / "tle2009.txt.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MEMBER, text)
    return path


def test_sets_outside_the_window_are_left_out(tmp_path: Path) -> None:
    early = (edited(IRIDIUM[0], 18, "09001.00000000"), IRIDIUM[1])
    read = data_tool.read_bulk(bulk(tmp_path, IRIDIUM, COSMOS, early), START, END)
    assert [omm.catalog_number for omm in read.sets] == [22675, 24946]
    assert read.counts["sets"] == 3
    assert read.counts["sets_in_window"] == 2
    assert read.counts["objects_in_window"] == 2
    assert all(START <= omm.epoch <= END for omm in read.sets)


def test_duplicate_epochs_kept_once(tmp_path: Path) -> None:
    reissued = (edited(IRIDIUM[0], 64, "4718"), IRIDIUM[1])
    read = data_tool.read_bulk(bulk(tmp_path, reissued, IRIDIUM), START, END)
    (kept,) = read.sets
    assert kept.element_set_no == 4718
    assert read.counts["duplicates"] == 1
    assert read.counts["corrected"] == 0


def test_correction_replaces_the_earlier_set(tmp_path: Path) -> None:
    corrected = (edited(IRIDIUM[0], 64, "4718"), edited(IRIDIUM[1], 8, "086.4000"))
    read = data_tool.read_bulk(bulk(tmp_path, corrected, IRIDIUM), START, END)
    (kept,) = read.sets
    assert kept.element_set_no == 4718
    assert kept.inclination == pytest.approx(math.radians(86.4))
    assert read.counts["duplicates"] == 1
    assert read.counts["corrected"] == 1


def test_designator_without_piece_is_blanked(tmp_path: Path) -> None:
    # 2009's file has two such objects (29420 "03037", 12530 "76077").
    short = (edited(GEO[0], 9, "01011   "), GEO[1])
    read = data_tool.read_bulk(bulk(tmp_path, short), START, END)
    (omm,) = read.sets
    assert omm.international_designator == "UNKNOWN"
    assert (
        omm.mean_motion
        == data_tool.read_bulk(bulk(tmp_path, GEO), START, END).sets[0].mean_motion
    )
    assert read.counts["designators_blanked"] == [26719]


def test_malformed_line_named(tmp_path: Path) -> None:
    path = bulk(tmp_path, IRIDIUM, extra=f"{COSMOS[1]}\n")
    with pytest.raises(data_tool.BulkFormatError, match=rf"{MEMBER}:3"):
        data_tool.read_bulk(path, START, END)


def test_unreadable_set_left_out_and_listed(tmp_path: Path) -> None:
    # 2009's file has one: 23211, a B* of "359008+0".
    broken = (IRIDIUM[0][:68] + "0", IRIDIUM[1])
    read = data_tool.read_bulk(bulk(tmp_path, broken, COSMOS), START, END)
    assert [omm.catalog_number for omm in read.sets] == [22675]
    (left_out,) = read.counts["unreadable"]
    assert left_out["catalog_number"] == 24946
    assert left_out["line"] == f"{MEMBER}:1"
    assert "checksum" in left_out["reason"]


def test_excerpt_keeps_near_objects(tmp_path: Path) -> None:
    read = data_tool.read_bulk(bulk(tmp_path, IRIDIUM, COSMOS, GEO), START, END)
    sets, selection = data_tool.excerpt_sets(read.sets, workers=1)
    assert {omm.catalog_number for omm in sets} == {24946, 22675}
    assert selection["objects"] == 2
    assert selection["screened_objects"] == 3


def test_excerpt_keeps_only_the_newest_set_of_other_objects(tmp_path: Path) -> None:
    # A copy of Cosmos 2251's orbit under another number passes Iridium 33 too.
    def twin(lines: tuple[str, str]) -> tuple[str, str]:
        return edited(lines[0], 2, "99999"), edited(lines[1], 2, "99999")

    older = twin((edited(COSMOS[0], 18, "09038.51970769"), COSMOS[1]))
    later = twin((edited(COSMOS[0], 18, "09040.51970769"), COSMOS[1]))
    early_pair = (edited(COSMOS[0], 18, "09030.00000000"), COSMOS[1])
    path = bulk(tmp_path, IRIDIUM, COSMOS, early_pair, older, twin(COSMOS), later)
    read = data_tool.read_bulk(path, START, END)
    sets, selection = data_tool.excerpt_sets(read.sets, workers=1)
    kept = [(omm.catalog_number, omm.epoch.isot[:10]) for omm in sets]
    # The twin's newest set known at the decision point (02-08 12:28), not
    # its older one, nor the one after; every set of the pair.
    assert kept == [
        (22675, "2009-01-30"),
        (22675, "2009-02-08"),
        (24946, "2009-02-08"),
        (99999, "2009-02-08"),
    ]
    assert selection["objects"] == 3


def test_build_is_deterministic_and_verify_detects_edit(tmp_path: Path) -> None:
    zip_path = bulk(tmp_path, IRIDIUM, COSMOS, GEO)
    out = tmp_path / "data"
    out.mkdir()
    args = ["build", str(zip_path), "--data", str(out), "--workers", "1"]
    assert data_tool.main(args) == 0
    first = {path.name: path.read_bytes() for path in out.iterdir()}
    assert set(first) == {study.CATALOG_FILE, data_tool.NOTICE, data_tool.MANIFEST}
    assert data_tool.main(["verify", "--data", str(out)]) == 0
    assert data_tool.main([*args, "--check"]) == 0
    assert {path.name: path.read_bytes() for path in out.iterdir()} == first
    (out / data_tool.NOTICE).write_text("edited\n", encoding="utf-8")
    assert data_tool.main(["verify", "--data", str(out)]) == 1
