import argparse
import gzip
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import pytest
import snapshot_files
from celestrak_builders import SAMPLE
from web_builders import digest

from keepout.omm import Omm


@dataclass(frozen=True)
class Listed:
    file: str
    sha256: str


@dataclass(frozen=True, slots=True)
class Answer:
    url: str
    size: int
    redacted: bool


INDEX = snapshot_files.RawIndex("answers", Answer)
ANSWERS = (Answer("https://a", 1, False), Answer("https://b", 22, True))


def test_a_saved_body_is_read_when_its_digest_matches(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"answer")
    listed = Listed("a.bin", digest(b"answer"))
    assert snapshot_files.read_body(tmp_path, listed) == b"answer"


def test_a_changed_body_is_refused(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"changed")
    listed = Listed("a.bin", digest(b"answer"))
    with pytest.raises(ValueError, match=r"does not match its digest in raw\.json"):
        snapshot_files.read_body(tmp_path, listed)


def test_a_raw_index_reads_back_what_was_written(tmp_path: Path) -> None:
    INDEX.write(tmp_path, ANSWERS)
    assert INDEX.read(tmp_path) == ANSWERS
    record = json.loads((tmp_path / snapshot_files.RAW_INDEX).read_text())
    assert record == {
        "format": 1,
        "answers": [
            {"url": "https://a", "size": 1, "redacted": False},
            {"url": "https://b", "size": 22, "redacted": True},
        ],
    }


def test_a_missing_raw_index_lists_nothing(tmp_path: Path) -> None:
    assert INDEX.read(tmp_path) == ()


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ([], "not a raw index of format 1"),
        ({"format": 2, "answers": []}, "not a raw index of format 1"),
        ({"format": 1}, "lists no answers"),
        ({"format": 1, "answers": {}}, "lists no answers"),
        ({"format": 1, "answers": ["https://a"]}, "not a Answer"),
        ({"format": 1, "answers": [{"url": "https://a", "size": 1}]}, "not a Answer"),
        (
            {
                "format": 1,
                "answers": [
                    {"url": "https://a", "size": 1, "redacted": False, "more": 0}
                ],
            },
            "not a Answer",
        ),
        (
            {
                "format": 1,
                "answers": [{"url": "https://a", "size": True, "redacted": False}],
            },
            "not a Answer",
        ),
        (
            {"format": 1, "answers": [{"url": 1, "size": 1, "redacted": False}]},
            "not a Answer",
        ),
    ],
)
def test_a_raw_index_of_another_layout_is_refused(
    tmp_path: Path, record: object, message: str
) -> None:
    (tmp_path / snapshot_files.RAW_INDEX).write_text(json.dumps(record))
    with pytest.raises(ValueError, match=message):
        INDEX.read(tmp_path)


class Manifest(TypedDict):
    format: Literal[1]
    files: dict[str, snapshot_files.FileDigest]


def test_a_manifest_is_read_when_it_has_the_layout(tmp_path: Path) -> None:
    manifest = {"format": 1, "files": {"a": {"bytes": 3, "sha256": "x"}}}
    (tmp_path / snapshot_files.MANIFEST).write_text(json.dumps(manifest))
    assert snapshot_files.read_manifest(tmp_path, Manifest) == manifest


def test_a_manifest_of_another_layout_is_refused(tmp_path: Path) -> None:
    manifest = {"format": 1, "files": {"a": {"bytes": "3", "sha256": "x"}}}
    (tmp_path / snapshot_files.MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="is not a Manifest"):
        snapshot_files.read_manifest(tmp_path, Manifest)


def test_a_catalog_csv_holds_passes_the_round_trip() -> None:
    snapshot_files.check_round_trip(SAMPLE)


def test_a_catalog_csv_cannot_hold_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(snapshot_files, "loads_csv", lambda text: ())
    with pytest.raises(ValueError, match="does not read back unchanged from CSV"):
        snapshot_files.check_round_trip(SAMPLE)


def test_catalog_counts_give_objects_regimes_and_epochs() -> None:
    def kind(omm: Omm) -> str:
        return "even" if omm.catalog_number % 2 == 0 else "odd"

    counts = snapshot_files.catalog_counts(SAMPLE, kind, ["odd", "even", "none"])
    assert counts == {
        "objects": 8,
        "six_digit": 3,
        "regimes": {"odd": 2, "even": 6, "none": 0},
        "oldest_epoch": "2026-10-03T01:10:55.327872",
        "newest_epoch": "2026-10-06T07:10:42.009312",
    }
    assert counts["oldest_epoch"] == min(omm.epoch for omm in SAMPLE).isot
    assert counts["newest_epoch"] == max(omm.epoch for omm in SAMPLE).isot


def test_digests_give_each_file_its_length_and_sha256() -> None:
    assert snapshot_files.file_digests({"a": b"xyz"}) == {
        "a": {"bytes": 3, "sha256": digest(b"xyz")}
    }


def lengths(files: Mapping[str, bytes]) -> str:
    """A manifest: the stored length of each file."""
    return json.dumps({name: len(stored) for name, stored in files.items()})


def test_only_the_named_files_are_gzipped_deterministically(tmp_path: Path) -> None:
    files = snapshot_files.stored_files(
        tmp_path, {"data.csv.gz": b"rows", "NOTICE": b"text"}, {"data.csv.gz"}, lengths
    )
    assert files["NOTICE"] == b"text"
    assert files["data.csv.gz"] == gzip.compress(b"rows", compresslevel=9, mtime=0)


def test_the_manifest_comes_last_from_the_stored_files(tmp_path: Path) -> None:
    files = snapshot_files.stored_files(
        tmp_path, {"data.csv.gz": b"rows", "NOTICE": b"text"}, {"data.csv.gz"}, lengths
    )
    assert list(files) == ["data.csv.gz", "NOTICE", snapshot_files.MANIFEST]
    stored = len(files["data.csv.gz"])
    assert json.loads(files[snapshot_files.MANIFEST]) == {
        "data.csv.gz": stored,
        "NOTICE": 4,
    }


def test_a_committed_gzip_with_the_same_data_is_kept_as_it_is(tmp_path: Path) -> None:
    # Another zlib build may compress the same data to other bytes.
    kept = gzip.compress(b"rows", compresslevel=1, mtime=12345)
    (tmp_path / "data.csv.gz").write_bytes(kept)
    files = snapshot_files.stored_files(
        tmp_path, {"data.csv.gz": b"rows"}, {"data.csv.gz"}, lengths
    )
    assert files["data.csv.gz"] == kept


def test_a_committed_gzip_with_other_data_is_replaced(tmp_path: Path) -> None:
    (tmp_path / "data.csv.gz").write_bytes(gzip.compress(b"old", mtime=0))
    files = snapshot_files.stored_files(
        tmp_path, {"data.csv.gz": b"rows"}, {"data.csv.gz"}, lengths
    )
    assert gzip.decompress(files["data.csv.gz"]) == b"rows"


def run(argv: list[str], default: Path) -> int:
    """A snapshot tool whose RAW and data hold one file, ``text``."""
    parser = argparse.ArgumentParser(prog="tool.py")
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot_files.add_data_commands(commands.add_parser, default, "bodies")
    args = parser.parse_args(argv)

    def read(directory: Path) -> str:
        return (directory / "text").read_text()

    def files(data: Path, text: str) -> dict[str, bytes]:
        return {"text": text.encode()}

    return snapshot_files.build_or_verify(args, read, read, files, "tool.py")


def test_build_writes_the_data_from_raw_and_verify_checks_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw, data = tmp_path / "raw", tmp_path / "data"
    raw.mkdir()
    (raw / "text").write_text("new")
    assert run(["build", str(raw), "--check"], data) == 1
    assert (
        capsys.readouterr().out
        == f"{data / 'text'} is out of date; run tool.py build\n"
    )
    assert run(["build", str(raw)], data) == 0
    assert (data / "text").read_text() == "new"
    assert run(["verify"], data) == 0
    assert run(["build", str(raw), "--check", "--data", str(data)], tmp_path) == 0


def test_verify_reads_the_committed_data(tmp_path: Path) -> None:
    (tmp_path / "text").write_text("committed")
    assert run(["verify", "--data", str(tmp_path)], tmp_path / "elsewhere") == 0
