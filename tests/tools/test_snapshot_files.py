import gzip
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest
import snapshot_files


@dataclass(frozen=True)
class Listed:
    file: str
    sha256: str


def test_a_saved_body_is_read_when_its_digest_matches(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"answer")
    listed = Listed("a.bin", hashlib.sha256(b"answer").hexdigest())
    assert snapshot_files.read_body(tmp_path, listed, "raw.json") == b"answer"


def test_a_changed_body_is_refused(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"changed")
    listed = Listed("a.bin", hashlib.sha256(b"answer").hexdigest())
    with pytest.raises(ValueError, match=r"does not match its digest in raw\.json"):
        snapshot_files.read_body(tmp_path, listed, "raw.json")


def test_digests_give_each_file_its_length_and_sha256() -> None:
    assert snapshot_files.file_digests({"a": b"xyz"}) == {
        "a": {"bytes": 3, "sha256": hashlib.sha256(b"xyz").hexdigest()}
    }


def test_only_the_named_files_are_gzipped_deterministically(tmp_path: Path) -> None:
    files = snapshot_files.stored_files(
        tmp_path, {"data.csv.gz": b"rows", "NOTICE": b"text"}, {"data.csv.gz"}
    )
    assert files["NOTICE"] == b"text"
    assert files["data.csv.gz"] == gzip.compress(b"rows", compresslevel=9, mtime=0)


def test_a_committed_gzip_with_the_same_data_is_kept_as_it_is(tmp_path: Path) -> None:
    # Another zlib build may compress the same data to other bytes.
    kept = gzip.compress(b"rows", compresslevel=1, mtime=12345)
    (tmp_path / "data.csv.gz").write_bytes(kept)
    files = snapshot_files.stored_files(
        tmp_path, {"data.csv.gz": b"rows"}, {"data.csv.gz"}
    )
    assert files["data.csv.gz"] == kept


def test_a_committed_gzip_with_other_data_is_replaced(tmp_path: Path) -> None:
    (tmp_path / "data.csv.gz").write_bytes(gzip.compress(b"old", mtime=0))
    files = snapshot_files.stored_files(
        tmp_path, {"data.csv.gz": b"rows"}, {"data.csv.gz"}
    )
    assert gzip.decompress(files["data.csv.gz"]) == b"rows"


@pytest.mark.parametrize("data", [b"not gzip", b"\x1f\x8b\x08truncated"])
def test_gunzip_of_something_else_is_none(data: bytes) -> None:
    assert snapshot_files.gunzip(data) is None


def test_json_is_written_indented_with_a_final_newline(tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    snapshot_files.write_json(path, {"format": 1})
    assert path.read_text(encoding="utf-8") == '{\n  "format": 1\n}\n'
