import hashlib
import re
import shutil
from pathlib import Path

import pytest
from astropy.time import Time
from celestrak_builders import SAMPLE

import keepout.omm
from keepout.errors import ChecksumMismatch, SourceUnavailable
from keepout.sources.cache import CopyMeta
from keepout.sources.catalog import (
    Catalog,
    CatalogFileRef,
    CatalogPart,
    CatalogReference,
    Retrieval,
    load_reference,
)

SAMPLE_CSV = Path(__file__).parents[1] / "omm" / "data" / "celestrak_sample.csv"
URL = "https://example.org/gp?GROUP=active&FORMAT=CSV"
NOW = 1_791_000_000.0


def _copy(tmp_path: Path, name: str = "active.csv") -> Path:
    target = tmp_path / name
    shutil.copyfile(SAMPLE_CSV, target)
    return target


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reference_round_trip_and_checksum(tmp_path: Path) -> None:
    path = _copy(tmp_path)
    now = Time(NOW, format="unix", scale="utc")
    omms = keepout.omm.load(path)
    catalog = Catalog.of([CatalogPart("active", omms, Retrieval(URL, path, now, now))])
    reference = catalog.reference("celestrak")
    assert reference == CatalogReference(
        "celestrak",
        (CatalogFileRef("active", path, _sha256(path), url=URL, retrieved=now),),
    )
    assert load_reference(reference) == catalog.omms


def test_a_reference_of_a_file_names_it_by_its_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy(tmp_path)
    monkeypatch.chdir(tmp_path)
    reference = CatalogReference.of_file(Path("active.csv"))
    path = tmp_path / "active.csv"
    assert reference == CatalogReference(
        str(path), (CatalogFileRef("active.csv", path, _sha256(path)),)
    )
    numbers = {omm.catalog_number for omm in load_reference(reference)}
    assert numbers == {omm.catalog_number for omm in SAMPLE}


def test_a_reference_keeps_each_objects_newest_set_over_its_files(
    tmp_path: Path,
) -> None:
    first = CatalogReference.of_file(_copy(tmp_path, "a.csv"))
    second = CatalogReference.of_file(_copy(tmp_path, "b.csv"))
    both = CatalogReference("both", first.files + second.files)
    assert load_reference(both) == load_reference(first)


def test_reference_to_a_changed_file_raises_checksum_mismatch(tmp_path: Path) -> None:
    path = _copy(tmp_path)
    reference = CatalogReference.of_file(path)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ChecksumMismatch, match=re.escape(str(path))):
        load_reference(reference)


def test_reference_to_a_missing_file_raises_source_unavailable(tmp_path: Path) -> None:
    path = _copy(tmp_path)
    reference = CatalogReference.of_file(path)
    path.unlink()
    with pytest.raises(SourceUnavailable, match=re.escape(str(path))):
        load_reference(reference)


def test_a_reference_needs_its_files(tmp_path: Path) -> None:
    with pytest.raises(SourceUnavailable, match=r"missing\.csv"):
        CatalogReference.of_file(tmp_path / "missing.csv")
    now = Time(NOW, format="unix", scale="utc")
    gone = Retrieval(URL, tmp_path / "gone.csv", now, now)
    with pytest.raises(SourceUnavailable, match=r"gone\.csv"):
        Catalog.of([CatalogPart("active", (), gone)]).reference("celestrak")


def test_a_downloaded_file_is_referenced_by_the_digest_its_cache_verified(
    tmp_path: Path,
) -> None:
    # A cache refresh replacing the file between the fetch and the reference
    # does not make the reference name the new bytes.
    path = _copy(tmp_path)
    fetched = _sha256(path)
    meta = CopyMeta(url=URL, sha256=fetched, retrieved=NOW, checked=NOW)
    retrieval = Retrieval.of(path, meta)
    assert retrieval.sha256 == fetched
    catalog = Catalog.of([CatalogPart("active", keepout.omm.load(path), retrieval)])
    path.write_bytes(path.read_bytes() + b"\n")
    reference = catalog.reference("celestrak")
    assert [file.sha256 for file in reference.files] == [fetched]
    with pytest.raises(ChecksumMismatch, match=re.escape(str(path))):
        reference.snapshot(tmp_path / "copies")
    assert not any((tmp_path / "copies").glob("*"))
