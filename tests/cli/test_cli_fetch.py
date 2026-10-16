import gzip
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
from cli_builders import FakeWeb, keepout

from keepout.cli.output import ExitCode
from keepout.sources import cara, kelvins

FILES = {
    "DataFiles/one.csv": b"a,b\n1,2\n",
    "DistributedMatlab/two.m": b"% two\n",
}
MANIFEST = tuple(
    cara.CaraFile(path, hashlib.sha256(data).hexdigest())
    for path, data in FILES.items()
)
CARA_PAGES = {file.url: FILES[file.path] for file in MANIFEST}


@pytest.fixture
def pinned_cara(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cara, "MANIFEST", MANIFEST)


@pytest.mark.usefixtures("pinned_cara")
def test_fetch_cara_downloads_every_file_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    web = FakeWeb(monkeypatch, CARA_PAGES)
    result = keepout("fetch", "cara", "--cache-dir", str(tmp_path))
    assert result.exit_code == ExitCode.OK
    folder = tmp_path / "cara" / cara.CARA_COMMIT
    assert result.stdout == (
        f"NASA CARA Analysis Tools, commit {cara.CARA_COMMIT[:12]}: "
        f"2 files in {folder}\n"
    )
    assert sorted(web.urls) == sorted(CARA_PAGES)
    for path, data in FILES.items():
        assert (folder / path).read_bytes() == data
    again = keepout("fetch", "cara", "--cache-dir", str(tmp_path), "--json")
    assert again.exit_code == ExitCode.OK
    assert json.loads(again.stdout) == {
        "source": "cara",
        "commit": cara.CARA_COMMIT,
        "files": 2,
        "directory": str(folder),
    }
    assert len(web.urls) == 2


@pytest.mark.usefixtures("pinned_cara")
def test_fetch_cara_uses_the_cache_of_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeWeb(monkeypatch, CARA_PAGES)
    result = keepout("fetch", "cara", env={"KEEPOUT_CACHE_DIR": str(tmp_path)})
    assert result.exit_code == ExitCode.OK
    assert str(tmp_path / "cara" / cara.CARA_COMMIT) in result.stdout


@pytest.mark.usefixtures("pinned_cara")
def test_fetch_cara_reports_a_failed_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeWeb(monkeypatch, dict(list(CARA_PAGES.items())[:1]))
    result = keepout("fetch", "cara", "--cache-dir", str(tmp_path))
    assert result.exit_code == ExitCode.ERROR
    assert result.stdout == ""
    assert result.stderr.startswith("Error: cannot download https://")


@pytest.mark.usefixtures("pinned_cara")
def test_fetch_cara_reports_a_corrupt_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeWeb(monkeypatch, dict.fromkeys(CARA_PAGES, b"tampered"))
    result = keepout("fetch", "cara", "--cache-dir", str(tmp_path))
    assert result.exit_code == ExitCode.ERROR
    assert "SHA-256" in result.stderr or "sha256" in result.stderr.lower()


def test_fetch_refuses_a_file_as_the_cache(tmp_path: Path) -> None:
    file = tmp_path / "file"
    file.write_text("")
    result = keepout("fetch", "cara", "--cache-dir", str(file))
    assert result.exit_code == ExitCode.USAGE


TABLE = gzip.compress(b"event_id,time_to_tca\n")


def zipped(member: str, data: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr(member, data)
    return buffer.getvalue()


def test_fetch_kelvins_keeps_the_raw_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = zipped(kelvins.RAW_MEMBER, TABLE)
    monkeypatch.setattr(kelvins, "ARCHIVE_SHA256", hashlib.sha256(archive).hexdigest())
    monkeypatch.setattr(kelvins, "RAW_SHA256", hashlib.sha256(TABLE).hexdigest())
    web = FakeWeb(monkeypatch, {kelvins.ARCHIVE_URL: archive})
    result = keepout("fetch", "kelvins", "--cache-dir", str(tmp_path), "--json")
    assert result.exit_code == ExitCode.OK
    path = tmp_path / "kelvins" / kelvins.ZENODO_RECORD / "raw_data_2015-2019.gz"
    assert json.loads(result.stdout) == {
        "source": "kelvins",
        "doi": kelvins.DOI,
        "path": str(path),
    }
    assert path.read_bytes() == TABLE
    again = keepout("fetch", "kelvins", "--cache-dir", str(tmp_path))
    assert again.stdout == (
        f"ESA Kelvins collision avoidance data set (doi:{kelvins.DOI}): {path}\n"
    )
    assert web.urls == [kelvins.ARCHIVE_URL]


def test_fetch_kelvins_reports_a_bad_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = zipped("other.csv", TABLE)
    monkeypatch.setattr(kelvins, "ARCHIVE_SHA256", hashlib.sha256(archive).hexdigest())
    FakeWeb(monkeypatch, {kelvins.ARCHIVE_URL: archive})
    result = keepout("fetch", "kelvins", "--cache-dir", str(tmp_path))
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr.startswith("Error: ")
    assert kelvins.RAW_MEMBER in result.stderr
