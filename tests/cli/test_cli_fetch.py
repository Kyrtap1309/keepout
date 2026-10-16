import gzip
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Final

import pytest
from cli_builders import (
    FIXTURE_CSV,
    NO_SPACE_TRACK_ENV,
    SPACE_TRACK_ENV,
    FakeWeb,
    celestrak_pages,
    fake_space_track,
    keepout,
)
from spacetrack_builders import (
    IDENTITY,
    PASSWORD,
    FakeSpaceTrack,
    public_cdm,
    space_track_cdm_public,
    space_track_gp,
)

from keepout import cli
from keepout.cli import fetch
from keepout.cli.output import ExitCode
from keepout.omm import load
from keepout.sources import cara, celestrak, kelvins, spacetrack

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
    assert "expected sha256" in result.stderr


def test_fetch_refuses_a_file_as_the_cache(tmp_path: Path) -> None:
    file = tmp_path / "file"
    file.write_text("")
    result = keepout("fetch", "cara", "--cache-dir", str(file))
    assert result.exit_code == ExitCode.USAGE
    assert "Invalid value for '--cache-dir'" in result.stderr


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


FIXTURE_OBJECTS = len(load(FIXTURE_CSV))


def test_fetch_catalog_downloads_every_celestrak_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = celestrak_pages()
    web = FakeWeb(monkeypatch, pages)
    result = keepout("fetch", "catalog", "--cache-dir", str(tmp_path))
    assert result.exit_code == ExitCode.OK
    assert sorted(web.urls) == sorted(pages)
    assert "CelesTrak public GP groups" in result.stdout
    assert result.stdout.endswith(
        f"{FIXTURE_OBJECTS} objects, one element set each (0 older sets of "
        "objects in several groups left out); cached in "
        f"{tmp_path / 'celestrak' / 'gp'}\n"
    )


def test_a_second_fetch_catalog_is_served_from_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = celestrak_pages()
    web = FakeWeb(monkeypatch, pages)
    first = keepout("fetch", "catalog", "--cache-dir", str(tmp_path))
    assert first.exit_code == ExitCode.OK
    again = keepout("fetch", "catalog", "--cache-dir", str(tmp_path))
    assert again.exit_code == ExitCode.OK
    assert len(web.urls) == len(pages)


def test_fetch_catalog_as_json_lists_each_group_and_its_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeWeb(monkeypatch, celestrak_pages())
    result = keepout("fetch", "catalog", "--cache-dir", str(tmp_path), "--json")
    assert result.exit_code == ExitCode.OK
    payload = json.loads(result.stdout)
    assert payload["source"] == "celestrak"
    assert payload["objects"] == FIXTURE_OBJECTS
    assert payload["duplicates"] == 0
    groups = {group["name"]: group for group in payload["groups"]}
    assert list(groups) == list(celestrak.PUBLIC_GROUPS)
    active = groups["active"]
    assert active["objects"] == FIXTURE_OBJECTS
    assert active["url"] == celestrak.group_url("active")
    assert active["path"] == str(tmp_path / "celestrak" / "gp" / "active.csv")
    assert active["retrieved"] == active["checked"]
    assert groups["analyst"]["objects"] == 0


def test_fetch_catalog_reports_an_unreachable_celestrak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeWeb(monkeypatch, {})
    result = keepout("fetch", "catalog", "--cache-dir", str(tmp_path))
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr.startswith("Error: ")
    assert "celestrak.org" in result.stderr


def test_fetch_catalog_from_space_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = fake_space_track(
        monkeypatch, {spacetrack.CATALOG.url: space_track_gp(load(FIXTURE_CSV))}
    )
    args = ("fetch", "catalog", "--source", "spacetrack", "--cache-dir", str(tmp_path))
    result = keepout(*args, "--json", env=SPACE_TRACK_ENV)
    assert result.exit_code == ExitCode.OK
    assert server.calls == ["login", spacetrack.CATALOG.url, "logout"]
    payload = json.loads(result.stdout)
    assert payload["source"] == "spacetrack"
    assert payload["objects"] == FIXTURE_OBJECTS
    (group,) = payload["groups"]
    assert group["name"] == spacetrack.GROUP
    assert group["path"] == str(tmp_path / "spacetrack" / "gp" / "catalog.json")
    again = keepout(*args, env=NO_SPACE_TRACK_ENV)
    assert again.exit_code == ExitCode.OK
    assert "Space-Track full GP catalog" in again.stdout
    assert len(server.calls) == 3


def test_fetch_catalog_from_space_track_needs_an_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = fake_space_track(monkeypatch, {})
    result = keepout(
        "fetch",
        "catalog",
        "--source",
        "spacetrack",
        "--cache-dir",
        str(tmp_path),
        env=NO_SPACE_TRACK_ENV,
    )
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr.startswith("Error: no Space-Track credentials: set ")
    assert server.calls == []


def test_fetch_catalog_from_space_track_keeps_to_its_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = fake_space_track(monkeypatch, {spacetrack.CATALOG.url: b"not json"})
    args = ("fetch", "catalog", "--source", "spacetrack", "--cache-dir", str(tmp_path))
    first = keepout(*args, env=SPACE_TRACK_ENV)
    assert first.exit_code == ExitCode.ERROR
    assert first.stderr.startswith("Error: Space-Track did not send OMM JSON")
    second = keepout(*args, env=SPACE_TRACK_ENV)
    assert second.exit_code == ExitCode.ERROR
    assert second.stderr.startswith(
        "Error: Space-Track allows one gp query per 1 h; the last was sent at "
    )
    assert server.calls == ["login", spacetrack.CATALOG.url, "logout"]
    assert PASSWORD not in first.stderr + second.stderr


PUBLIC_ROWS: Final = (
    public_cdm(),
    public_cdm(
        CDM_ID="1000002",
        TCA="2026-10-16T08:00:00.000000",
        PC=None,
        EMERGENCY_REPORTABLE="N",
        SAT_2_NAME=None,
    ),
)
"""Two public CDM summaries, the later TCA first."""


def serve_public_cdms(monkeypatch: pytest.MonkeyPatch) -> FakeSpaceTrack:
    return fake_space_track(
        monkeypatch, {spacetrack.PUBLIC_CDMS.url: space_track_cdm_public(*PUBLIC_ROWS)}
    )


def test_fetch_cdm_lists_the_public_summaries_by_tca(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = serve_public_cdms(monkeypatch)
    result = keepout("fetch", "cdm", "--cache-dir", str(tmp_path), env=SPACE_TRACK_ENV)
    assert result.exit_code == ExitCode.OK
    assert server.calls == ["login", spacetrack.PUBLIC_CDMS.url, "logout"]
    lines = result.stdout.splitlines()
    assert lines[3].split() == [
        "2026-10-16",
        "08:00",
        "900",
        "/",
        "25544",
        "640",
        "-",
        "no",
    ]
    assert lines[4].split() == [
        "2026-10-17",
        "03:25",
        "900",
        "/",
        "25544",
        "640",
        "2.50e-05",
        "yes",
    ]
    assert lines[-1].startswith("2 CDM summaries, retrieved ")


def test_a_second_fetch_cdm_needs_neither_space_track_nor_an_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = serve_public_cdms(monkeypatch)
    cache = str(tmp_path)
    first = keepout("fetch", "cdm", "--cache-dir", cache, env=SPACE_TRACK_ENV)
    assert first.exit_code == ExitCode.OK
    again = keepout("fetch", "cdm", "--cache-dir", cache, env=NO_SPACE_TRACK_ENV)
    assert again.exit_code == ExitCode.OK
    assert len(server.calls) == 3


def test_fetch_cdm_as_json_gives_every_field_of_a_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve_public_cdms(monkeypatch)
    result = keepout(
        "fetch", "cdm", "--cache-dir", str(tmp_path), "--json", env=SPACE_TRACK_ENV
    )
    assert result.exit_code == ExitCode.OK
    payload = json.loads(result.stdout)
    assert payload["source"] == "spacetrack"
    assert payload["retrieval"]["url"] == spacetrack.PUBLIC_CDMS.url
    assert [cdm["cdm_id"] for cdm in payload["cdms"]] == [1000002, 1000001]
    assert payload["cdms"][1] == {
        "cdm_id": 1000001,
        "created": "2026-10-15T06:12:44.000",
        "tca": "2026-10-17T03:25:31.250",
        "emergency_reportable": True,
        "min_range": 640.0,
        "pc": 2.5e-05,
        "object1": {
            "catalog_number": 900,
            "name": "CALSPHERE 1",
            "object_type": "PAYLOAD",
            "rcs_size": "MEDIUM",
            "exclusion_volume": "5.00",
        },
        "object2": {
            "catalog_number": 25544,
            "name": "ISS (ZARYA)",
            "object_type": "PAYLOAD",
            "rcs_size": "LARGE",
            "exclusion_volume": "5.00",
        },
    }
    assert payload["cdms"][0]["pc"] is None


def test_a_bug_shows_its_traceback_without_the_password(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # As the console script runs it: main(), and the bug reaches the
    # interpreter's excepthook, which Typer replaces with its pretty one.
    def load_catalog(source: object, cache: object) -> None:
        password = os.environ[spacetrack.PASSWORD_VARIABLE]
        raise RuntimeError(f"a bug with {len(password)} characters at hand")

    monkeypatch.setattr(fetch, "load_catalog", load_catalog)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setenv(spacetrack.IDENTITY_VARIABLE, IDENTITY)
    monkeypatch.setenv(spacetrack.PASSWORD_VARIABLE, PASSWORD)
    for variable in ("TYPER_STANDARD_TRACEBACK", "_TYPER_STANDARD_TRACEBACK"):
        monkeypatch.delenv(variable, raising=False)
    args = ("fetch", "catalog", "--source", "spacetrack", "--cache-dir", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["keepout", *args])
    with pytest.raises(RuntimeError) as bug:
        cli.main()
    sys.excepthook(bug.type, bug.value, bug.tb)
    captured = capsys.readouterr()
    assert "Traceback" in captured.err
    assert "RuntimeError: a bug with 17 characters at hand" in captured.err
    assert "locals" not in captured.err
    assert PASSWORD not in captured.out + captured.err


def test_the_cache_variable_is_checked_and_shown_in_the_help(tmp_path: Path) -> None:
    file = tmp_path / "file"
    file.write_text("")
    result = keepout("fetch", "cara", env={"KEEPOUT_CACHE_DIR": str(file)})
    assert result.exit_code == ExitCode.USAGE
    assert "KEEPOUT_CACHE_DIR" in keepout("fetch", "cara", "--help").stdout
