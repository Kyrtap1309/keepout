"""The whole flow on the command line, with fake servers: fetch the catalog,
screen a fleet against it writing CDMs, assess the CDMs."""

import json
from pathlib import Path

import pytest
from cli_builders import (
    DAYS,
    FIXTURE_CSV,
    NO_SPACE_TRACK_ENV,
    SPACE_TRACK_ENV,
    START,
    FakeWeb,
    celestrak_pages,
    fake_space_track,
    keepout,
    write_fleet,
)
from spacetrack_builders import space_track_gp

from keepout.cli.output import ExitCode
from keepout.omm import load
from keepout.sources import spacetrack


def screen_and_assess(folder: Path, source: str) -> list[str]:
    """``screen --source SOURCE --cdm-dir``, then ``assess`` its CDMs twice:
    as they are, and with a hard-body radius making one approach red; gives
    the texts of the CDMs."""
    cache = folder / "cache"
    cdms = folder / f"cdms-{source}"
    screened = keepout(
        "screen",
        "--fleet",
        str(write_fleet(folder)),
        "--source",
        source,
        "--start",
        START,
        "--days",
        DAYS,
        "--workers",
        "1",
        "--cdm-dir",
        str(cdms),
        "--cache-dir",
        str(cache),
        "--json",
        env=NO_SPACE_TRACK_ENV,
    )
    assert screened.exit_code == ExitCode.OK
    events = json.loads(screened.stdout)["events"]
    files = [event["cdm_file"] for event in events]
    assert sorted(files) == sorted(str(path) for path in cdms.iterdir())
    assessed = keepout("assess", *files, "--json")
    assert assessed.exit_code == ExitCode.OK
    payload = json.loads(assessed.stdout)
    assert payload["maneuver"] is False
    assert [e["action"] for e in payload["events"]] == ["ignore", "monitor"]
    assert [e["pc"] for e in payload["events"]] == [e["pc"] for e in events]
    red = keepout("assess", *files, "--hbr", "1000")
    assert red.exit_code == ExitCode.MANEUVER
    assert "2 events from 2 CDMs: 1 maneuver, 1 prepare maneuver." in red.stdout
    return [Path(file).read_text(encoding="utf-8") for file in sorted(files)]


def test_fetch_screen_assess_from_celestrak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = celestrak_pages()
    web = FakeWeb(monkeypatch, pages)
    cache = str(tmp_path / "cache")
    fetched = keepout("fetch", "catalog", "--cache-dir", cache)
    assert fetched.exit_code == ExitCode.OK
    assert len(web.urls) == len(pages)
    texts = screen_and_assess(tmp_path, "celestrak")
    assert len(web.urls) == len(pages)  # screen used the cached catalog
    assert len(texts) == 2


def test_fetch_screen_assess_from_space_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = fake_space_track(
        monkeypatch, {spacetrack.CATALOG.url: space_track_gp(load(FIXTURE_CSV))}
    )
    cache = str(tmp_path / "cache")
    fetched = keepout(
        "fetch",
        "catalog",
        "--source",
        "spacetrack",
        "--cache-dir",
        cache,
        env=SPACE_TRACK_ENV,
    )
    assert fetched.exit_code == ExitCode.OK
    texts = screen_and_assess(tmp_path, "spacetrack")
    assert server.calls == ["login", spacetrack.CATALOG.url, "logout"]
    FakeWeb(monkeypatch, celestrak_pages())
    assert keepout("fetch", "catalog", "--cache-dir", cache).exit_code == ExitCode.OK
    assert screen_and_assess(tmp_path, "celestrak") == texts
