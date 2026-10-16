import json
import types
from pathlib import Path

import pytest
from astropy.time import Time
from cli_builders import DAYS, FIXTURE_CSV, START, keepout, write_fleet

from keepout.cdm import kvn
from keepout.cli import screen
from keepout.cli.output import ExitCode
from keepout.omm import load
from keepout.synthetic import read_fleet
from keepout.synthetic.operator import run


def screen_fixture(folder: Path, *args: str, fleet: str | None = None) -> list[str]:
    path = write_fleet(folder) if fleet is None else write_fleet(folder, fleet)
    return [
        "screen",
        "--fleet",
        str(path),
        "--catalog",
        str(FIXTURE_CSV),
        "--start",
        START,
        "--days",
        DAYS,
        "--workers",
        "1",
        *args,
    ]


def test_screen_issues_and_assesses_a_cdm_per_approach(tmp_path: Path) -> None:
    cdms = tmp_path / "cdms"
    result = keepout(*screen_fixture(tmp_path, "--cdm-dir", str(cdms)))
    assert result.exit_code == ExitCode.OK
    lines = result.stdout.splitlines()
    assert lines[0] == (
        "Screened 2 fleet satellites against 217 objects "
        f"({FIXTURE_CSV}) from 2026-10-06T00:00:00 to 2026-10-09T00:00:00 UTC: "
        "2 close approaches."
    )
    assert lines[1] == "Close approaches"
    assert lines[2].split() == [
        "TCA",
        "(UTC)",
        "Objects",
        "Miss",
        "(m)",
        "CDMs",
        "Pc",
        "Level",
        "Action",
    ]
    assert lines[4].split() == [
        "2026-10-06",
        "19:36",
        "52939",
        "/",
        "47826",
        "1029",
        "1",
        "4.37e-08",
        "green",
        "ignore",
    ]
    assert lines[5].split() == [
        "2026-10-08",
        "12:03",
        "44476",
        "/",
        "55841",
        "3647",
        "1",
        "7.38e-06",
        "yellow",
        "monitor",
    ]
    assert lines[6] == (
        "5 element sets left out (stale, future epochs, SGP4 failures; "
        "--json lists them)."
    )
    assert lines[7].startswith(
        "44476/55841/2026-10-08T12:03:02.771792: monitor (latest Pc 7.38e-06 is "
        "yellow, 2.50 d before TCA; trend undetermined; "
    )
    assert lines[8:] == [
        "Covariances are assumed (keepout's model of GP element set errors), "
        "not orbit determination.",
        f"2 CDMs written to {cdms}",
    ]
    expected = run(
        read_fleet((tmp_path / "fleet.toml").read_text()),
        load(FIXTURE_CSV),
        Time(START, scale="utc"),
        float(DAYS) * 86400.0,
    )
    written = sorted(cdms.iterdir())
    assert [path.name for path in written] == sorted(
        f"{cdm.message_id}.kvn" for cdm in expected.cdms
    )
    by_id = {cdm.message_id: cdm for cdm in expected.cdms}
    for path in written:
        assert path.read_text(encoding="utf-8") == kvn.dumps(by_id[path.stem])


def test_screen_prints_json(tmp_path: Path) -> None:
    result = keepout(*screen_fixture(tmp_path, "--json"))
    assert result.exit_code == ExitCode.OK
    payload = json.loads(result.stdout)
    assert payload["fleet"] == [52939, 44476]
    assert payload["catalog"] == str(FIXTURE_CSV)
    assert payload["catalog_objects"] == 217
    assert payload["start"] == "2026-10-06T00:00:00.000"
    assert payload["end"] == "2026-10-09T00:00:00.000"
    assert payload["cdm_dir"] is None
    assert len(payload["skipped"]) == 5
    assert set(payload["skipped"][0]) == {"catalog_number", "role", "reason", "detail"}
    first, second = payload["events"]
    assert first["event_id"] == "52939/47826/2026-10-06T19:36:42.717332"
    assert first["object1"] == {"designator": "52939", "name": "FLEET ONE"}
    assert first["object2"]["designator"] == "47826"
    assert first["miss_distance"] == pytest.approx(1028.74, abs=0.01)
    assert first["relative_speed"] > 1000.0
    assert first["cdm_file"] is None
    assert first["level"] == "green"
    assert first["action"] == "ignore"
    assert first["trend"] == "undetermined"
    (observation,) = first["observations"]
    assert observation["message_id"] == first["cdms"][0]
    assert observation["method"] == "foster"
    assert observation["pc"] == first["pc"] == pytest.approx(4.367e-8, rel=1e-3)
    assert observation["error"] is None
    assert second["action"] == "monitor"
    assert second["level"] == "yellow"


def test_screen_starts_now_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = Time("2026-10-06T00:00:00.75", scale="utc").unix
    monkeypatch.setattr(screen, "time", types.SimpleNamespace(time=lambda: now))
    args = screen_fixture(tmp_path, "--json")
    del args[5:7]  # --start START
    result = keepout(*args)
    assert result.exit_code == ExitCode.OK
    payload = json.loads(result.stdout)
    assert payload["start"] == "2026-10-06T00:00:00.000"
    assert len(payload["events"]) == 2


def test_screen_reports_a_fleet_satellite_missing_from_the_catalog(
    tmp_path: Path,
) -> None:
    fleet = (
        "catalog_hard_body_radius = 1.0\n\n[[satellites]]\n"
        "catalog_number = 99999\nhard_body_radius = 5.0\nmaneuverable = true\n"
    )
    result = keepout(*screen_fixture(tmp_path, fleet=fleet))
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr == (
        "Error: no element set in the catalog for fleet satellite(s) 99999\n"
    )


def test_screen_lists_a_fleet_satellite_it_could_not_screen(tmp_path: Path) -> None:
    # An element set older than 30 days at the window's end is not screened:
    # the satellite gets no event and is listed among the element sets left out.
    catalog = tmp_path / "catalog.csv"
    catalog.write_text(
        FIXTURE_CSV.read_text(encoding="utf-8").replace(
            ",2026-10-05T04:28:37.438752,", ",2026-09-01T04:28:37.438752,"
        ),
        encoding="utf-8",
    )
    args = screen_fixture(tmp_path)
    args[args.index("--catalog") + 1] = str(catalog)
    text = keepout(*args)
    assert text.exit_code == ExitCode.OK
    lines = text.stdout.splitlines()
    assert lines[0] == (
        "Screened 1 of 2 fleet satellites against 216 objects "
        f"({catalog}) from 2026-10-06T00:00:00 to 2026-10-09T00:00:00 UTC: "
        "1 close approach."
    )
    assert lines[1] == (
        "Fleet satellite 44476 (INTELSAT 39 (IS-39)) was NOT screened: stale, "
        "the window ends 37.81 days after the epoch 2026-09-01T04:28:37.438752 "
        "(max_age 30 days)."
    )
    assert lines[2] == "Close approaches"
    result = keepout(*args, "--json")
    assert result.exit_code == ExitCode.OK
    payload = json.loads(result.stdout)
    assert [e["object1"]["designator"] for e in payload["events"]] == ["52939"]
    (fleet_satellite,) = [s for s in payload["skipped"] if s["role"] == "primary"]
    assert fleet_satellite["catalog_number"] == 44476
    assert fleet_satellite["reason"] == "stale"
    # It is in the catalog too, and left out there for the same reason.
    assert [
        (s["role"], s["reason"])
        for s in payload["skipped"]
        if s["catalog_number"] == 44476
    ] == [("primary", "stale"), ("catalog", "stale")]


def test_screen_reports_a_bad_fleet_file(tmp_path: Path) -> None:
    result = keepout(*screen_fixture(tmp_path, fleet="satellites = 3\n"))
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr.startswith("Error: ")
    assert "Traceback" not in result.stderr


def test_screen_reports_a_bad_catalog_file(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.csv"
    catalog.write_text("not a catalog\n", encoding="utf-8")
    args = screen_fixture(tmp_path)
    args[args.index("--catalog") + 1] = str(catalog)
    result = keepout(*args)
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr.startswith("Error: not OMM data")


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--days", "0"),
        ("--days", "-1"),
        ("--days", "nan"),
        ("--start", "yesterday"),
        ("--start", "2026-13-01"),
        ("--workers", "0"),
    ],
)
def test_screen_refuses_bad_options(tmp_path: Path, option: str, value: str) -> None:
    args = screen_fixture(tmp_path)
    args[args.index(option) + 1] = value
    result = keepout(*args)
    assert result.exit_code == ExitCode.USAGE
    assert option in result.stderr


def test_screen_needs_an_existing_fleet_file(tmp_path: Path) -> None:
    args = screen_fixture(tmp_path)
    args[args.index("--fleet") + 1] = str(tmp_path / "missing.toml")
    result = keepout(*args)
    assert result.exit_code == ExitCode.USAGE
    assert "--fleet" in result.stderr


def test_screen_takes_a_catalog_file_or_a_source_not_both(tmp_path: Path) -> None:
    result = keepout(*screen_fixture(tmp_path, "--source", "celestrak"))
    assert result.exit_code == ExitCode.USAGE
    assert "give either --catalog or --source, not both" in result.stderr


def test_parse_time_reads_one_iso_utc_time() -> None:
    assert screen.parse_time("2026-10-06T12:00:00").isot == "2026-10-06T12:00:00.000"
    assert screen.parse_time("2026-10-06").scale == "utc"
    with pytest.raises(ValueError, match="not an ISO 8601 UTC time: 'soon'"):
        screen.parse_time("soon")
