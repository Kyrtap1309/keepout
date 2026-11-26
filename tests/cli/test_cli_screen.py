import hashlib
import json
import re
import types
from pathlib import Path
from typing import Any

import pytest
from astropy.time import Time
from builders import approx, cached_pc
from cli_builders import (
    DAYS,
    FIXTURE_CSV,
    FLEET_TOML,
    START,
    keepout,
    operator_run,
    table_event_ids,
    usage_error,
    write_fleet,
)
from faults import recorded_runs
from inline_snapshot import snapshot

from keepout.cdm import kvn
from keepout.cli import screen
from keepout.cli.output import ExitCode
from keepout.omm import load
from keepout.omm.celestrak import dumps_csv
from keepout.settings import load_settings
from keepout.store import open_store


def screen_fixture(
    folder: Path, *args: str, fleet: str | None = None, start: str = START
) -> list[str]:
    path = write_fleet(folder) if fleet is None else write_fleet(folder, fleet)
    return [
        "screen",
        "--fleet",
        str(path),
        "--catalog",
        str(FIXTURE_CSV),
        "--start",
        start,
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
    shown = result.stdout.replace(str(FIXTURE_CSV), "<catalog>")
    assert shown.replace(str(tmp_path), "<tmp>") == snapshot("""\
Screened 2 fleet satellites against 217 objects (<catalog>) from 2026-10-06T00:00:00 to 2026-10-09T00:00:00 UTC: 2 close approaches.
Close approaches
Event         TCA (UTC)         Objects        Miss (m)  CDMs        Pc  Level   Action \n\
────────────────────────────────────────────────────────────────────────────────────────
2a85ed50c4e6  2026-10-06 19:36  52939 / 47826      1029     1  4.37e-08  green   ignore \n\
3befe6f1305d  2026-10-08 12:03  44476 / 55841      3647     1  7.38e-06  yellow  monitor
5 element sets left out (5 stale; --json lists them).
3befe6f1305d: monitor (latest Pc 7.38e-06 is yellow, 2.50 d before TCA; trend undetermined)
  diagnostic: 2-D Pc assumptions questionable: the encounter lasts about 2.51e+03 s, 2.9% of the primary's orbital period (limit 1%); a 3-D Pc may differ
  diagnostic: the covariances of primary (44476) and secondary (55841) are known only in position (zero velocity block): the 3-D methods need the velocity uncertainty, so the 2-D Pc is used
Covariances are assumed (keepout's model of GP element set errors), not orbit determination.
2 CDMs written to <tmp>/cdms
""")
    expected = operator_run()
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
    assert first["event_id"] == "2a85ed50c4e6"
    assert first["object1"] == {"object_id": "52939", "name": "FLEET ONE"}
    assert first["object2"]["object_id"] == "47826"
    assert first["miss_distance_m"] == pytest.approx(1028.74, abs=0.01)
    assert first["relative_speed_m_s"] > 1000.0
    assert first["cdm_file"] is None
    assert first["level"] == "green"
    assert first["action"] == "ignore"
    assert first["trend"] == "undetermined"
    (observation,) = first["observations"]
    assert observation["message_id"] == first["cdms"][0]
    assert observation["method"] == "foster"
    assert observation["pc"] == first["pc"] == approx(4.367e-8, rel=1e-3)
    assert observation["error"] is None
    assert second["action"] == "monitor"
    assert second["level"] == "yellow"


def test_screen_json_says_the_covariances_are_assumed(tmp_path: Path) -> None:
    result = keepout(*screen_fixture(tmp_path, "--json"))
    assert result.exit_code == ExitCode.OK
    payload = json.loads(result.stdout)
    assert payload["assumed_covariance_events"] == len(payload["events"]) == 2
    assert all(event["assumed_covariance"] for event in payload["events"])


def test_screens_on_two_days_issue_successive_cdms_of_each_event(
    tmp_path: Path,
) -> None:
    # The same element sets screened from two starts: the same approaches,
    # issued at two creation times, so each event has two CDMs.
    files: list[str] = []
    for start in (START, "2026-10-06T12:00:00"):
        folder = tmp_path / start.replace(":", "")
        folder.mkdir()
        cdms = folder / "cdms"
        result = keepout(*screen_fixture(folder, "--cdm-dir", str(cdms), start=start))
        assert result.exit_code == ExitCode.OK
        files += sorted(str(path) for path in cdms.iterdir())
    assert len(files) == len({Path(file).name for file in files}) == 4
    assessed = keepout("assess", *files, "--json")
    assert assessed.exit_code == ExitCode.OK
    events = json.loads(assessed.stdout)["events"]
    assert [len(event["cdms"]) for event in events] == [2, 2]
    for event in events:
        older, newer = event["observations"]
        assert older["time_to_tca_s"] == pytest.approx(newer["time_to_tca_s"] + 43200.0)


def test_screen_starts_now_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = Time("2026-10-06T00:00:00.75", scale="utc").unix
    monkeypatch.setattr(screen, "time", types.SimpleNamespace(time=lambda: now))
    args = screen_fixture(tmp_path, "--json")
    start = args.index("--start")
    del args[start : start + 2]
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
    assert [e["object1"]["object_id"] for e in payload["events"]] == ["52939"]
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
    assert "missing key(s) catalog_hard_body_radius" in result.stderr
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
    ("option", "value", "reason"),
    [
        ("--days", "0", "must be a positive number, got 0.0"),
        ("--days", "-1", "must be a positive number, got -1.0"),
        ("--days", "nan", "must be a positive number, got nan"),
        ("--start", "yesterday", "not an ISO 8601 UTC time: 'yesterday'"),
        ("--start", "2026-13-01", "not an ISO 8601 UTC time: '2026-13-01'"),
        ("--workers", "0", "0 is not in the range x>=1."),
    ],
)
def test_screen_refuses_bad_options(
    tmp_path: Path, option: str, value: str, reason: str
) -> None:
    args = screen_fixture(tmp_path)
    args[args.index(option) + 1] = value
    result = keepout(*args)
    assert result.exit_code == ExitCode.USAGE
    assert f"Invalid value for '{option}': {reason}" in usage_error(result)


def test_screen_needs_an_existing_fleet_file(tmp_path: Path) -> None:
    args = screen_fixture(tmp_path)
    missing = tmp_path / "missing.toml"
    args[args.index("--fleet") + 1] = str(missing)
    # Wide enough for the path to stay on one line.
    result = keepout(*args, env={"COLUMNS": "500"})
    assert result.exit_code == ExitCode.USAGE
    assert (
        f"Invalid value for '--fleet': File '{missing}' does not exist."
    ) in usage_error(result)


def test_screen_takes_a_catalog_file_or_a_source_not_both(tmp_path: Path) -> None:
    result = keepout(*screen_fixture(tmp_path, "--source", "celestrak"))
    assert result.exit_code == ExitCode.USAGE
    assert "give either --catalog or --source, not both" in result.stderr


def test_parse_time_reads_one_iso_utc_time() -> None:
    assert screen.parse_time("2026-10-06T12:00:00").isot == "2026-10-06T12:00:00.000"
    assert screen.parse_time("2026-10-06").scale == "utc"
    with pytest.raises(ValueError, match="not an ISO 8601 UTC time: 'soon'"):
        screen.parse_time("soon")


def test_screen_of_a_quiet_window_reports_no_approach(tmp_path: Path) -> None:
    # The fleet satellite and the Starlink it meets 0.82 d after the start,
    # both current: half a day sees no approach and leaves nothing out.
    catalog = tmp_path / "pair.csv"
    pair = [omm for omm in load(FIXTURE_CSV) if omm.catalog_number in {52939, 47826}]
    catalog.write_text(dumps_csv(pair), encoding="utf-8")
    fleet = FLEET_TOML.split("\n\n[[satellites]]\ncatalog_number = 44476")[0]
    args = screen_fixture(tmp_path, fleet=fleet)
    args[args.index("--catalog") + 1] = str(catalog)
    args[args.index("--days") + 1] = "0.5"
    result = keepout(*args)
    assert result.exit_code == ExitCode.OK
    assert result.stdout.replace(str(catalog), "<catalog>") == snapshot("""\
Screened 1 fleet satellite against 2 objects (<catalog>) from 2026-10-06T00:00:00 to 2026-10-06T12:00:00 UTC: 0 close approaches.
Covariances are assumed (keepout's model of GP element set errors), not orbit determination.
""")


def test_screen_with_store_records_the_catalog_digest(tmp_path: Path) -> None:
    db = tmp_path / "keepout.db"
    result = keepout(*screen_fixture(tmp_path, "--store", str(db), "--json"))
    assert result.exit_code == ExitCode.OK
    payload = json.loads(result.stdout)
    digest = hashlib.sha256(FIXTURE_CSV.read_bytes()).hexdigest()
    with open_store(db, create=False) as store:
        events = store.events()
        assert [cdm.message_id for e in events for cdm in e.cdms] == [
            e["cdms"][0] for e in payload["events"]
        ]
        ids = [event.event_id for event in events]
        for event_id in ids:
            reference = store.catalog(event_id)
            assert reference is not None
            (file,) = reference.files
            assert file.sha256 == digest
            assert file.path.parent == store.catalog_dir
            assert file.path.read_bytes() == FIXTURE_CSV.read_bytes()
    assert payload["store"] == {
        "path": str(db),
        "added": [e["cdms"][0] for e in payload["events"]],
        "already_stored": [],
        "events": ids,
        "new_events": ids,
    }
    text = keepout(*screen_fixture(tmp_path, "--store", str(db)))
    assert f"0 CDMs added to {db} (2 already stored), 0 new events.\n" in text.stdout


def test_screen_without_store_gives_no_store(tmp_path: Path) -> None:
    result = keepout(*screen_fixture(tmp_path, "--json"))
    assert json.loads(result.stdout)["store"] is None


def test_screen_with_store_shows_the_store_s_event_ids(tmp_path: Path) -> None:
    db = tmp_path / "keepout.db"
    cdms = tmp_path / "cdms"
    payload = json.loads(
        keepout(
            *screen_fixture(
                tmp_path, "--store", str(db), "--cdm-dir", str(cdms), "--json"
            )
        ).stdout
    )
    with open_store(db, create=False) as store:
        stored = store.events()
    ids = [event.event_id for event in stored]
    assert all(re.fullmatch(r"[0-9a-f]{12}", event_id) for event_id in ids)
    assert [event["event_id"] for event in payload["events"]] == ids
    assert [event["cdms"] for event in payload["events"]] == [
        [cdm.message_id for cdm in event.cdms] for event in stored
    ]
    assert payload["store"]["events"] == ids
    text = keepout(*screen_fixture(tmp_path, "--store", str(db))).stdout
    assert table_event_ids(text) == ids
    assert f"\n{ids[1]}: monitor (latest Pc 7.38e-06" in text
    # The Pc is cached with the default settings: store add finds it.
    rebuilt = keepout("store", "rebuild", "--store", str(db))
    assert rebuilt.stdout == f"Computed 0 Pc observations in {db}.\n"
    added = keepout(
        "store", "add", *sorted(map(str, cdms.iterdir())), "--store", str(db)
    )
    assert added.exit_code == ExitCode.OK
    assert table_event_ids(added.stdout) == ids
    assert keepout("store", "rebuild", "--store", str(db)).stdout == rebuilt.stdout


def test_a_screen_row_stays_the_approach_screened_on_day_two(tmp_path: Path) -> None:
    # The event keeps its id; its row shows the day's CDM, the store has both.
    db = tmp_path / "keepout.db"
    pair = {"object1": "44476", "object2": "55841"}

    def screened(start: str) -> dict[str, Any]:
        result = keepout(
            *screen_fixture(tmp_path, "--store", str(db), "--json", start=start)
        )
        assert result.exit_code == ExitCode.OK
        rows: list[dict[str, Any]] = [
            event
            for event in json.loads(result.stdout)["events"]
            if {k: event[k]["object_id"] for k in pair} == pair
        ]
        (row,) = rows
        return row

    first = screened("2026-10-06")
    second = screened("2026-10-07")
    assert second["event_id"] == first["event_id"]
    assert len(first["cdms"]) == len(second["cdms"]) == 1
    assert second["cdms"] != first["cdms"]
    assert len(second["observations"]) == 1
    both = [*first["cdms"], *second["cdms"]]
    with open_store(db, create=False) as store:
        stored = store.event(str(first["event_id"]))
        assert sorted(cdm.message_id for cdm in stored.cdms) == sorted(both)
        texts = {cdm.message_id: kvn.dumps(cdm) for cdm in stored.cdms}
    files = []
    for message_id, text in texts.items():
        path = tmp_path / f"{message_id}.kvn"
        path.write_text(text, encoding="utf-8")
        files.append(str(path))
    for command in (("store", "add"), ("assess",)):
        result = keepout(*command, *files, "--store", str(db), "--json")
        (event,) = json.loads(result.stdout)["events"]
        assert event["event_id"] == first["event_id"]
        assert sorted(event["cdms"]) == sorted(both)
        assert len(event["observations"]) == 2


def test_screen_computes_and_stores_with_the_settings(tmp_path: Path) -> None:
    settings = tmp_path / "settings.toml"
    settings.write_text("[pc]\nhbr_m = 1000.0\n[risk]\nred_from = 0.9\n")
    db = tmp_path / "keepout.db"
    args = ("--settings", str(settings), "--store", str(db), "--json")
    result = keepout(*screen_fixture(tmp_path, *args))
    assert result.exit_code == ExitCode.OK
    first = json.loads(result.stdout)["events"][0]
    assert first["pc"] == pytest.approx(4.25e-2, rel=1e-2)
    assert first["level"] == "yellow"
    with open_store(db, create=False) as store:
        (stored, _) = store.events()
        (observation,) = cached_pc(store, stored.event_id, load_settings(settings))
    assert observation.pc == first["pc"]


def test_screen_with_store_computes_each_pc_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = recorded_runs(monkeypatch)
    db = tmp_path / "keepout.db"
    result = keepout(*screen_fixture(tmp_path, "--store", str(db), "--json"))
    assert result.exit_code == ExitCode.OK
    assert len(runs) == len(json.loads(result.stdout)["events"]) == 2
