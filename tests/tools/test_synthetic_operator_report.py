import copy
import dataclasses
import shutil
import textwrap
from pathlib import Path

import pytest
import screening_snapshot as snapshot
import synthetic_operator_report as report
from astropy.time import Time
from cli_builders import write_fleet
from generated import read_json
from report_builders import assert_writes_then_checks, fixture_data
from screening_data import FIXTURE_START, load_fixture

from keepout.cdm import kvn
from keepout.synthetic import Fleet, FleetSatellite, run
from keepout.units import DAY


def arguments(tmp_path: Path) -> list[str]:
    """The fixture as the catalog, the manifest, a fleet file and the page."""
    fixture_data(tmp_path)
    write_fleet(tmp_path)
    return arguments_of(tmp_path)


def arguments_of(folder: Path) -> list[str]:
    """The arguments of a run on the files of ``folder``."""
    return [
        "--data",
        str(folder / "data"),
        "--fleet",
        str(folder / "fleet.toml"),
        "--page",
        str(folder / "docs" / "synthetic-operator.md"),
        "--workers",
        "1",
    ]


def written(tmp_path: Path) -> report.Summary:
    return read_json(tmp_path / "docs" / "synthetic-operator.json", report.Summary)


def page(tmp_path: Path) -> str:
    return (tmp_path / "docs" / "synthetic-operator.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory where the report was written once from the fixture."""
    tmp_path = tmp_path_factory.mktemp("report")
    assert report.main(arguments(tmp_path)) == 0
    return tmp_path


def test_main_writes_then_checks(tmp_path: Path) -> None:
    assert_writes_then_checks(report.main, arguments(tmp_path))


def test_the_summary_has_the_settings_catalog_fleet_and_actions(
    generated: Path,
) -> None:
    summary = written(generated)
    settings = summary["settings"]
    assert settings["fleet"] == (generated / "fleet.toml").as_posix()
    assert settings["start"] == "2026-10-06T00:00:00.000"
    assert settings["days"] == 7.0
    assert settings["volume"] == "5 km sphere (as in CelesTrak's SOCRATES)"
    assert (settings["green_below"], settings["red_from"]) == (1e-7, 1e-4)
    assert settings["decision_point_days"] == 2.0
    assert summary["catalog"] == {
        "retrieved": "2026-10-06",
        "element_sets": 222,
        "screened": 217,
        "skipped": 5,
    }
    assert [(s["name"], s["regime"], s["approaches"]) for s in summary["fleet"]] == [
        ("FLEET ONE", "LEO", 2),
        ("INTELSAT 39 (IS-39)", "GEO", 1),
    ]
    assert summary["actions"] == {
        "ignore": 2,
        "monitor": 1,
        "prepare maneuver": 0,
        "maneuver": 0,
    }
    assert summary["example"] == "52939_47826_20261006T193642.717Z_20261006T000000.000Z"


def test_each_event_has_its_pc_the_bound_and_the_decision(generated: Path) -> None:
    events = written(generated)["events"]
    assert [(e["primary"], e["secondary"]) for e in events] == [
        (52939, 47826),
        (44476, 55841),
        (52939, 57087),
    ]
    assert all(e["method"] == "foster" for e in events)
    geo = events[1]
    assert geo["secondary_name"] == "LUCH-5X (OLYMP-K 2)"
    assert (geo["secondary_regime"], geo["level"], geo["action"]) == (
        "GEO",
        "yellow",
        "monitor",
    )
    # The report rounds to 3 significant figures.
    assert geo["pc"] == pytest.approx(7.38e-6, abs=5e-9)
    assert geo["max_pc"] == pytest.approx(7.73e-6, abs=5e-9)
    assert geo["diluted"] is True
    assert len(geo["notes"]) == 2
    assert [e["diluted"] for e in events] == [False, True, False]
    for event in events:
        assert event["pc"] is not None
        assert event["max_pc"] is not None
        assert event["max_pc"] >= event["pc"]
        primary, secondary = event["transverse_sigmas"]
        assert 1e3 < primary < 25e3
        assert 1e3 < secondary < 25e3


def test_the_page_states_the_assumption_and_shows_the_closest_cdm(
    generated: Path,
) -> None:
    text = page(generated)
    assert text.startswith("# Synthetic operator on public data\n")
    assert '!!! warning "Assumed covariances"' in text
    assert "| FLEET ONE (52939) | LEO | 5 m | yes | 2026-10-05 12:01:08 | 2 |" in text
    assert "| monitor | 1 |" in text
    assert "1 of 3 Pc value is in the dilution region" in text
    assert "For the other 2 the maximum needs a larger covariance" in text
    assert "- **44476 - 55841** (2026-10-08 12:03:02.772): 2-D Pc" in text
    assert (
        "The CDM of the closest approach (FLEET ONE (52939) and "
        "STARLINK-2411 (47826), 1029 m)"
    ) in text
    assert (
        "MESSAGE_ID                     = "
        "52939_47826_20261006T193642.717Z_20261006T000000.000Z\n"
    ) in text
    assert "COVARIANCE_METHOD              = DEFAULT" in text
    assert "CT_R                           = 0.0 [m**2]" in text
    assert '??? example "The CDM in KVN"' in text
    assert text.endswith("    ```\n")


def test_check_fails_when_a_file_is_out_of_date(
    generated: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shutil.copytree(generated, tmp_path, dirs_exist_ok=True)
    args = arguments_of(tmp_path)
    # The fleet stays generated's: the summary records its path.
    args[args.index("--fleet") + 1] = str(generated / "fleet.toml")
    json_file = tmp_path / "docs" / "synthetic-operator.json"
    json_file.write_text("{}\n", encoding="utf-8")
    assert report.main(["--check", *args]) == 1
    assert "synthetic-operator.json is out of date" in capsys.readouterr().out


def test_the_window_is_the_week_of_the_socrates_report() -> None:
    manifest = snapshot.read_manifest(snapshot.DATA)
    start = report.window_start(manifest)
    assert start == Time("2026-10-06T00:00:00", scale="utc")
    assert start.scale == "utc"


def test_paths_are_shown_from_the_repository() -> None:
    assert report.shown(report.FLEET) == "examples/synthetic-operator/fleet.toml"
    assert report.shown(Path("/elsewhere/fleet.toml")) == "/elsewhere/fleet.toml"


def test_rounded_shortens_real_values_only() -> None:
    text = "\n".join(
        [
            "CCSDS_CDM_VERS                 = 2.0",
            "COMMENT Sigmas: R 0.204 km, T 19.579123456789 km",
            "TCA                            = 2026-10-12T11:28:28.709977",
            "MESSAGE_ID                     = 38771_35196_20261012T112828.710Z",
            "OBJECT_DESIGNATOR              = 38771",
            "MISS_DISTANCE                  = 212.3720068512821 [m]",
            "Z_DOT                          = -0.35235483308181775 [km/s]",
            "CR_R                           = 41616.00000000085 [m**2]",
            "CT_R                           = 6.058122821e-09 [m**2]",
            "CN_T                           = -1.8e-07 [m**2]",
            "CT_T                           = 1.2345678901234e+20 [m**2]",
            "CD_AREA_OVER_MASS              = 0.012345678901234",
            "",
        ]
    )
    assert report.rounded(text).split("\n") == [
        "CCSDS_CDM_VERS                 = 2.0",
        "COMMENT Sigmas: R 0.204 km, T 19.579123456789 km",
        "TCA                            = 2026-10-12T11:28:28.709977",
        "MESSAGE_ID                     = 38771_35196_20261012T112828.710Z",
        "OBJECT_DESIGNATOR              = 38771",
        "MISS_DISTANCE                  = 212.3720069 [m]",
        "Z_DOT                          = -0.3523548331 [km/s]",
        "CR_R                           = 41616.0 [m**2]",
        "CT_R                           = 0.0 [m**2]",
        "CN_T                           = 0.0 [m**2]",
        "CT_T                           = 1.23456789e+20 [m**2]",
        "CD_AREA_OVER_MASS              = 0.0123456789",
        "",
    ]
    assert report.rounded("X = 1.23456 [km]", digits=3, floor=0.0) == "X = 1.23 [km]"


def test_a_rounded_cdm_still_reads(generated: Path) -> None:
    text = page(generated)
    block = text.split("```text\n", 1)[1].split("```", 1)[0]
    cdm = textwrap.dedent(block)
    message = kvn.loads(cdm)
    assert message.message_id == "52939_47826_20261006T193642.717Z_20261006T000000.000Z"
    assert message.miss_distance == pytest.approx(1029.0, abs=1.0)


def test_a_run_without_approaches_shows_none(generated: Path) -> None:
    summary = copy.deepcopy(written(generated))
    summary["events"] = []
    summary["example"] = None
    text = report.render_page(summary, None)
    assert "## Approaches\n\nNone.\n" in text
    assert text.endswith("## Example CDM\n\nNone.\n")
    assert "sigma_T is" not in text


def test_an_event_without_a_pc_shows_its_error(generated: Path) -> None:
    summary = copy.deepcopy(written(generated))
    event = summary["events"][0]
    event.update(
        {
            "pc": None,
            "method": None,
            "max_pc": None,
            "scale_factor": None,
            "diluted": None,
            "level": None,
            "notes": ["covariance is not positive semi-definite"],
        }
    )
    text = report.render_page(summary, None)
    assert "| - | - | - | ignore |" in text
    assert "0 of 3 Pc values are in the dilution region" not in text
    assert "1 of 3 Pc value is in the dilution region" in text
    assert "For the other one the maximum" in text
    assert "(2026-10-06 19:36:42.717): covariance is not positive" in text


def test_an_observation_without_a_pc_is_summarised_with_its_error() -> None:
    fixture = load_fixture()
    fleet = Fleet((FleetSatellite(44476, 20.0, maneuverable=False),), 1.0)
    result = run(fleet, fixture, FIXTURE_START, 3 * DAY)
    (event,) = result.events
    observation = dataclasses.replace(
        event.observation, pc=None, result=None, error="no covariance", diagnostics=()
    )
    failed = dataclasses.replace(event, observation=observation)
    failing = dataclasses.replace(result, events=(failed,))
    summary = report.summarise(failing, fixture, "day", "fleet.toml", failed)
    (summarised,) = summary["events"]
    assert summarised["pc"] is None
    assert summarised["max_pc"] is None
    assert summarised["diluted"] is None
    assert summarised["notes"] == ["no covariance"]
    assert summarised["level"] == "yellow"  # the assessment is left as it was
