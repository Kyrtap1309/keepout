"""The manoeuvre report, from a builder CDM and the screening fixture (offline).

The CDM is a fast LEO crossing created 8 h before TCA, so its grid has a few
burn times; the demonstration runs the fixture's GEO event (Intelsat 39
drifting past Luch-5X), whose primary the fleet below makes manoeuvrable.
"""

import copy
import gzip
import math
import shutil
from pathlib import Path
from typing import Final

import astropy.units as u
import maneuver_report as report
import numpy as np
import pytest
import screening_snapshot as snapshot
import snapshot_files
from astropy.time import Time
from cli_builders import write_fleet
from generated import json_text, read_json
from screening_data import load_fixture

from keepout.cdm import Maneuverable, ObjectMetadata, kvn, make_cdm
from keepout.conjunction import Conjunction
from keepout.covariance import Covariance
from keepout.objects import ObjectState
from keepout.omm.celestrak import dumps_csv
from keepout.orbits import EARTH_MU
from keepout.propagation import EARTH_RADIUS
from keepout.sources import cara
from keepout.sources.cara import CARA_COMMIT
from keepout.state import StateVector
from keepout.units import HOUR, KM

TCA: Final = Time("2026-10-06T00:00:00", scale="utc")
NAME: Final = "builder_event"
FLEET: Final = """\
catalog_hard_body_radius = 1.0

[[satellites]]
catalog_number = 44476
hard_body_radius = 20.0
maneuverable = true
"""


def _item(object_id: str, r: np.ndarray, v: np.ndarray) -> ObjectState:
    matrix = np.diag([100.0**2] * 3 + [0.01**2] * 3)
    return ObjectState(
        object_id, StateVector(TCA, r, v, "GCRF"), Covariance(matrix, "GCRF", TCA), 10.0
    )


def builder_cdm() -> str:
    """A fast LEO crossing at 630 km, 150 m apart radially, created 8 h before
    TCA: Pc about 5e-3."""
    radius = EARTH_RADIUS + 630 * KM
    speed = math.sqrt(EARTH_MU / radius)
    along = np.array([0.0, math.cos(0.9), math.sin(0.9)])
    across = np.array([0.0, math.cos(0.9), -math.sin(0.9)])
    conjunction = Conjunction(
        _item("1001", np.array([radius, 0.0, 0.0]), speed * along),
        _item("2002", np.array([radius + 150.0, 0.0, 0.0]), speed * across),
        20.0,
    )
    cdm = make_cdm(
        conjunction,
        ObjectMetadata("PRIMARY SAT", "2020-001A", maneuverable=Maneuverable.YES),
        ObjectMetadata("DEBRIS", "2019-001A"),
        created=TCA - 8 * HOUR * u.s,
    )
    return kvn.dumps(cdm)


def arguments(tmp_path: Path) -> list[str]:
    """The builder CDM, the fixture as the catalog, the fleet, the page, the
    images and a README with the markers."""
    data = tmp_path / "data"
    data.mkdir()
    csv = dumps_csv(load_fixture()).encode()
    (data / snapshot.SNAPSHOT).write_bytes(gzip.compress(csv))
    shutil.copy(snapshot.DATA / snapshot_files.MANIFEST, data / snapshot_files.MANIFEST)
    cdm = tmp_path / f"{NAME}.kvn"
    cdm.write_text(builder_cdm(), encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{report.START}\n{report.END}\n", encoding="utf-8")
    write_fleet(tmp_path, FLEET)
    return arguments_of(tmp_path)


def arguments_of(folder: Path) -> list[str]:
    """The arguments of a run on the files of ``folder``."""
    return [
        "--cdm",
        str(folder / f"{NAME}.kvn"),
        "--data",
        str(folder / "data"),
        "--fleet",
        str(folder / "fleet.toml"),
        "--page",
        str(folder / "docs" / "maneuver-report.md"),
        "--images",
        str(folder / "docs" / "img" / "maneuver"),
        "--readme",
        str(folder / "README.md"),
        "--workers",
        "1",
        "--hbr",
        "10",
        "--repeat",
        "1",
    ]


def written(folder: Path) -> report.Summary:
    return read_json(folder / "docs" / "maneuver-report.json", report.Summary)


def page(folder: Path) -> str:
    return (folder / "docs" / "maneuver-report.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory where the report was written once."""
    folder = tmp_path_factory.mktemp("report")
    assert report.main(arguments(folder)) == 0
    return folder


def test_main_writes_then_checks(tmp_path: Path) -> None:
    args = arguments(tmp_path)
    assert report.main(["--check", *args]) == 1
    assert report.main(args) == 0
    assert report.main(["--check", *args]) == 0


def test_check_fails_on_a_stale_report(
    generated: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shutil.copytree(generated, tmp_path, dirs_exist_ok=True)
    args = arguments_of(tmp_path)
    json_file = tmp_path / "docs" / "maneuver-report.json"
    summary = written(tmp_path)
    summary["event"]["baseline"]["pc"] = 0.5
    json_file.write_text(json_text(summary), encoding="utf-8")
    assert report.main(["--check", *args]) == 1
    assert "maneuver-report.json is out of date" in capsys.readouterr().out


def test_check_needs_the_heatmap(
    generated: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shutil.copytree(generated, tmp_path, dirs_exist_ok=True)
    (tmp_path / "docs" / "img" / "maneuver" / report.HEATMAP).unlink()
    assert report.main(["--check", *arguments_of(tmp_path)]) == 1
    assert "heatmap.svg is missing" in capsys.readouterr().out


def test_the_heatmap_is_a_reproducible_svg(generated: Path) -> None:
    heatmap = generated / "docs" / "img" / "maneuver" / report.HEATMAP
    text = heatmap.read_text(encoding="utf-8")
    assert text.startswith("<?xml")
    assert "<dc:date>" not in text
    assert "retrograde" in text
    assert "prograde" in text


def test_the_summary_has_the_event_its_options_and_the_settings(
    generated: Path,
) -> None:
    summary = written(generated)
    settings = summary["settings"]
    assert settings["cdm"] == NAME
    assert settings["cara_commit"] == CARA_COMMIT
    assert settings["hard_body_radius"] == 10.0
    assert settings["hard_body_radius_source"] == "given"
    assert settings["target_pc"] == 1e-5
    assert settings["grid"]["delta_v_per_direction"] == 31
    event = summary["event"]
    assert (event["primary_name"], event["secondary_name"]) == (
        "PRIMARY SAT",
        "DEBRIS",
    )
    assert event["hours_to_tca"] == 8.0
    assert event["cells"] == event["burn_times"] * 62
    assert event["baseline"]["pc"] > 1e-3
    assert event["baseline"]["range_at_tca"] == 150.0
    for name in ("prograde", "retrograde"):
        option = event[name]
        assert option is not None
        assert option["meets_target"]
        assert option["pc"] < 1e-5
        assert 6.0 <= option["burn_lead_hours"] <= 8.0
        assert event["below_target"][name] > 0
    timings = summary["timings"]
    assert timings is not None
    assert timings["runs"] == 1
    assert timings["wall"][0] <= timings["wall"][1]


def test_the_demonstration_replans_the_yellow_event_and_rescreens(
    generated: Path,
) -> None:
    demo = written(generated)["demonstration"]
    assert demo is not None
    assert (demo["level"], demo["action"]) == ("yellow", "monitor")
    plan = demo["plan"]
    assert plan is not None
    assert (plan["primary"], plan["secondary"]) == ("44476", "55841")
    assert plan["target_pc"] == 1e-7
    assert plan["prograde"] is not None
    assert plan["retrograde"] is not None
    assert [item["direction"] for item in demo["rescreens"]] == [
        "prograde",
        "retrograde",
    ]
    for item in demo["rescreens"]:
        assert item["secondary_screened"]
        assert item["catalog_objects"] > 100
        assert item["clean"] == (not item["conflict"] and not item["event_returns"])
        for back in item["event_returns"]:
            assert str(back["secondary"]) == "55841"
            assert back["pc"] >= plan["target_pc"]


def with_returns(summary: report.Summary, directions: set[str]) -> str:
    """The page with the pair back above the target after these burns only."""
    summary = copy.deepcopy(summary)
    demo = summary["demonstration"]
    assert demo is not None
    for item in demo["rescreens"]:
        back: report.Approach = {
            "secondary": 55841,
            "tca": "2026-10-09T00:07:30.000",
            "range_at_tca": 4113.0,
            "relative_speed": 2.0,
            "pc": 9.87e-6,
            "level": "yellow",
        }
        returning = item["direction"] in directions
        item["event_returns"] = [back] if returning else []
        item["conflict"] = False
        item["clean"] = not returning
    return report.render_page(summary, "img")


def test_a_burn_bringing_the_pair_back_is_not_clean(generated: Path) -> None:
    text = with_returns(written(generated), {"prograde"})
    assert (
        "  - **back above the target**: the burn brings this pair back at "
        "2026-10-09 00:07 UTC at Pc 9.87e-06 (assumed), above the target "
        "1.00e-07, so this option is not clean"
    ) in text
    assert "The prograde burn clears the event at its TCA but brings" in text
    assert "The retrograde option is clean, so `keepout maneuver` exits with " in text
    assert "status 0" in text


def test_every_burn_bringing_the_pair_back_exits_four_on_the_page(
    generated: Path,
) -> None:
    text = with_returns(written(generated), {"prograde", "retrograde"})
    assert text.count("**back above the target**") == 2
    assert "The prograde and the retrograde burns clear the event" in text
    assert "No option is clean, so `keepout maneuver` exits with status 4" in text


def test_no_return_no_outcome_paragraph(generated: Path) -> None:
    text = with_returns(written(generated), set())
    assert "**back above the target**" not in text
    assert "in an encounter this slow" not in text


def test_the_page_states_the_assumptions_and_the_sources(generated: Path) -> None:
    text = page(generated)
    assert text.startswith("# Manoeuvre planning on a CARA test event\n")
    assert "Combined hard-body radius: HBR 10 m, given with `--hbr`." in text
    assert "with HBR 10 m, given with `--hbr`; gravity" in text
    assert "(keepout's default, a decade below red from 1e-04, as a margin)" in text
    assert f"CARA test CDM `{NAME}`" in text
    assert f"commit `{CARA_COMMIT}`" in text
    assert "PRIMARY SAT (1001) passing DEBRIS (2002), 2026 test data, red" in text
    assert "and no drag (the CDM gives the primary no CD·A/m)." in text
    alt = "Pc after an along-track burn of PRIMARY SAT (1001)"
    assert f"![{alt}](img/maneuver/heatmap.svg)" in (text)
    assert '!!! note "A demonstration of the mechanism, not a recommendation"' in text
    assert "| none | - | - |" in text
    assert "`--check` checks the numbers, not the times." in text


def test_the_readme_block_has_the_heatmap_and_the_options(generated: Path) -> None:
    readme = (generated / "README.md").read_text(encoding="utf-8")
    assert "(docs/img/maneuver/heatmap.svg)" in readme
    assert "| retrograde |" in readme
    assert "The burns of PRIMARY SAT (1001) that clear a red event" in readme
    assert "(2026 test data; HBR 10 m, given with `--hbr`;" in readme


def test_a_page_without_a_demonstration_or_timings(generated: Path) -> None:
    summary = copy.deepcopy(written(generated))
    summary["demonstration"] = None
    summary["timings"] = None
    text = report.render_page(summary, "img")
    assert "Not measured yet." in text
    assert "Not run." in text


def test_a_plan_without_options_says_so(generated: Path) -> None:
    summary = copy.deepcopy(written(generated))
    summary["event"]["prograde"] = None
    text = report.render_page(summary, "img")
    assert "| prograde | - | - | none on the grid | - | - |" in text


def test_edges_lie_midway_and_beyond_the_ends() -> None:
    edges = report._edges(np.array([1.0, 2.0, 4.0]), geometric=False)
    assert edges.tolist() == [0.5, 1.5, 3.0, 5.0]
    geometric = report._edges(np.array([1e-2, 1e-1, 1.0]), geometric=True)
    assert np.allclose(np.log10(geometric), [-2.5, -1.5, -0.5, 0.5])
    assert report._edges(np.array([7.0]), geometric=False).tolist() == [6.5, 7.5]


def test_the_radius_names_its_source(generated: Path) -> None:
    settings = copy.deepcopy(written(generated)["settings"])
    settings["hard_body_radius_source"] = "cara"
    assert report.radius_text(settings) == (
        "HBR 10 m, CARA's value for this CDM (column `HBR_m` of "
        "`DataFiles/PcTestCaseCDMs/CARA_PcMethod_Test_Conjunctions.xlsx` at the "
        "same commit)"
    )


def test_timings_over_runs_are_a_range(generated: Path) -> None:
    summary = copy.deepcopy(written(generated))
    summary["timings"] = {
        "cpu_model": "CPU",
        "logical_cpus": 2,
        "python": "3.13.7",
        "date": "2026-11-03",
        "one_thread": True,
        "runs": 3,
        "wall": [9.9, 10.1],
        "cpu": [9.3, 9.5],
    }
    text = report.render_page(summary, "img")
    assert (
        "took 9.9 to 10.1 s (CPU time 9.3 to 9.5 s, numerical libraries on one "
        "thread) over 3 runs on 2026-11-03, CPU (2 logical CPUs)"
    ) in text


def test_without_hbr_the_radius_is_caras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = tmp_path / "table.xlsx"
    rows = [
        {"Conjunction_ID": "another", "HBR_m": 99},
        {"Conjunction_ID": NAME, "HBR_m": 12},
    ]
    fetched: list[object] = []

    def fetch(file: cara.CaraFile, **_: object) -> Path:
        fetched.append(file)
        return table

    def load_xlsx(path: Path) -> list[dict[str, object]]:
        assert path == table
        return rows

    monkeypatch.setattr(cara, "fetch", fetch)
    monkeypatch.setattr(cara, "load_xlsx", load_xlsx)
    args = arguments(tmp_path)
    hbr = args.index("--hbr")
    del args[hbr : hbr + 2]
    assert report.main(args) == 0
    assert fetched == [cara.PC_METHOD_TABLE]
    settings = written(tmp_path)["settings"]
    assert settings["hard_body_radius"] == 12.0
    assert settings["hard_body_radius_source"] == "cara"
    assert "Combined hard-body radius: HBR 12 m, CARA's value" in page(tmp_path)
    rows.pop()
    with pytest.raises(ValueError, match="has no row builder_event"):
        report.cara_radius(table, NAME)


@pytest.mark.parametrize("radius", ["0", "-1", "nan", "inf"])
def test_a_radius_must_be_positive(radius: str) -> None:
    with pytest.raises(SystemExit):
        report.main(["--hbr", radius])


def test_delta_v_keeps_three_figures_both_ways() -> None:
    assert report.delta_v_text(-0.0019953) == "-0.00200"
    assert report.delta_v_text(0.003981) == "+0.00398"
    assert report.delta_v_text(0.0794) == "+0.0794"
