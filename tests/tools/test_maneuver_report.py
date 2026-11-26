"""The manoeuvre report, from a builder CDM and the screening fixture (offline).

The CDM is a fast LEO crossing created 8 h before TCA, so its grid has a few
burn times; the demonstration runs the fixture's GEO event (Intelsat 39
drifting past Luch-5X), whose primary the fleet below makes manoeuvrable.
"""

import copy
import dataclasses
import shutil
from pathlib import Path
from typing import Final

import astropy.units as u
import maneuver_heatmap as heatmap
import maneuver_report as report
import pytest
from builders import CROSSING_TCA, crossing_conjunction
from cli_builders import write_fleet
from generated import json_text, read_json, readme_with, relative, save_svg
from matplotlib.collections import QuadMesh
from matplotlib.colors import to_hex
from report_builders import fixture_data, readme_with_markers

from keepout.cdm import Maneuverable, ObjectMetadata, kvn, make_cdm
from keepout.figures import LOWEST_LOG_PC, SEQUENTIAL, grid_figure
from keepout.sources import cara
from keepout.sources.cara import CARA_COMMIT
from keepout.units import HOUR

TCA: Final = CROSSING_TCA
NAME: Final = "builder_event"
FLEET: Final = """\
catalog_hard_body_radius = 1.0

[[satellites]]
catalog_number = 44476
hard_body_radius = 20.0
maneuverable = true
"""


def builder_cdm() -> str:
    """A fast LEO crossing at 630 km, 150 m apart radially, created 8 h before
    TCA: Pc about 5e-3."""
    cdm = make_cdm(
        crossing_conjunction(),
        ObjectMetadata("PRIMARY SAT", "2020-001A", maneuverable=Maneuverable.YES),
        ObjectMetadata("DEBRIS", "2019-001A"),
        created=TCA - 8 * HOUR * u.s,
    )
    return kvn.dumps(cdm)


def arguments(tmp_path: Path) -> list[str]:
    """The builder CDM, the fixture as the catalog, the fleet, the page, the
    images and a README with the markers."""
    fixture_data(tmp_path)
    cdm = tmp_path / f"{NAME}.kvn"
    cdm.write_text(builder_cdm(), encoding="utf-8")
    readme_with_markers(tmp_path, report.START, report.END)
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


def copied(generated: Path, folder: Path) -> list[str]:
    """The arguments of a run on a copy of ``generated`` in ``folder``.

    The fleet stays ``generated``'s: the summary records its path.
    """
    shutil.copytree(generated, folder, dirs_exist_ok=True)
    args = arguments_of(folder)
    args[args.index("--fleet") + 1] = str(generated / "fleet.toml")
    return args


def test_check_passes_on_the_written_report(generated: Path) -> None:
    assert report.main(["--check", *arguments_of(generated)]) == 0


def test_check_fails_on_a_stale_report(
    generated: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = copied(generated, tmp_path)
    json_file = tmp_path / "docs" / "maneuver-report.json"
    summary = written(tmp_path)
    summary["event"]["baseline"]["pc"] = 0.5
    json_file.write_text(json_text(summary), encoding="utf-8")
    assert report.main(["--check", *args]) == 1
    assert "maneuver-report.json is out of date" in capsys.readouterr().out


def test_check_needs_the_heatmap(
    generated: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = copied(generated, tmp_path)
    (tmp_path / "docs" / "img" / "maneuver" / report.HEATMAP).unlink()
    assert report.main(["--check", *args]) == 1
    assert "heatmap.svg is missing" in capsys.readouterr().out


def with_timings(folder: Path, timings: report.Timings) -> None:
    """The report in ``folder`` rewritten with other timings, consistently."""
    summary = written(folder)
    summary["timings"] = timings
    docs = folder / "docs"
    images = docs / "img" / "maneuver"
    (docs / "maneuver-report.json").write_text(json_text(summary), encoding="utf-8")
    (docs / "maneuver-report.md").write_text(
        report.render_page(summary, relative(images, docs)), encoding="utf-8"
    )
    readme = folder / "README.md"
    block = report.render_summary(summary, relative(images, folder))
    readme.write_text(
        readme_with(readme, report.START, report.END, block), encoding="utf-8"
    )


OTHER_TIMINGS: report.Timings = {
    "cpu_model": "another CPU",
    "logical_cpus": 64,
    "python": "3.99.0",
    "date": "2000-01-01",
    "one_thread": False,
    "runs": 7,
    "wall": [0.1, 0.2],
    "cpu": [0.3, 0.4],
}


@pytest.mark.slow
def test_check_takes_the_committed_timings(generated: Path, tmp_path: Path) -> None:
    # The times depend on the machine: --check shows the committed ones and
    # checks the rest.
    args = copied(generated, tmp_path)
    with_timings(tmp_path, OTHER_TIMINGS)
    assert report.main(["--check", *args]) == 0
    assert "over 7 runs on 2000-01-01, another CPU (64 logical CPUs)" in page(tmp_path)


def test_keep_timings_writes_without_measuring_again(
    generated: Path, tmp_path: Path
) -> None:
    args = copied(generated, tmp_path)
    with_timings(tmp_path, OTHER_TIMINGS)
    (tmp_path / "docs" / "maneuver-report.md").write_text("stale", encoding="utf-8")
    assert report.main(["--keep-timings", *args]) == 0
    assert written(tmp_path)["timings"] == OTHER_TIMINGS
    assert "another CPU (64 logical CPUs)" in page(tmp_path)


def test_the_heatmap_is_a_reproducible_svg(generated: Path, tmp_path: Path) -> None:
    event, _ = report.load_event(generated / f"{NAME}.kvn")
    plan, _, _ = report.timed_plan(event, 10.0, 1)
    title = report.heatmap_title(written(generated)["event"])
    one, two = tmp_path / "one.svg", tmp_path / "two.svg"
    for path in (one, two):
        drawing = heatmap.heatmap_figure(grid_figure(plan), title)
        save_svg(drawing, path, facecolor=heatmap.FACECOLOR, dpi=heatmap.DPI)
    assert one.read_bytes() == two.read_bytes()
    written_svg = generated / "docs" / "img" / "maneuver" / report.HEATMAP
    assert one.read_bytes() == written_svg.read_bytes()
    text = one.read_text(encoding="utf-8")
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
    # As keepout.maneuver.judge has it: one clean option is enough.
    demo["recommendable"] = any(item["clean"] for item in demo["rescreens"])
    return report.render_page(summary, "img")


def test_a_burn_bringing_the_pair_back_is_not_clean(generated: Path) -> None:
    summary = written(generated)
    text = with_returns(summary, {"prograde"})
    demo = summary["demonstration"]
    assert demo is not None
    plan = demo["plan"]
    assert plan is not None
    pair = f"{plan['primary_name']} and {plan['secondary_name']}"
    assert (
        f"  - **back above the target**: the burn brings {pair} "
        "back together at 2026-10-09 00:07 UTC, range 4113 m, 2.0 m/s, assumed "
        "Pc 9.87e-06, yellow: above the target 1.00e-07, so this option is not "
        "clean"
    ) in text
    assert "- **prograde** (not clean): " in text
    assert "- **retrograde** (clean): " in text
    assert (
        "Only the retrograde burn is clean; `keepout maneuver` still lists the "
        "prograde one, marked not clean. The prograde burn clears the event"
    ) in text


def test_every_burn_bringing_the_pair_back_exits_four_on_the_page(
    generated: Path,
) -> None:
    text = with_returns(written(generated), {"prograde", "retrograde"})
    assert text.count("**back above the target**") == 2
    assert "The prograde and the retrograde burns clear the event" in text
    assert "No burn is clean, so `keepout maneuver` exits with status 4" in text


def test_no_return_no_outcome_paragraph(generated: Path) -> None:
    text = with_returns(written(generated), set())
    assert "**back above the target**" not in text
    assert "only the re-screen finds such a return" not in text


def test_the_page_states_the_assumptions_and_the_sources(generated: Path) -> None:
    text = page(generated)
    assert text.startswith("# Manoeuvre planning on a CARA test event\n")
    assert "Combined hard-body radius: HBR 10 m, given with `--hbr`." in text
    assert "with HBR 10 m; gravity" in text
    assert "(keepout's default: ten times below the red threshold, 1e-04)" in text
    assert f"CARA test CDM `{NAME}`" in text
    assert f"commit `{CARA_COMMIT}`" in text
    assert (
        "a 2026 test conjunction of PRIMARY SAT (1001) with DEBRIS (2002) at "
        "11815 m/s, red without a burn."
    ) in text
    assert "and no drag (the CDM gives the primary no CD·A/m)." in text
    alt = "Pc after an along-track burn of PRIMARY SAT (1001)"
    assert f"![{alt}](img/maneuver/heatmap.svg)" in (text)
    assert '!!! note "A demonstration of the mechanism, not a recommendation"' in text
    assert "| none | - | - |" in text
    assert "--check checks the numbers, not the times. -->" in text
    assert text.index("## Timing") > text.index("## Data")


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


def test_plot_uses_figure_colours(generated: Path) -> None:
    event, _ = report.load_event(generated / f"{NAME}.kvn")
    plan, _, _ = report.timed_plan(event, 10.0, 1)
    drawing = heatmap.heatmap_figure(grid_figure(plan), "a heatmap")
    meshes = [
        mesh
        for axes in drawing.axes[:2]
        for mesh in axes.collections
        if isinstance(mesh, QuadMesh)
    ]
    assert len(meshes) == 2
    for mesh in meshes:
        cmap = mesh.get_cmap()
        assert to_hex(cmap(0.0)) == SEQUENTIAL[0]
        assert to_hex(cmap(1.0)) == SEQUENTIAL[-1]
        assert mesh.norm.vmin == LOWEST_LOG_PC


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


def test_caras_radius_needs_the_cdms_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [{"Conjunction_ID": "another", "HBR_m": 99}]
    monkeypatch.setattr(cara, "load_xlsx", lambda path: rows)
    with pytest.raises(ValueError, match="has no row builder_event"):
        report.cara_radius(tmp_path / "table.xlsx", NAME)


@pytest.mark.parametrize("radius", ["0", "-1", "nan", "inf"])
def test_a_radius_must_be_positive(
    radius: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        report.main(["--hbr", radius])
    assert raised.value.code == 2
    message = (
        f"argument --hbr: hard-body radius must be finite and > 0, got {float(radius)}"
    )
    assert message in capsys.readouterr().err


def test_delta_v_keeps_three_figures_both_ways() -> None:
    assert heatmap.delta_v_text(-0.0019953) == "-0.00200"
    assert heatmap.delta_v_text(0.003981) == "+0.00398"
    assert heatmap.delta_v_text(0.0794) == "+0.0794"


def test_a_note_repeated_for_the_options_is_one_caveat(generated: Path) -> None:
    plan = copy.deepcopy(written(generated)["event"])
    prograde, retrograde = plan["prograde"], plan["retrograde"]
    assert prograde is not None
    assert retrograde is not None
    plan["notes"] = [
        "2-D Pc assumptions questionable: lasts 1 s; a 3-D Pc may differ",
        "drag is off: no CD_AREA_OVER_MASS",
    ]
    prograde["notes"] = [
        "2-D Pc assumptions questionable: lasts 2 s; a 3-D Pc may differ"
    ]
    retrograde["notes"] = ["covariance repaired"]
    assert report._caveats(plan) == [
        "",
        "Caveats:",
        "",
        "- 2-D Pc assumptions questionable (a 3-D Pc may differ):",
        "  - without a burn: lasts 1 s.",
        "  - prograde: lasts 2 s.",
        "- drag is off: no CD_AREA_OVER_MASS.",
        "- retrograde: covariance repaired.",
    ]


def test_an_options_notes_are_written_with_the_option_only(generated: Path) -> None:
    event, _ = report.load_event(generated / f"{NAME}.kvn")
    plan, _, _ = report.timed_plan(event, 10.0, 1)
    assert plan.prograde is not None
    option = dataclasses.replace(plan.prograde, notes=("covariance repaired",))
    notes = ("drag is off", "prograde: covariance repaired")
    summary = report.plan_summary(
        dataclasses.replace(plan, prograde=option, notes=notes)
    )
    assert summary["notes"] == ["drag is off"]
    assert summary["prograde"] is not None
    assert summary["prograde"]["notes"] == ["covariance repaired"]
