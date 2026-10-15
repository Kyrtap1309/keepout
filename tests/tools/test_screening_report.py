import copy
import dataclasses
import gzip
import json
import math
import shutil
from collections.abc import Sequence
from pathlib import Path

import astropy.units as u
import pytest
import screening_report as report
import screening_snapshot as snapshot
from astropy.time import Time
from screening_data import socrates_text

from keepout.omm import Omm, load
from keepout.omm.celestrak import dumps_csv
from keepout.sources.socrates import COLUMNS, read_socrates

FIXTURE = {omm.catalog_number: omm for omm in load(snapshot.DATA / snapshot.FIXTURE)}
EXCERPT = socrates_text()
HEADER = snapshot.socrates_header(EXCERPT)

PAIRS = {
    (63216, 64145),  # 5 fast approaches
    (58509, 63856),  # 3 fast
    (57062, 63216),  # 2 fast
    (44476, 55841),  # 1 slow (2 m/s)
    (56155, 56156),  # 21 slow formation approaches; keepout finds one more
}
"""Pairs of the excerpt whose objects are in the test fixture."""

ROWS = [a for a in read_socrates(EXCERPT) if (a.primary, a.secondary) in PAIRS]
START = Time("2026-10-06T00:00:00", scale="utc")


def objects() -> list[Omm]:
    numbers = {n for pair in PAIRS for n in pair}
    return [FIXTURE[n] for n in sorted(numbers)]


def with_value(line: str, column: str, value: str) -> str:
    """A SOCRATES line with one value replaced."""
    values = line.rstrip("\r\n").split(",")
    values[COLUMNS.index(column)] = value
    return ",".join(values) + "\r\n"


def data_dir(
    tmp_path: Path,
    lines: Sequence[str] | None = None,
    catalog: Sequence[Omm] | None = None,
) -> Path:
    """A data directory: the chosen pairs' rows and objects, the manifest."""
    data = tmp_path / "data"
    data.mkdir()
    rows = [a.line for a in ROWS] if lines is None else lines
    (data / snapshot.SOCRATES).write_bytes(
        gzip.compress((HEADER + "".join(rows)).encode())
    )
    omms = objects() if catalog is None else catalog
    (data / snapshot.SNAPSHOT).write_bytes(gzip.compress(dumps_csv(omms).encode()))
    shutil.copy(snapshot.DATA / snapshot.MANIFEST, data / snapshot.MANIFEST)
    return data


def arguments(tmp_path: Path, data: Path) -> list[str]:
    readme = tmp_path / "README.md"
    if not readme.exists():
        readme.write_text(f"intro\n{report.START}\n{report.END}\n", encoding="utf-8")
    page = tmp_path / "docs" / "screening-socrates.md"
    return [
        "--data",
        str(data),
        "--page",
        str(page),
        "--readme",
        str(readme),
        "--workers",
        "1",
    ]


def written(tmp_path: Path) -> report.Summary:
    path = tmp_path / "docs" / "screening-socrates.json"
    summary: report.Summary = json.loads(path.read_text(encoding="utf-8"))
    return summary


def test_main_writes_then_checks(tmp_path: Path) -> None:
    args = arguments(tmp_path, data_dir(tmp_path))
    assert report.main(["--check", *args]) == 1
    assert report.main(args) == 0
    assert report.main(["--check", *args]) == 0
    summary = written(tmp_path)
    assert summary["settings"]["start"] == "2026-10-06T00:00:00.000"
    assert (summary["settings"]["primaries"], summary["settings"]["objects"]) == (5, 9)
    fast, slow = summary["groups"]["fast"], summary["groups"]["slow"]
    assert (fast["compared"], fast["agree"]) == (10, 10)
    assert (slow["compared"], slow["agree"]) == (22, 22)
    assert fast["worst"] is not None
    assert fast["worst"]["tca"] <= 1e-3
    assert fast["worst"]["miss_distance"] <= 0.501
    assert summary["differences"] == []
    assert summary["not_compared"] == {
        "outside_catalog": 40211,
        "other_element_sets": 144743,
        "skipped": [],
    }
    only = summary["keepout_only"]
    assert [(a["primary"], a["secondary"], a["tca"]) for a in only] == [
        (56155, 56156, "2026-10-06 04:18:26.173")
    ]
    assert only[0]["nearest_other"] == pytest.approx(571.0, abs=1.0)
    page = (tmp_path / "docs" / "screening-socrates.md").read_text(encoding="utf-8")
    assert "## Differences\n\nNone.\n" in page
    assert "| 56155 - 56156 | 2026-10-06 04:18:26.173 | 412.543 m |" in page
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "([report](docs/screening-socrates.md))" in readme
    assert "| fast (>= 1 km/s) | 10 | 10 |" in readme
    assert "keepout also reports 1 minimum of these pairs" in readme


def test_an_approach_beyond_the_tolerances_fails_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = [a.line for a in ROWS]
    changed = next(i for i, a in enumerate(ROWS) if a.primary == 58509)
    distance = f"{ROWS[changed].miss_distance / 1e3 + 0.002:.3f}"
    lines[changed] = with_value(lines[changed], "TCA_RANGE", distance)
    args = arguments(tmp_path, data_dir(tmp_path, lines))
    assert report.main(args) == 1
    (difference,) = written(tmp_path)["differences"]
    assert (difference["primary"], difference["secondary"]) == (58509, 63856)
    assert difference["reported"]
    assert difference["cause"] is None
    assert difference["keepout"] is not None
    assert abs(difference["keepout"]["tca_difference"]) <= 1e-3
    assert "beyond the tolerances" in capsys.readouterr().out
    page = (tmp_path / "docs" / "screening-socrates.md").read_text(encoding="utf-8")
    assert "| unexplained |" in page
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "1 approach differs without a known cause." in readme


def test_an_approach_far_from_keepouts_is_not_matched(tmp_path: Path) -> None:
    # SOCRATES's TCA moved by two minutes: keepout's approach is no longer
    # within the match window and is reported as keepout's only.
    lines = [a.line for a in ROWS]
    changed = next(i for i, a in enumerate(ROWS) if a.primary == 44476)
    moved = (ROWS[changed].tca + 120 * u.s).iso
    lines[changed] = with_value(lines[changed], "TCA", moved)
    args = arguments(tmp_path, data_dir(tmp_path, lines))
    assert report.main(args) == 1
    summary = written(tmp_path)
    (difference,) = summary["differences"]
    assert difference["keepout"] is None
    assert not difference["reported"]
    assert (44476, 55841, None) in {
        (a["primary"], a["secondary"], a["nearest_other"])
        for a in summary["keepout_only"]
    }
    page = (tmp_path / "docs" / "screening-socrates.md").read_text(encoding="utf-8")
    assert "none within 50 km" in page
    assert "| 44476 - 55841 |" in page


def test_an_object_screen_leaves_out_is_not_compared(tmp_path: Path) -> None:
    stale = dataclasses.replace(FIXTURE[55841], epoch=START - 30 * u.day)
    catalog = [stale if omm.catalog_number == 55841 else omm for omm in objects()]
    args = arguments(tmp_path, data_dir(tmp_path, catalog=catalog))
    assert report.main(args) == 0
    summary = written(tmp_path)
    (skipped,) = summary["not_compared"]["skipped"]
    assert (skipped["primary"], skipped["secondary"]) == (44476, 55841)
    assert (skipped["object"], skipped["reason"]) == (55841, "stale")
    assert skipped["detail"].startswith("the window ends 37.00 days after the epoch")
    assert summary["groups"]["slow"]["compared"] == 21
    page = (tmp_path / "docs" / "screening-socrates.md").read_text(encoding="utf-8")
    assert "1 approach has an object `screen` left out" in page


@pytest.fixture(scope="module")
def comparison() -> report.Comparison:
    return report.compare(ROWS, objects(), START)


@pytest.fixture(scope="module")
def inputs() -> report.Inputs:
    manifest = snapshot.read_manifest(snapshot.DATA)
    return report.Inputs(tuple(ROWS), tuple(objects()), manifest["socrates"], "day")


def test_matches_keep_the_excerpts_order(comparison: report.Comparison) -> None:
    assert [m.socrates for m in comparison.matches] == ROWS
    assert all(m.agrees() for m in comparison.matches)
    assert comparison.skipped == ()
    assert comparison.primaries == 5


def test_a_known_cause_explains_a_difference(
    comparison: report.Comparison, inputs: report.Inputs
) -> None:
    match = comparison.matches[0]
    assert match.keepout is not None
    moved = dataclasses.replace(
        match.socrates, miss_distance=match.socrates.miss_distance + 2.0
    )
    changed = dataclasses.replace(
        comparison,
        matches=(report.Match(moved, match.keepout), *comparison.matches[1:]),
    )
    known = {moved.primary: "Its element set differs."}
    summary = report.summarise(changed, inputs, START, known)
    assert [d["cause"] for d in summary["differences"]] == [moved.primary]
    assert report.problems(summary, known) == []
    page = report.render_page(summary, known)
    assert f"- **{moved.primary}**: Its element set differs." in page
    assert "1 approach outside tolerance involve 1 object" in report.render_summary(
        summary, "page.md"
    )


def test_a_known_cause_that_no_longer_shows_fails(
    comparison: report.Comparison, inputs: report.Inputs
) -> None:
    summary = report.summarise(comparison, inputs, START, {})
    assert report.problems(summary, {47834: "gone"}) == [
        "47834 is a known difference but none of its approaches differs"
    ]


def test_a_difference_keepout_did_not_report_is_looked_up(
    comparison: report.Comparison, inputs: report.Inputs
) -> None:
    # The approach is there, but say keepout had not reported it: the report
    # looks for keepout's closest approach around SOCRATES's TCA.
    first = next(m for m in comparison.matches if report.group(m.socrates) == "fast")
    lost = dataclasses.replace(
        comparison,
        matches=tuple(
            report.Match(m.socrates, None) if m is first else m
            for m in comparison.matches
        ),
    )
    summary = report.summarise(lost, inputs, START, {})
    (difference,) = summary["differences"]
    assert not difference["reported"]
    assert first.keepout is not None
    assert difference["keepout"] is not None
    assert difference["keepout"]["tca_difference"] == pytest.approx(
        first.tca_difference, abs=1e-5
    )
    assert difference["keepout"]["miss_distance"] == pytest.approx(
        first.keepout.miss_distance, abs=1e-3
    )
    page = report.render_page(summary, {})
    assert "(not reported: outside the volume)" in page


def test_render_without_findings(
    comparison: report.Comparison, inputs: report.Inputs
) -> None:
    summary = copy.deepcopy(report.summarise(comparison, inputs, START, {}))
    summary["keepout_only"] = []
    summary["groups"]["slow"] = {"compared": 0, "agree": 0, "worst": None}
    page = report.render_page(summary, {})
    assert "## Found by keepout only\n\nNone.\n" in page
    assert "| slow (< 1 km/s) | 0 | 0 | - | - | - | - |" in page
    assert "it is allowed 0.5 s.\n" in page
    readme = report.render_summary(summary, "page.md")
    assert "| slow (< 1 km/s) | 0 | 0 | - | - |" in readme
    assert readme.endswith("|")


@pytest.mark.parametrize(
    ("speed", "group"), [(999.999, "slow"), (1000.0, "fast"), (14000.0, "fast")]
)
def test_groups_split_at_one_kilometre_per_second(speed: float, group: str) -> None:
    assert report.group(dataclasses.replace(ROWS[0], relative_speed=speed)) == group


def test_the_window_starts_at_midnight_of_the_first_tca() -> None:
    assert report.window_start(ROWS).isot == "2026-10-06T00:00:00.000"


def test_tolerances_bound_each_value(comparison: report.Comparison) -> None:
    match = next(m for m in comparison.matches if report.group(m.socrates) == "fast")
    assert match.keepout is not None
    tolerance = report.TOLERANCES["fast"]
    distance = match.keepout.miss_distance
    assert report.Match(
        dataclasses.replace(match.socrates, miss_distance=distance + 0.5), match.keepout
    ).agrees()
    assert not report.Match(
        dataclasses.replace(
            match.socrates, miss_distance=distance + tolerance.miss_distance + 1e-6
        ),
        match.keepout,
    ).agrees()
    speed = match.keepout.relative_speed
    assert not report.Match(
        dataclasses.replace(
            match.socrates, relative_speed=speed - tolerance.relative_speed - 1e-6
        ),
        match.keepout,
    ).agrees()
    late = dataclasses.replace(match.socrates, tca=match.keepout.tca - 1.1e-3 * u.s)
    assert not report.Match(late, match.keepout).agrees()
    assert not report.Match(match.socrates, None).agrees()


def test_the_range_change_is_undefined_without_a_range(
    comparison: report.Comparison,
) -> None:
    match = comparison.matches[0]
    assert math.isnan(report.Match(match.socrates, None).range_change)
    assert math.isnan(report.Match(match.socrates, None).tca_difference)
    assert match.keepout is not None
    docked = dataclasses.replace(
        match.keepout, secondary_state=match.keepout.primary_state
    )
    assert math.isnan(report.Match(match.socrates, docked).range_change)
    assert match.range_change >= 0.0


def test_known_differences_are_those_of_the_excerpts_objects() -> None:
    assert report.known_differences(ROWS) == {}
    row = dataclasses.replace(ROWS[0], secondary=100379)
    assert report.known_differences([row]) == {100379: report.KNOWN_DIFFERENCES[100379]}
