import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import risk_report as report
from builders import KELVINS_EXCERPT as EXCERPT
from builders import series
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba_array
from report_builders import assert_writes_then_checks, readme_with_markers

from keepout.figures import RISK_COLOURS, TrendFigure, trend_figure
from keepout.risk import (
    Assessment,
    DecisionConfig,
    PcObservation,
    RiskLevel,
    Thresholds,
    TrendConfig,
)
from keepout.risk import assess as real_assess
from keepout.sources import kelvins
from keepout.units import DAY


def paths(tmp_path: Path) -> list[str]:
    readme = readme_with_markers(tmp_path, report.START, report.END)
    return [
        "--table",
        str(EXCERPT),
        "--page",
        str(tmp_path / "docs" / "risk-kelvins.md"),
        "--images",
        str(tmp_path / "docs" / "img"),
        "--readme",
        str(readme),
    ]


def test_main_writes_then_checks(tmp_path: Path) -> None:
    assert_writes_then_checks(report.main, paths(tmp_path))
    summary = json.loads(
        (tmp_path / "docs" / "risk-kelvins.json").read_text(encoding="utf-8")
    )
    assert (summary["events"], summary["cdms"]) == (6, 24)
    assert summary["actions"] == {
        "ignore": 4,
        "monitor": 1,
        "prepare maneuver": 0,
        "maneuver": 1,
    }
    assert summary["examples"] == {"maneuver": 5490}
    assert (tmp_path / "docs" / "img" / "event-5490.svg").exists()
    page = (tmp_path / "docs" / "risk-kelvins.md").read_text(encoding="utf-8")
    assert "![Kelvins event 5490](img/event-5490.svg)" in page
    assert "([conservative basis](risk.md#risk-levels))" in page
    assert "There is no ground truth." in page
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "| maneuver | 1 | 16.7% |" in readme
    assert "(docs/img/event-5490.svg)" in readme


def test_check_needs_the_plots(tmp_path: Path) -> None:
    args = paths(tmp_path)
    assert report.main(args) == 0
    (tmp_path / "docs" / "img" / "event-5490.svg").unlink()
    assert report.main(["--check", *args]) == 1


def test_plots_are_reproducible(tmp_path: Path) -> None:
    args = paths(tmp_path)
    plot = tmp_path / "docs" / "img" / "event-5490.svg"
    assert report.main(args) == 0
    first = plot.read_bytes()
    assert report.main(args) == 0
    assert plot.read_bytes() == first
    assert b"<dc:date>" not in first


def test_the_decision_point_table_counts_every_event(tmp_path: Path) -> None:
    assert report.main(paths(tmp_path)) == 0
    summary = json.loads(
        (tmp_path / "docs" / "risk-kelvins.json").read_text(encoding="utf-8")
    )
    table = summary["decision_point_vs_final"]
    assert sum(sum(row.values()) for row in table.values()) == 6
    assert table["yellow"]["red"] == 1  # event 5490: 1.4e-5 at 2.28 d, then red


def test_examples_pick_matching_events_of_the_excerpt() -> None:
    assessed = {
        e.event_id: real_assess(e.observations()) for e in kelvins.load(EXCERPT)
    }
    assert report.choose(report.EXAMPLES, assessed) == {"maneuver": 5490}


def test_each_example_picks_its_kind_of_event() -> None:
    assessed = {
        1: real_assess(series((4.0, 1e-3), (3.0, 1e-3))),
        2: real_assess(series((3.0, 1e-12), (2.0, 1e-10), (1.0, 1e-8))),
        3: real_assess(series((3.0, 1e-8), (2.0, 1e-10), (1.0, 1e-12))),
        4: real_assess(series((3.0, 1e-3), (1.0, 1e-3))),
    }
    assert report.choose(report.EXAMPLES, assessed) == {
        "maneuver": 4,
        "prepare": 1,
        "rising": 2,
        "falling": 3,
    }


def test_an_example_prefers_more_cdms_then_the_lowest_id() -> None:
    falling = series((3.0, 1e-8), (2.0, 1e-10), (1.0, 1e-12))
    assessed = {
        9: real_assess(falling),
        3: real_assess(series((2.0, 1e-8), (1.0, 1e-12))),
        7: real_assess(falling),
    }
    assert report.choose(report.EXAMPLES, assessed) == {"falling": 7}


def test_plots_use_the_reports_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    thresholds = Thresholds(green_below=2e-7, red_from=1e-4)
    trend = TrendConfig(window=2, steady_within=0.25)
    decision = DecisionConfig(decision_point=5 * DAY)
    monkeypatch.setattr(report, "THRESHOLDS", thresholds)
    monkeypatch.setattr(report, "TREND", trend)
    monkeypatch.setattr(report, "DECISION", decision)
    settings: list[dict[str, object]] = []

    def spy(observations: Iterable[PcObservation], **kwargs: Any) -> Assessment:
        settings.append(kwargs)
        return real_assess(observations, **kwargs)

    monkeypatch.setattr(report, "assess", spy)
    figures: dict[int, Assessment] = {}
    plotted: dict[int, Assessment] = {}

    def record(assessment: Assessment) -> TrendFigure:
        assert assessment.thresholds == thresholds
        figure = trend_figure(assessment)
        figures[id(figure)] = assessment
        return figure

    def draw(figure: TrendFigure, title: str, path: Path) -> None:
        event_id = int(path.stem.removeprefix("event-"))
        assert title.startswith(f"Kelvins event {event_id} ")
        plotted[event_id] = figures[id(figure)]

    monkeypatch.setattr(report, "trend_figure", record)
    monkeypatch.setattr(report, "plot", draw)
    assert report.main(paths(tmp_path)) == 0
    expected = {
        "thresholds": thresholds,
        "trend_config": trend,
        "decision_config": decision,
    }
    assert settings
    assert all(kwargs == expected for kwargs in settings)
    assert set(plotted) == {907, 5490}
    assert all(a.thresholds is thresholds for a in plotted.values())
    assert "within the decision point (5.00 d before TCA)" in (
        plotted[5490].decision.reasons
    )


def test_the_high_risk_table_is_explained(tmp_path: Path) -> None:
    assert report.main(paths(tmp_path)) == 0
    page = tmp_path / "docs" / "risk-kelvins.md"
    summary: report.Summary = json.loads(
        page.with_suffix(".json").read_text(encoding="utf-8")
    )
    summary["challenge_label"]["ignore"]["high"] = 2
    summary["high_risk_after_tca"]["ignore"] = 1
    summary["challenge_label"]["monitor"]["high"] = 3
    summary["high_risk_levels"]["monitor"]["yellow"] = 3
    text = report.render_page(summary, "img")
    assert (
        "keepout ignores 2 high-risk events; for 1 of them the high-risk label "
        "comes from a CDM issued after TCA"
    ) in text
    assert "All 3 high-risk events keepout monitors are yellow" in text
    summary["challenge_label"]["ignore"]["high"] = 1
    text = report.render_page(summary, "img")
    assert "keepout ignores 1 high-risk event: its high-risk label comes" in text


def test_plot_uses_figure_colours() -> None:
    items = [
        PcObservation(5 * DAY, "ESA", pc=1e-30, kind="below_floor"),
        PcObservation(4 * DAY, "ESA", pc=1e-5),
        PcObservation(2 * DAY, "ESA", pc=1e-3),
    ]
    figure = trend_figure(report.assess_with_settings(items))
    assert figure.levels == (RiskLevel.GREEN, RiskLevel.YELLOW, RiskLevel.RED)
    assert figure.below_floor == (True, False, False)
    (axes,) = report.trend_plot(figure, "a trend").axes
    points = [c for c in axes.collections if isinstance(c, PathCollection)]
    assert len(points) == len(figure.levels)
    for point, level, floored in zip(
        points, figure.levels, figure.below_floor, strict=True
    ):
        colour = to_rgba_array([RISK_COLOURS[level]])
        np.testing.assert_array_equal(to_rgba_array(point.get_edgecolor()), colour)
        faces = to_rgba_array(point.get_facecolor())
        # A point below the floor is hollow.
        np.testing.assert_array_equal(faces, colour[:0] if floored else colour)
