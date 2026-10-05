import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
import risk_report as report
from builders import KELVINS_EXCERPT as EXCERPT
from builders import series

from keepout.risk import (
    DAY,
    Assessment,
    DecisionConfig,
    PcObservation,
    Thresholds,
    TrendConfig,
)
from keepout.risk import assess as real_assess
from keepout.sources import kelvins
from keepout.sources.kelvins import KelvinsEvent


def paths(tmp_path: Path) -> list[str]:
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{report.START}\n{report.END}\n", encoding="utf-8")
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
    args = paths(tmp_path)
    assert report.main(["--check", *args]) == 1
    assert report.main(args) == 0
    assert report.main(["--check", *args]) == 0
    summary = json.loads(
        (tmp_path / "docs" / "risk-kelvins.json").read_text(encoding="utf-8")
    )
    assert (summary["events"], summary["cdms"]) == (6, 24)
    assert summary["actions"] == {
        "ignore": 4,
        "monitor": 1,
        "monitor, prepare manoeuvre": 0,
        "manoeuvre": 1,
    }
    assert summary["examples"] == {"manoeuvre": 5490}
    assert (tmp_path / "docs" / "img" / "event-5490.svg").exists()
    page = (tmp_path / "docs" / "risk-kelvins.md").read_text(encoding="utf-8")
    assert "![Kelvins event 5490](img/event-5490.svg)" in page
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "| manoeuvre | 1 | 16.7% |" in readme
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
    assert report.choose(report.EXAMPLES, assessed) == {"manoeuvre": 5490}


def test_each_example_picks_its_kind_of_event() -> None:
    assessed = {
        1: real_assess(series((4.0, 1e-3), (3.0, 1e-3))),
        2: real_assess(series((3.0, 1e-12), (2.0, 1e-10), (1.0, 1e-8))),
        3: real_assess(series((3.0, 1e-8), (2.0, 1e-10), (1.0, 1e-12))),
        4: real_assess(series((3.0, 1e-3), (1.0, 1e-3))),
    }
    assert report.choose(report.EXAMPLES, assessed) == {
        "manoeuvre": 4,
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
    plotted: dict[int, Assessment] = {}

    def record(event: KelvinsEvent, assessment: Assessment, path: Path) -> None:
        plotted[event.event_id] = assessment

    monkeypatch.setattr(report, "plot", record)
    assert report.main(paths(tmp_path)) == 0
    expected = {
        "thresholds": thresholds,
        "trend_config": trend,
        "decision_config": decision,
    }
    assert settings
    assert all(kwargs == expected for kwargs in settings)
    assert all(a.thresholds is thresholds for a in plotted.values())
    assert "within the decision point (5.00 d before TCA)" in (
        plotted[5490].decision.reasons
    )
