"""The Iridium-Cosmos report: its page from a summary, and that the
committed page and JSON follow from the committed summary."""

import copy

import iridium_cosmos as study
import iridium_cosmos_data
import iridium_cosmos_report as report
import pytest
from generated import REPO, REPORTS, read_json

IMAGES = "img/iridium-cosmos"

SECTIONS = (
    "## Summary",
    "## Data",
    "## What screening saw",
    "## Pc and trend",
    "## Decision two days before",
    "## Rank among the week's approaches",
    "## A manoeuvre",
    "## Seventeen years later",
    "## Assumptions",
    "## Sources",
)


@pytest.fixture(scope="module")
def summary() -> report.Summary:
    return read_json(report.PAGE.with_suffix(".json"), report.Summary)


@pytest.fixture(scope="module")
def page(summary: report.Summary) -> str:
    return report.render_page(summary, IMAGES)


def test_render_page_sections(page: str) -> None:
    positions = [page.index(heading) for heading in SECTIONS]
    assert positions == sorted(positions)
    for figure in (report.TREND, report.RANK, report.HEATMAP):
        assert page.count(f"{IMAGES}/{figure}") == 1, figure


def test_page_states_assumptions(page: str) -> None:
    for text in (
        "no publication latency",
        f"{study.BASE_HBR:g} m",
        "same combined hard-body radius for every secondary",
        "assumed (GP age model)",
    ):
        assert text in page, text


def test_page_quotes_the_literature(page: str) -> None:
    for key in ("last_report_miss", "miss_span", "overall_rank", "report_5"):
        assert report.KELSO[key] in page, key


def test_committed_page_follows_from_the_summary(page: str) -> None:
    assert report.PAGE.read_text(encoding="utf-8") == page


def test_summary_block_in_reports(summary: report.Summary) -> None:
    text = REPORTS.read_text(encoding="utf-8")
    block = report.render_summary(summary, "case-study-iridium-cosmos.md", IMAGES)
    assert f"{report.START}\n{block}\n{report.END}" in text


def test_meaning_follows_the_decision(summary: report.Summary) -> None:
    text = report.meaning(summary)
    assert f"**{summary['decision']['action']}**" in text
    assert "do not call for a burn" in text
    changed = copy.deepcopy(summary)
    changed["decision"]["action"] = "maneuver"
    text = report.meaning(changed)
    assert "**maneuver**" in text
    assert "do not call for a burn" not in text


def test_red_claims_follow_the_numbers(summary: report.Summary, page: str) -> None:
    assert "never red" in page
    changed = copy.deepcopy(summary)
    changed["series"][-1]["pc"] = 2e-4
    assert "never red" not in report.render_page(changed, IMAGES)


def test_window_is_the_data_tool_s(summary: report.Summary, page: str) -> None:
    window = study.TCA_REFERENCE - iridium_cosmos_data.WINDOW_START
    days = summary["settings"]["window_days"]
    assert days == pytest.approx(window.to_value("day"), abs=1e-3)
    assert f"from {days:g} days before the collision" in page


def test_assumptions_page_lists_the_case_study() -> None:
    text = (REPO / "docs/assumptions.md").read_text(encoding="utf-8")
    assert "(case-study-iridium-cosmos.md" in text
