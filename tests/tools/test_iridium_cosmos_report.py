"""The Iridium-Cosmos report: its page from a summary, and that the
committed page and JSON follow from the committed summary."""

import iridium_cosmos as study
import iridium_cosmos_report as report
import pytest
from generated import REPORTS, read_json

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


def test_render_page_sections(summary: report.Summary) -> None:
    page = report.render_page(summary, "img/iridium-cosmos")
    positions = [page.index(heading) for heading in SECTIONS]
    assert positions == sorted(positions)
    for figure in (report.TREND, report.RANK, report.HEATMAP):
        assert f"img/iridium-cosmos/{figure}" in page


def test_page_states_assumptions(summary: report.Summary) -> None:
    page = report.render_page(summary, "img/iridium-cosmos")
    for text in (
        "no publication latency",
        f"{study.BASE_HBR:g} m",
        "same combined hard-body radius for every secondary",
        "assumed (GP age model)",
        "Kelso (2009)",
    ):
        assert text in page, text


def test_page_quotes_the_literature(summary: report.Summary) -> None:
    page = report.render_page(summary, "img/iridium-cosmos")
    for value in ("584 m", "117 m", "1.812 km", "152"):
        assert value in page, value


def test_committed_page_follows_from_the_summary(summary: report.Summary) -> None:
    images = "img/iridium-cosmos"
    assert report.PAGE.read_text(encoding="utf-8") == report.render_page(
        summary, images
    )


def test_summary_block_in_reports(summary: report.Summary) -> None:
    text = REPORTS.read_text(encoding="utf-8")
    block = report.render_summary(
        summary, "case-study-iridium-cosmos.md", "img/iridium-cosmos"
    )
    assert f"{report.START}\n{block}\n{report.END}" in text
