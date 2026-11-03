"""The example synthetic operator on the frozen catalog (offline, about 10 s):
the events and decisions the report describes."""

import pytest
import screening_snapshot as snapshot
import synthetic_operator_report as report

from keepout.screening import DEFAULT_DURATION, DEFAULT_VOLUME
from keepout.synthetic import load_fleet, run

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def summary() -> report.Summary:
    manifest = snapshot.read_manifest(snapshot.DATA)
    catalog = snapshot.read_catalog(snapshot.DATA)
    result = run(
        load_fleet(report.FLEET),
        catalog,
        report.window_start(manifest),
        DEFAULT_DURATION,
        DEFAULT_VOLUME,
        workers=4,
        thresholds=report.THRESHOLDS,
        decision_config=report.DECISION,
    )
    return report.summarise(
        result,
        catalog,
        snapshot.retrieval_day(manifest),
        report.shown(report.FLEET),
        report.closest(result),
    )


def test_the_example_fleet_has_twelve_approaches_in_the_week(
    summary: report.Summary,
) -> None:
    assert summary["catalog"]["element_sets"] == 19914
    assert [
        (s["catalog_number"], s["regime"], s["approaches"]) for s in summary["fleet"]
    ] == [
        (25544, "LEO", 4),
        (38771, "LEO", 7),
        (44476, "GEO", 1),
    ]
    assert summary["actions"] == {
        "ignore": 10,
        "monitor": 2,
        "prepare maneuver": 0,
        "maneuver": 0,
    }
    assert all(e["method"] == "foster" for e in summary["events"])


def test_the_two_yellow_events_are_the_diluted_ones(summary: report.Summary) -> None:
    yellow = [
        (e["primary"], e["secondary"])
        for e in summary["events"]
        if e["level"] == "yellow"
    ]
    diluted = [
        (e["primary"], e["secondary"]) for e in summary["events"] if e["diluted"]
    ]
    assert yellow == diluted == [(44476, 55841), (38771, 35196)]


def test_the_example_is_metop_b_and_a_fengyun_fragment(summary: report.Summary) -> None:
    assert summary["example"] == "38771_35196_20261012T112828.710Z_20261006T000000.000Z"
    sigmas = [s for e in summary["events"] for s in e["transverse_sigmas"]]
    assert max(sigmas) == pytest.approx(20.4e3, abs=0.1e3)
