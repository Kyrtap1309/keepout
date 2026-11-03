"""keepout's screening against CelesTrak's SOCRATES on the frozen data (offline,
about 10 s): every comparable approach found within tolerance, but those of two
objects whose element sets differ from SOCRATES's; the claims of the report's
text hold."""

import pytest
import screening_report as report
import screening_snapshot as snapshot

from keepout.screening import screen

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def inputs() -> report.Inputs:
    return report.read_inputs(snapshot.DATA)


@pytest.fixture(scope="module")
def comparison(inputs: report.Inputs) -> report.Comparison:
    start = report.window_start(inputs.approaches)
    return report.compare(inputs.approaches, inputs.catalog, start, workers=4)


@pytest.fixture(scope="module")
def summary(inputs: report.Inputs, comparison: report.Comparison) -> report.Summary:
    start = report.window_start(inputs.approaches)
    return report.summarise(comparison, inputs, start)


def test_every_comparable_approach_is_found_within_tolerance(
    summary: report.Summary,
) -> None:
    assert summary["socrates"]["comparable"] == 1275
    assert report.problems(summary) == []
    fast, slow = summary["groups"]["fast"], summary["groups"]["slow"]
    assert (fast["compared"], fast["agree"]) == (266, 261)
    assert (slow["compared"], slow["agree"]) == (1008, 1008)
    assert [
        (d["primary"], d["secondary"], d["cause"], d["reported"])
        for d in summary["differences"]
    ] == [
        (47834, 38513, 47834, True),
        (47804, 100379, 100379, True),
        (66193, 100379, 100379, True),
        (66193, 100379, 100379, False),
        (63707, 100379, 100379, False),
    ]


def test_the_known_differences_are_as_described(
    comparison: report.Comparison, summary: report.Summary
) -> None:
    # 47834: the debris it passes agrees with everything else.
    others = [
        m
        for m in comparison.matches
        if 38513 in (m.socrates.primary_number, m.socrates.secondary_number)
        and m.socrates.primary_number != 47834
    ]
    assert len(others) == 10
    assert all(m.agrees() for m in others)
    # 100379: TCA differences growing with the time since its epoch.
    rows = sorted(
        (d["tca"], d["keepout"]["tca_difference"])
        for d in summary["differences"]
        if d["cause"] == 100379 and d["keepout"] is not None
    )
    assert [round(dt, 2) for _, dt in rows] == [-0.21, -0.23, -0.31, -0.31]


def test_what_is_not_compared(summary: report.Summary) -> None:
    not_compared = summary["not_compared"]
    assert (not_compared["outside_catalog"], not_compared["other_element_sets"]) == (
        40211,
        144743,
    )
    assert [
        (s["primary"], s["secondary"], s["object"], s["reason"])
        for s in not_compared["skipped"]
    ] == [(55823, 38495, 38495, "stale")]


def test_keepout_reports_double_minima_of_formation_flyers(
    summary: report.Summary,
) -> None:
    only = summary["keepout_only"]
    assert len(only) == 5
    assert all(a["relative_speed"] < 1.0 for a in only)
    assert all(
        a["nearest_other"] is not None and a["nearest_other"] < 15 * 60 for a in only
    )


def test_socrates_leaves_out_terrasar_x_and_tandem_x(inputs: report.Inputs) -> None:
    # Both are in the excerpt with SOCRATES's element sets, so a SOCRATES
    # approach of the pair would be in it too.
    pairs = {
        frozenset((a.primary_number, a.secondary_number)) for a in inputs.approaches
    }
    objects = set().union(*pairs)
    assert {31698, 36605} <= objects
    assert frozenset((31698, 36605)) not in pairs
    by_number = {omm.catalog_number: omm for omm in inputs.catalog}
    result = screen(
        [by_number[31698]], [by_number[36605]], report.window_start(inputs.approaches)
    )
    assert len(result.approaches) > 100
    assert all(100.0 < a.range_at_tca < 500.0 for a in result.approaches)
