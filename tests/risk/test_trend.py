import math
from random import Random

import pytest
from builders import at, series
from hypothesis import given
from hypothesis import strategies as st
from risk_builders import observations

from keepout.risk import (
    PcObservation,
    RiskLevel,
    Thresholds,
    Trend,
    TrendConfig,
    analyse_trend,
)
from keepout.units import DAY


def test_points_are_oldest_first_with_levels() -> None:
    trend = analyse_trend(series((1.0, 2e-4), (5.0, 1e-8), (3.0, 1e-6)))
    assert [point.time_to_tca / DAY for point in trend.points] == [5.0, 3.0, 1.0]
    assert [point.level for point in trend.points] == [
        RiskLevel.GREEN,
        RiskLevel.YELLOW,
        RiskLevel.RED,
    ]
    assert trend.latest is trend.points[-1]
    assert trend.notes == ()


@pytest.mark.parametrize(
    ("pcs", "direction"),
    [
        ((1e-8, 1e-6, 1e-4), "rising"),
        ((1e-4, 1e-6, 1e-8), "falling"),
        ((1e-5, 2e-5, 1e-5), "steady"),
    ],
)
def test_direction_follows_the_fitted_slope(
    pcs: tuple[float, float, float], direction: str
) -> None:
    trend = analyse_trend(series(*zip((3.0, 2.0, 1.0), pcs, strict=True)))
    assert trend.direction == direction


def test_slope_is_in_decades_per_day() -> None:
    trend = analyse_trend(series((3.0, 1e-8), (2.0, 1e-7), (1.0, 1e-6)))
    assert trend.slope == pytest.approx(1.0)
    assert trend.direction == "rising"


def test_only_the_window_is_fitted() -> None:
    falling_then_flat = series((5.0, 1e-3), (4.0, 1e-6), (3.0, 1e-6), (2.0, 1e-6))
    assert analyse_trend(falling_then_flat).direction == "steady"
    wide = TrendConfig(window=4)
    assert analyse_trend(falling_then_flat, config=wide).direction == "falling"


def test_steady_within_sets_the_band() -> None:
    trend = series((2.0, 1e-6), (1.0, 3e-6))  # +0.48 decades per day
    assert analyse_trend(trend).direction == "steady"
    assert analyse_trend(trend, config=TrendConfig(steady_within=0.4)).direction == (
        "rising"
    )


def test_a_slope_of_exactly_steady_within_is_steady() -> None:
    seen = series((2.0, 1e-6), (1.0, 1e-5))
    trend = analyse_trend(seen, config=TrendConfig(steady_within=1.0))
    assert (trend.slope, trend.direction) == (1.0, "steady")


def test_an_observation_at_tca_is_a_point_and_one_after_it_is_not() -> None:
    assert len(analyse_trend(series((0.0, 1e-5))).points) == 1
    assert analyse_trend([at(-1e-9, 1e-5)]).points == ()


def test_one_observation_has_no_direction() -> None:
    trend = analyse_trend(series((1.0, 1e-5)))
    assert (trend.direction, trend.slope) == ("undetermined", None)
    assert trend.notes == (
        "fewer than 2 observations with a Pc before TCA: direction undetermined",
    )


def test_simultaneous_observations_have_no_direction() -> None:
    trend = analyse_trend(series((1.0, 1e-5), (1.0, 1e-3)))
    assert trend.direction == "undetermined"
    assert trend.notes == ("observations at one time: direction undetermined",)


def test_observations_after_tca_are_left_out() -> None:
    trend = analyse_trend(series((2.0, 1e-6), (1.0, 1e-6), (-0.1, 1e-2)))
    assert len(trend.points) == 2
    assert len(trend.observations) == 3
    assert trend.notes == ("1 observation(s) issued after TCA left out",)
    latest = trend.latest_observation
    assert latest is not None
    assert latest.time_to_tca == DAY


def test_failed_observations_are_named_in_the_notes() -> None:
    failed = at(1.5, None, message_id="CDM-7", error="PcConvergenceError: stuck")
    trend = analyse_trend([*series((2.0, 1e-6), (1.0, 1e-6)), failed])
    assert len(trend.points) == 2
    assert trend.notes == (
        "no Pc for CDM-7 (1.50 d before TCA): PcConvergenceError: stuck",
    )


def test_latest_observation_may_be_a_failed_one() -> None:
    trend = analyse_trend([*series((2.0, 1e-6)), at(1.0, None)])
    assert trend.latest is not None
    assert trend.latest.time_to_tca == 2 * DAY
    latest = trend.latest_observation
    assert latest is not None
    assert not latest.ok


def test_zero_pc_is_fitted_at_the_floor() -> None:
    trend = analyse_trend(series((2.0, 0.0), (1.0, 1e-30)))
    assert trend.slope == 0.0
    assert trend.direction == "steady"
    assert trend.notes == ("Pc below 1e-30 fitted at 1e-30",)


def test_points_use_the_threshold_basis() -> None:
    interval = at(1.0, 5e-5, ci_high=2e-4)
    assert analyse_trend([interval]).points[0].level is RiskLevel.RED
    estimate = analyse_trend([interval], thresholds=Thresholds(basis="estimate"))
    assert estimate.points[0].level is RiskLevel.YELLOW
    assert estimate.points[0].value == 5e-5


def test_a_simultaneous_interval_with_a_higher_upper_end_counts_as_latest() -> None:
    interval = at(1.0, 1e-5, ci_high=1e-3)
    estimate = at(1.0, 2e-5)
    for seen in ([interval, estimate], [estimate, interval]):
        trend = analyse_trend([at(2.0, 1e-6), *seen])
        assert trend.latest is not None
        assert trend.latest.observation is interval
        assert trend.latest.level is RiskLevel.RED
        assert trend.peak is trend.latest


def test_notes_on_simultaneous_failures_do_not_depend_on_input_order() -> None:
    first = at(1.0, None, error="GeometryError: no plane")
    second = at(1.0, None, error="PcConvergenceError: stuck")
    forward = analyse_trend([at(2.0, 1e-6), first, second])
    backward = analyse_trend([second, first, at(2.0, 1e-6)])
    assert forward.notes == backward.notes
    assert forward.notes[:2] == (
        "no Pc for 1.00 d before TCA: GeometryError: no plane",
        "no Pc for 1.00 d before TCA: PcConvergenceError: stuck",
    )


def test_peak_is_the_highest_value_and_the_latest_of_equal_ones() -> None:
    trend = analyse_trend(series((3.0, 1e-4), (2.0, 1e-5), (1.0, 1e-4)))
    assert trend.peak is trend.points[-1]


def test_an_event_without_points_has_no_latest_or_peak() -> None:
    trend = analyse_trend([at(1.0, None)])
    assert (trend.latest, trend.peak, trend.points) == (None, None, ())


def test_no_observations_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least one observation"):
        analyse_trend([])


@pytest.mark.parametrize(
    ("window", "steady", "message"),
    [
        (1, 0.5, "window must be at least 2"),
        (3, 0.0, "steady_within must be finite and > 0"),
        (3, math.inf, "steady_within must be finite and > 0"),
    ],
)
def test_invalid_config_is_refused(window: int, steady: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TrendConfig(window=window, steady_within=steady)


@given(st.lists(observations(), min_size=1, max_size=8), st.randoms())
def test_the_order_of_the_input_does_not_matter(
    seen: list[PcObservation], random: Random
) -> None:
    shuffled = list(seen)
    random.shuffle(shuffled)
    first, second = analyse_trend(seen), analyse_trend(shuffled)

    def summary(trend: Trend) -> object:
        points = [(p.time_to_tca, p.value, p.level) for p in trend.points]
        kept = [
            (o.time_to_tca, o.pc, o.kind, o.diagnostics) for o in trend.observations
        ]
        return points, kept, trend.direction, trend.slope, trend.notes

    assert summary(first) == summary(second)


@given(st.lists(observations(), min_size=1, max_size=8))
def test_observations_are_kept_oldest_first(seen: list[PcObservation]) -> None:
    times = [o.time_to_tca for o in analyse_trend(seen).observations]
    assert times == sorted(times, reverse=True)
