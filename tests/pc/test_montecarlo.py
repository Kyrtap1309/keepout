import numpy as np
import pytest
from builders import (
    DRIFT_WINDOW,
    LEO_R,
    LEO_V,
    TYPICAL_HEAD_ON,
    drifting,
    head_on,
    object_state,
    without_covariance,
)
from faults import (
    misleading_newton_search,
    monte_carlo_batches,
    not_converging,
    overlap_failing_where,
)

from keepout.conjunction import Conjunction
from keepout.errors import CovarianceError, PcConvergenceError
from keepout.pc import MonteCarloOptions, PcResult, foster, hall3d, montecarlo


def interval(result: PcResult) -> tuple[float, float]:
    """The result's confidence interval, which Monte Carlo always gives."""
    assert result.interval is not None
    return result.interval


def test_fast_encounter_agrees_with_2d_pc() -> None:
    # A fixed seed whose 99% interval holds the exact Pc (seed 1's misses it,
    # as 1 seed in 100 may).
    conjunction = head_on(8.0, 10.0, 20.0, 5.0)
    result = montecarlo(conjunction, samples=20_000, seed=2)
    assert result.method == "montecarlo"
    expected = foster(conjunction.plane, 5.0).value
    assert interval(result)[0] <= expected <= interval(result)[1]
    assert result.details["samples"] == 20_000
    assert result.value == result.details["hits"] / 20_000
    # The interval is a field; details keep the numbers shown with it.
    assert result.details["confidence"] == 0.99
    assert {"ci_low", "ci_high"}.isdisjoint(result.details)


def test_slow_uncertain_drift_agrees_with_3d_not_2d_pc() -> None:
    # The one offline check that 3-D Pc is right where it differs from 2-D:
    # Monte Carlo (independent of Hall's rate, apart from where to look)
    # brackets Hall's Nc and excludes Foster's straight-line Pc.
    conjunction = drifting(velocity_sigma=0.5)
    result = montecarlo(conjunction, window=DRIFT_WINDOW, samples=20_000, seed=1)
    low, high = interval(result)
    assert low <= hall3d(conjunction, window=DRIFT_WINDOW).value <= high
    assert foster(conjunction.plane, 5.0).value < low


def test_a_seed_makes_the_run_reproducible() -> None:
    conjunction = head_on(8.0, 10.0, 20.0, 5.0)
    first = montecarlo(conjunction, samples=5_000, seed=3)
    second = montecarlo(conjunction, samples=5_000, seed=3)
    assert first.value == second.value
    assert first.details == second.details
    assert first.diagnostics == second.diagnostics


@pytest.mark.parametrize("seed", [3, 4])
def test_each_seed_s_interval_holds_the_exact_pc(seed: int) -> None:
    # A fast encounter, where the 2-D Pc is exact: any seed's 99% interval
    # holds it (both seeds are fixed, so this cannot flake).
    conjunction = head_on(8.0, 10.0, 20.0, 5.0)
    result = montecarlo(conjunction, samples=5_000, seed=seed)
    expected = foster(conjunction.plane, 5.0).value
    assert interval(result)[0] <= expected <= interval(result)[1]


def test_stops_when_the_interval_is_narrow_enough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Small batches: the stopping rule is checked between several of them.
    monte_carlo_batches(monkeypatch, 2_000)
    conjunction = head_on(8.0, 10.0, 20.0, 5.0)
    result = montecarlo(conjunction, rtol=0.2, max_samples=100_000, seed=5)
    low, high = interval(result)
    half = 0.5 * (high - low)
    assert half <= 0.2 * result.value
    assert 2_000 < result.details["samples"] < 100_000
    expected = foster(conjunction.plane, 5.0).value
    assert interval(result)[0] <= expected <= interval(result)[1]
    assert result.diagnostics == ()


@pytest.mark.parametrize(
    "options",
    [{"samples": 6_000, "seed": 7}, {"rtol": 0.15, "max_samples": 100_000, "seed": 7}],
)
def test_processes_give_the_result_of_one(
    monkeypatch: pytest.MonkeyPatch, options: MonteCarloOptions
) -> None:
    # Each batch has its own stream of the seed: two processes sample the
    # same batches, and stop after the same one.
    monte_carlo_batches(monkeypatch, 2_000)
    conjunction = head_on(8.0, 10.0, 20.0, 5.0)
    alone = montecarlo(conjunction, **options)
    in_two: MonteCarloOptions = {**options, "workers": 2}
    shared = montecarlo(conjunction, **in_two)
    assert alone.details["samples"] > 2_000
    assert (shared.value, shared.interval) == (alone.value, alone.interval)
    assert shared.details == alone.details
    assert shared.diagnostics == alone.diagnostics


def test_budget_exhausted_is_reported() -> None:
    far = head_on(200.0, 10.0, 20.0, 5.0)  # Pc ~ 1e-43: no hits
    result = montecarlo(far, max_samples=2_000, seed=1)
    assert result.value == 0.0
    assert result.details["samples"] == 2_000
    low, high = interval(result)
    assert low == 0.0 < high
    assert result.diagnostics[0].startswith("Monte Carlo stopped at 2000 samples")


def test_samples_already_inside_at_the_window_start_do_not_count() -> None:
    # The secondary sits inside the hard body at TCA; a window starting at
    # TCA begins with nearly every trial already overlapping.
    primary = object_state(LEO_R, LEO_V, 0.01 * np.eye(3))
    secondary = object_state(
        LEO_R + np.array([1.0, 0.0, 0.0]), -LEO_V, 0.01 * np.eye(3)
    )
    conjunction = Conjunction(primary, secondary, 5.0)
    result = montecarlo(conjunction, window=(0.0, 0.01), samples=2_000, seed=2)
    assert result.details["inside_at_start"] == 1.0
    assert result.value == 0.0
    assert "already overlap" in result.diagnostics[0]


def test_unknown_covariance_is_rejected() -> None:
    base = TYPICAL_HEAD_ON
    conjunction = Conjunction(without_covariance(base.primary), base.secondary, 5.0)
    with pytest.raises(
        CovarianceError, match="montecarlo needs both covariances; unknown: primary"
    ):
        montecarlo(conjunction, samples=10)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"samples": 0}, "samples must be at least 1"),
        ({"rtol": 0.0}, "rtol must be finite and > 0"),
        ({"max_samples": 0}, "max_samples must be at least 1"),
        ({"confidence": 1.0}, r"confidence must be in \(0, 1\), got 1.0"),
        ({"workers": 0}, "workers must be at least 1"),
    ],
)
def test_bad_arguments_are_rejected(options: MonteCarloOptions, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        montecarlo(TYPICAL_HEAD_ON, **options)


def test_no_encounter_in_the_window_runs_no_trial() -> None:
    result = montecarlo(TYPICAL_HEAD_ON, window=(100.0, 101.0))
    assert result.value == 0.0
    assert result.details["samples"] == 0.0
    assert result.interval == (0.0, 0.0)
    (note,) = result.diagnostics
    assert note.startswith(
        "no encounter: collision rate negligible over the whole window"
    )


def test_failed_iteration_inside_an_encounter_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # As in Hall: a fallback inside a segment could have narrowed the
    # segments searched, so the run stops instead.
    overlap_failing_where(
        monkeypatch, lambda t: (t >= 4.2e-4) & (t <= 5.0e-4), not_converging
    )
    with pytest.raises(PcConvergenceError, match="inside an encounter segment"):
        montecarlo(TYPICAL_HEAD_ON, samples=1_000, seed=1)


def test_newton_keeps_a_hit_it_passed() -> None:
    closest_below = misleading_newton_search()
    assert closest_below(np.array([0]), -4.0, 4.0, 1.2).tolist() == [True]


def test_confidence_is_printed_with_its_digits() -> None:
    result = montecarlo(
        TYPICAL_HEAD_ON,
        rtol=1e-6,
        max_samples=1_000,
        confidence=0.995,
        seed=1,
    )
    assert "the 99.5% interval" in result.diagnostics[0]


@pytest.mark.slow
def test_a_long_fast_run_is_unbiased_to_a_percent_or_two() -> None:
    # The 20k-sample checks above have a 99% interval of ~+-8%: a bias below
    # that passes them. Here N = 3e5 at Pc ~ 0.27 (sigmas as small as the
    # hard body, so the hits are many) gives a binomial std
    # sqrt(Pc (1 - Pc) / N) ~ 0.3% of Pc; 5 std (the seed is fixed, so this
    # cannot flake) still sees a 1.5% bias. About 25 s.
    samples = 300_000
    conjunction = head_on(5.0, 5.0, 5.0, 5.0)
    expected = foster(conjunction.plane, 5.0).value
    result = montecarlo(conjunction, samples=samples, seed=11)
    std = np.sqrt(expected * (1.0 - expected) / samples)
    assert abs(result.value - expected) <= 5.0 * std
