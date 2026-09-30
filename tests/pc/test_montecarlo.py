import importlib
from typing import Any

import numpy as np
import pytest
from builders import DRIFT_WINDOW, LEO_R, LEO_V, drifting, head_on, object_state
from scipy import stats

from keepout.conjunction import Conjunction
from keepout.errors import CovarianceError, PcConvergenceError
from keepout.objects import ObjectState
from keepout.pc import foster, hall3d, montecarlo
from keepout.pc.montecarlo import _Trials, clopper_pearson

rate_module = importlib.import_module("keepout.pc._rate")
montecarlo_module = importlib.import_module("keepout.pc.montecarlo")


def test_clopper_pearson_matches_the_beta_quantiles() -> None:
    low, high = clopper_pearson(0, 100, 0.99)
    assert low == 0.0
    assert high == pytest.approx(1.0 - 0.005 ** (1.0 / 100.0), rel=1e-12)
    low, high = clopper_pearson(30, 1000, 0.95)
    assert low == pytest.approx(stats.beta.ppf(0.025, 30, 971))
    assert high == pytest.approx(stats.beta.ppf(0.975, 31, 970))
    assert clopper_pearson(10, 10, 0.99)[1] == 1.0


def test_fast_encounter_agrees_with_2d_pc() -> None:
    conjunction = head_on(8.0, 10.0, 20.0, 5.0)
    result = montecarlo(conjunction, samples=20_000, seed=1)
    assert result.method == "montecarlo"
    expected = foster(conjunction.plane, 5.0).value
    assert result.details["ci_low"] <= expected <= result.details["ci_high"]
    assert result.details["samples"] == 20_000
    assert result.value == result.details["hits"] / 20_000


def test_slow_uncertain_drift_agrees_with_3d_not_2d_pc() -> None:
    # The one offline check that 3-D Pc is right where it differs from 2-D:
    # Monte Carlo (independent of Hall's rate, apart from where to look)
    # brackets Hall's Nc and excludes Foster's straight-line Pc.
    conjunction = drifting(velocity_sigma=0.5)
    result = montecarlo(conjunction, window=DRIFT_WINDOW, samples=20_000, seed=1)
    low, high = result.details["ci_low"], result.details["ci_high"]
    assert low <= hall3d(conjunction, window=DRIFT_WINDOW).value <= high
    assert foster(conjunction.plane, 5.0).value < low


def test_a_seed_makes_the_run_reproducible() -> None:
    conjunction = head_on(8.0, 10.0, 20.0, 5.0)
    first = montecarlo(conjunction, samples=5_000, seed=3)
    second = montecarlo(conjunction, samples=5_000, seed=3)
    other = montecarlo(conjunction, samples=5_000, seed=4)
    assert first.value == second.value
    assert first.details == second.details
    assert first.diagnostics == second.diagnostics
    assert other.details["hits"] != first.details["hits"]


def test_stops_when_the_interval_is_narrow_enough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Small batches: the stopping rule is checked between several of them.
    monkeypatch.setattr(montecarlo_module, "MC_BATCH", 2_000)
    conjunction = head_on(8.0, 10.0, 20.0, 5.0)
    result = montecarlo(conjunction, rtol=0.2, max_samples=100_000, seed=5)
    half = 0.5 * (result.details["ci_high"] - result.details["ci_low"])
    assert half <= 0.2 * result.value
    assert 2_000 < result.details["samples"] < 100_000
    expected = foster(conjunction.plane, 5.0).value
    assert result.details["ci_low"] <= expected <= result.details["ci_high"]
    assert result.diagnostics == ()


def test_budget_exhausted_is_reported() -> None:
    far = head_on(200.0, 10.0, 20.0, 5.0)  # Pc ~ 1e-43: no hits
    result = montecarlo(far, max_samples=2_000, seed=1)
    assert result.value == 0.0
    assert result.details["samples"] == 2_000
    assert result.details["ci_high"] > 0.0
    assert result.diagnostics[0].startswith("Monte Carlo stopped at 2000 trials")


def test_trials_already_inside_at_the_window_start_do_not_count() -> None:
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
    base = head_on(30.0, 10.0, 20.0, 5.0)
    conjunction = Conjunction(
        ObjectState("P", base.primary.state, None), base.secondary, 5.0
    )
    with pytest.raises(
        CovarianceError, match="montecarlo needs both covariances; unknown: primary"
    ):
        montecarlo(conjunction, samples=10)


@pytest.mark.parametrize(
    "options",
    [{"samples": 0}, {"rtol": 0.0}, {"max_samples": 0}, {"confidence": 1.0}],
)
def test_bad_arguments_are_rejected(options: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=r"samples|rtol|confidence"):
        montecarlo(head_on(30.0, 10.0, 20.0, 5.0), **options)  # type: ignore[arg-type]


def test_no_encounter_in_the_window_runs_no_trial() -> None:
    result = montecarlo(head_on(30.0, 10.0, 20.0, 5.0), window=(100.0, 101.0))
    assert result.value == 0.0
    assert result.details["samples"] == 0.0
    (note,) = result.diagnostics
    assert note.startswith("no encounter: the collision rate is negligible")


def test_failed_iteration_inside_an_encounter_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # As in Hall: a fallback inside a segment could have narrowed the
    # segments searched, so the run stops instead.
    real = rate_module._peak_overlap

    def failing(*args: Any) -> Any:
        return None if 4.2e-4 <= args[2] <= 5.0e-4 else real(*args)

    monkeypatch.setattr(rate_module, "_peak_overlap", failing)
    with pytest.raises(PcConvergenceError, match="inside an encounter segment"):
        montecarlo(head_on(30.0, 10.0, 20.0, 5.0), samples=1_000, seed=1)


class _Line(_Trials):
    """Relative motion ``r = (1, t - 0.1, 0)`` with a misleading acceleration.

    The acceleration ``(0, 20, 0)`` makes Newton's slope wrong, so from the
    interval's midpoint 0 (distance 1.005) it jumps to 2 and then creeps back
    by ~1/20 of the way per step: after every step it is still far from the
    closest approach at 0.1.
    """

    def __init__(self) -> None:
        pass

    def relative(self, t: Any, rows: Any = None) -> Any:
        t = np.asarray(t, dtype=np.float64)
        one = np.ones_like(t)
        r = np.stack([one, t - 0.1, 0.0 * t], axis=1)
        v = np.stack([0.0 * t, one, 0.0 * t], axis=1)
        a = np.stack([0.0 * t, 20.0 * one, 0.0 * t], axis=1)
        return r, v, a


def test_newton_keeps_a_hit_it_passed() -> None:
    below = _Line().closest_below(np.array([0]), -4.0, 4.0, 1.2)
    assert below.tolist() == [True]


def test_confidence_is_printed_with_its_digits() -> None:
    result = montecarlo(
        head_on(30.0, 10.0, 20.0, 5.0),
        rtol=1e-6,
        max_samples=1_000,
        confidence=0.995,
        seed=1,
    )
    assert "the 99.5% interval" in result.diagnostics[0]
