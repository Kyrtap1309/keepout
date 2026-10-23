import importlib
import math
import pickle
import re
from typing import Final

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
    shifted_back,
    without_covariance,
)
from faults import (
    Failure,
    WindowSearch,
    never_converging_overlap,
    not_converging,
    overlap_failing_where,
    raising,
    rate_raising,
    rate_raising_for,
    rate_returning,
    straight_line_closest_at,
    time_integral_orders,
    window_search,
)
from pc_builders import side_by_side
from reference_pc import (
    SHELL,
    SIDE_BY_SIDE_NC,
    SIDE_BY_SIDE_NC_ERROR,
    SIDE_BY_SIDE_SEED,
    SIDE_BY_SIDE_WINDOW,
    side_by_side_entries,
)
from scipy.spatial.transform import Rotation

from keepout.conjunction import Conjunction
from keepout.errors import CovarianceError, GeometryError, PcConvergenceError
from keepout.orbits import orbital_period
from keepout.pc import _overlap as overlap_module
from keepout.pc import foster, hall3d
from keepout.pc._encounters import Encounters, UnusableOrbit
from keepout.pc._plane import MissingCovariance

# The package exports the function hall3d under the module's name.
hall3d_module = importlib.import_module("keepout.pc.hall3d")


@pytest.mark.parametrize(
    ("miss", "sx", "sz", "radius"), [(30.0, 10.0, 20.0, 5.0), (80.0, 40.0, 15.0, 10.0)]
)
def test_fast_encounter_matches_2d_pc(
    miss: float, sx: float, sz: float, radius: float
) -> None:
    # 15 km/s head-on: the encounter lasts milliseconds, so the straight-line
    # 2-D model is exact to far below the tolerance.
    conjunction = head_on(miss, sx, sz, radius)
    result = hall3d(conjunction)
    assert result.method == "hall3d"
    assert result.value == pytest.approx(
        foster(conjunction.plane, radius).value, rel=1e-4
    )
    assert result.details["encounters"] == 1.0
    assert result.details["fallbacks"] == 0.0
    assert result.diagnostics == ()


def test_a_certain_slow_drift_matches_the_2d_pc() -> None:
    # Without velocity uncertainty the minutes-long drift is still close to
    # a straight line at a known speed: 3-D and 2-D agree.
    conjunction = drifting(velocity_sigma=1e-4)
    result = hall3d(conjunction, window=DRIFT_WINDOW)
    assert result.value == pytest.approx(foster(conjunction.plane, 5.0).value, rel=1e-3)


def test_velocity_uncertainty_changes_the_3d_pc() -> None:
    # Velocity sigmas of the order of the 0.3 m/s drift spread the crossing
    # times: Nc rises by about a fifth over the 2-D Pc, which ignores them.
    conjunction = drifting(velocity_sigma=0.5)
    two_d = foster(conjunction.plane, 5.0).value
    result = hall3d(conjunction, window=DRIFT_WINDOW)
    assert 1.15 * two_d < result.value < 1.3 * two_d


def test_swapping_the_objects_changes_nothing() -> None:
    base = TYPICAL_HEAD_ON
    swapped = Conjunction(base.secondary, base.primary, 5.0)
    assert hall3d(swapped).value == pytest.approx(hall3d(base).value, rel=1e-8)


def test_explicit_window_is_used() -> None:
    conjunction = TYPICAL_HEAD_ON
    result = hall3d(conjunction, window=(-0.002, 0.002))
    window = (result.details["window_start_s"], result.details["window_end_s"])
    assert window == (-0.002, 0.002)


@pytest.mark.parametrize("bad", [(1.0, -1.0), (0.0, math.inf)])
def test_invalid_window_is_rejected(bad: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="window"):
        hall3d(TYPICAL_HEAD_ON, window=bad)


def test_unknown_covariance_is_rejected() -> None:
    base = TYPICAL_HEAD_ON
    conjunction = Conjunction(base.primary, without_covariance(base.secondary), 5.0)
    with pytest.raises(CovarianceError, match="hall3d needs both covariances"):
        hall3d(conjunction)


def test_unbound_orbit_is_rejected() -> None:
    fast = np.array([0.0, 15000.0, 0.0])
    primary = object_state(LEO_R, fast, np.eye(3))
    secondary = object_state(LEO_R + np.array([30.0, 0.0, 0.0]), LEO_V, np.eye(3))
    with pytest.raises(GeometryError, match="unbound"):
        hall3d(Conjunction(primary, secondary, 5.0))


def test_the_encounter_scan_names_the_object_not_a_method() -> None:
    # Shared by hall3d, montecarlo and auto's fallback: each adds its own
    # context to these errors.
    base = TYPICAL_HEAD_ON
    unknown = Conjunction(base.primary, without_covariance(base.secondary), 5.0)
    with pytest.raises(MissingCovariance) as missing:
        Encounters.scan(unknown, None)
    assert missing.value.roles == ("secondary",)
    assert str(missing.value) == "both covariances are needed; unknown: secondary"
    fast = object_state(LEO_R, np.array([0.0, 15000.0, 0.0]), np.eye(3))
    with pytest.raises(UnusableOrbit, match="unbound") as unusable:
        Encounters.scan(Conjunction(base.primary, fast, 5.0), None)
    assert unusable.value.role == "secondary"
    for error in (missing.value, unusable.value):
        copy = pickle.loads(pickle.dumps(error))
        assert (type(copy), str(copy)) == (type(error), str(error))


def test_zero_relative_velocity_nc_matches_a_brute_force_count() -> None:
    # Co-located objects 10 m apart on the same circular orbit never meet in
    # a straight line, but their uncertainties overlap: 3-D Pc is defined.
    # SIDE_BY_SIDE_NC is a brute-force count with a 0.8% standard error;
    # rel=0.025 is three of them.
    result = hall3d(side_by_side(), window=SIDE_BY_SIDE_WINDOW)
    assert result.value == pytest.approx(SIDE_BY_SIDE_NC, rel=0.025)


def test_zero_relative_velocity_is_noted_as_an_upper_bound() -> None:
    result = hall3d(side_by_side(), window=SIDE_BY_SIDE_WINDOW)
    assert result.diagnostics == (
        (
            "relative velocity is zero: Nc counts repeated entries into the "
            "hard-body sphere, so it is an upper bound on Pc"
        ),
    )


def test_zero_relative_velocity_is_not_a_long_encounter() -> None:
    # The straight-line duration is undefined without a relative velocity:
    # the zero-velocity note above stands in for the long-encounter one.
    assert not hall3d_module.is_long(side_by_side())


@pytest.mark.validation
def test_side_by_side_nc_matches_a_brute_force_count() -> None:
    # Recomputes SIDE_BY_SIDE_NC (~15 s): 1e8 Monte Carlo trials of the TCA
    # Gaussians, moved by two-body Runge-Kutta steps of 0.5 s.
    count = side_by_side_entries(10**8, seed=SIDE_BY_SIDE_SEED)
    assert count.largest_drift < SHELL
    nc = count.entries / count.trials
    assert nc == pytest.approx(SIDE_BY_SIDE_NC, abs=3.0 * SIDE_BY_SIDE_NC_ERROR)


def test_failed_iteration_where_it_matters_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    never_converging_overlap(monkeypatch)
    with pytest.raises(PcConvergenceError, match="did not converge"):
        hall3d(TYPICAL_HEAD_ON)


def crossing_twice() -> Conjunction:
    """A 1 m/s crossing at zero miss, 1 m sigmas against a 5 m hard body.

    Inclined by 1 m/s, the two orbits cross again half an orbit later, where
    the along-track uncertainty has grown: Nc ~ 0.998 at TCA, ~0.005 there.
    Pass ``window=TWICE_WINDOW``.
    """
    covariance = np.eye(3)
    return Conjunction(
        object_state(LEO_R, LEO_V, covariance, velocity_sigma=0.01),
        object_state(
            LEO_R, LEO_V + np.array([0.0, 0.0, 1.0]), covariance, velocity_sigma=0.01
        ),
        5.0,
    )


_LEO_PERIOD = orbital_period(crossing_twice().primary.state)
assert _LEO_PERIOD is not None
TWICE_WINDOW: Final = (-0.25 * _LEO_PERIOD, 0.75 * _LEO_PERIOD)
"""A window holding both crossings of :func:`crossing_twice`."""


def test_nc_above_one_is_reported_and_pc_set_to_one() -> None:
    result = hall3d(crossing_twice(), window=TWICE_WINDOW)
    nc = result.details["nc"]
    assert nc > 1.0
    assert result.value == 1.0
    assert result.diagnostics[0] == (
        f"expected number of collisions Nc = {nc:.4g} exceeds 1; Pc set to 1"
    )


def test_several_encounters_are_reported() -> None:
    result = hall3d(crossing_twice(), window=TWICE_WINDOW)
    assert result.details["encounters"] == 2.0
    largest = result.details["largest_encounter_nc"]
    assert largest < result.details["nc"]
    assert result.diagnostics[1].startswith(
        f"2 encounters in the window: Pc lies between the largest one's Nc "
        f"({largest:.3g})"
    )


def test_rate_at_the_edge_of_an_explicit_window_is_not_reported() -> None:
    # The automatic window of this slow crossing ends with the rate still
    # significant (see test_automatic_window_stops_at_half_an_orbit); the
    # same window passed explicitly is the caller's choice.
    conjunction = slow_crossing()
    automatic = hall3d(conjunction)
    window = (automatic.details["window_start_s"], automatic.details["window_end_s"])
    explicit = hall3d(conjunction, window=window)
    assert explicit.value == pytest.approx(automatic.value, rel=1e-6)
    assert not any("pass window=" in note for note in explicit.diagnostics)


@pytest.mark.parametrize(
    "window", [(-1000.05, 999.97), (-300.03, 700.0), (-30.0, 30.0015)]
)
def test_long_explicit_window_resolves_the_encounter(
    window: tuple[float, float],
) -> None:
    # The crossing takes 0.4 ms: a uniform grid of at most 20001 points over
    # these windows steps over it; the screen must still find and resolve it.
    conjunction = TYPICAL_HEAD_ON
    result = hall3d(conjunction, window=window)
    assert result.value == pytest.approx(foster(conjunction.plane, 5.0).value, rel=1e-4)
    assert result.details["encounters"] == 1.0


def test_unresolvable_window_raises() -> None:
    # Days of head-on passes, each 0.4 ms long: more scan points than allowed.
    with pytest.raises(
        PcConvergenceError, match="cannot resolve the collision rate"
    ) as info:
        hall3d(TYPICAL_HEAD_ON, window=(-1e5, 1e5))
    assert "'hall3d', window=(a, b)" in str(info.value)


def test_window_without_an_encounter_is_reported() -> None:
    result = hall3d(TYPICAL_HEAD_ON, window=(100.0, 101.0))
    assert result.value == 0.0
    (note,) = result.diagnostics
    assert note.startswith("collision rate negligible over the whole window")


@pytest.mark.parametrize("failure", [not_converging, raising])
def test_failure_inside_the_encounter_raises(
    monkeypatch: pytest.MonkeyPatch,
    failure: Failure,
) -> None:
    # The rate peaks near -0.25 ms; at +0.42 to +0.50 ms it is 1e-16 to 1e-10
    # of its peak: inside the encounter segment, but a fallback there stays
    # below the relative tolerance of FALLBACK_TOLERANCE.
    overlap_failing_where(monkeypatch, lambda t: (t >= 4.2e-4) & (t <= 5.0e-4), failure)
    with pytest.raises(PcConvergenceError, match="did not converge"):
        hall3d(TYPICAL_HEAD_ON)


@pytest.mark.parametrize("failure", [not_converging, raising])
def test_failure_in_the_far_tail_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
    failure: Failure,
) -> None:
    conjunction = TYPICAL_HEAD_ON
    overlap_failing_where(monkeypatch, lambda t: np.abs(t) >= 3e-3, failure)
    result = hall3d(conjunction, window=(-0.02, 0.02))
    assert result.details["fallbacks"] > 0
    assert result.value == pytest.approx(foster(conjunction.plane, 5.0).value, rel=1e-4)


def test_failing_fallback_raises_a_convergence_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    never_converging_overlap(monkeypatch)
    rate_raising(monkeypatch, np.linalg.LinAlgError("Singular matrix"))
    with pytest.raises(PcConvergenceError, match="cannot be computed"):
        hall3d(TYPICAL_HEAD_ON)


def test_a_rate_failure_names_a_failing_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The rates are computed in batches; the error names a time whose rate
    # fails on its own, not merely the first of its batch. The secondary
    # closes along -y at ~15 km/s, so these are the times after 0.1 ms.
    rate_raising_for(
        monkeypatch,
        lambda gaussians: gaussians.relative_position[:, 1] < -1.5,
        np.linalg.LinAlgError("Singular matrix"),
    )
    with pytest.raises(PcConvergenceError, match="cannot be computed") as raised:
        hall3d(TYPICAL_HEAD_ON)
    named = re.search(r"rate at t = (\S+) s", str(raised.value))
    assert named is not None
    assert float(named.group(1)) > 0.9e-4


def test_non_finite_rate_raises_a_convergence_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rate_returning(monkeypatch, math.nan)
    with pytest.raises(PcConvergenceError, match=r"rate at t = .* is not finite"):
        hall3d(TYPICAL_HEAD_ON)


def test_a_time_integral_that_overflows_raises_a_convergence_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every rate is finite, but their Clenshaw-Curtis sum is not.
    rate_returning(monkeypatch, 1e308)
    with (
        pytest.raises(PcConvergenceError, match=r"Nc is not finite \(inf\)"),
        pytest.warns(RuntimeWarning, match="overflow"),
    ):
        hall3d(TYPICAL_HEAD_ON)


@pytest.mark.parametrize("rtol", [0.0, -1e-6, math.nan, math.inf])
def test_rtol_is_checked(rtol: float) -> None:
    with pytest.raises(ValueError, match="rtol"):
        hall3d(TYPICAL_HEAD_ON, rtol=rtol)


def test_unmet_rtol_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    # Orders 4 and 8 cannot integrate a padded encounter to 1e-4.
    time_integral_orders(monkeypatch, start=4, largest=8)
    result = hall3d(TYPICAL_HEAD_ON, rtol=1e-15)
    (note,) = result.diagnostics
    assert re.match(
        r"the time integral did not reach rtol: its worst piece agrees to "
        r"\d\.\de[-+]\d+ of Nc \(rtol 1\.0e-15\)",
        note,
    )


def test_small_rtol_shortfall_is_not_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    # Orders 32 and 64 agree to far better than QUIET_AGREEMENT, not to 1e-15.
    time_integral_orders(monkeypatch, largest=64)
    result = hall3d(TYPICAL_HEAD_ON, rtol=1e-15)
    assert result.diagnostics == ()


def correlated_head_on(correlation: float) -> Conjunction:
    """Head-on, 100 m miss along x, with x correlated with the along-track y.

    The correlation moves the peak of the rate ``correlation * sigma_y * miss
    / sigma_x`` = 9.5 sigma_y along track, beyond the straight-line
    half-window of 8.6 sigma_y about TCA.
    """
    sx, sy, sz = 10.0, 3.0, 20.0
    position = np.diag([sx**2, sy**2, sz**2])
    position[0, 1] = position[1, 0] = correlation * sx * sy
    primary = object_state(LEO_R, LEO_V, position, object_id="PRI")
    secondary = object_state(
        LEO_R + np.array([100.0, 0.0, 0.0]),
        -LEO_V,
        np.zeros((3, 3)),
        object_id="SEC",
    )
    return Conjunction(primary, secondary, 5.0)


def test_automatic_window_grows_to_hold_a_displaced_peak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conjunction = correlated_head_on(0.95)
    widths: list[float] = []

    def record(search: WindowSearch) -> WindowSearch:
        widths.append(search.half_width)
        return search

    window_search(monkeypatch, record)
    result = hall3d(conjunction)
    # One call, centred on TCA (the straight-line closest approach): the
    # window doubles once, to 17.2 sigma_y, which holds the 9.5 sigma_y peak.
    [initial] = widths
    assert result.details["window_end_s"] == pytest.approx(2.0 * initial, rel=1e-12)
    assert result.details["window_start_s"] == pytest.approx(-2.0 * initial, rel=1e-12)
    assert result.diagnostics == ()
    # Pc ~ 1.1e-22, ten sigma out radially, so abs=0. Hall's Gaussian lives
    # in equinoctial elements linearised about the peak-overlap state; its
    # curvilinear map lowers the Mahalanobis d^2 of the hard body by ~0.0086
    # out of 100, which raises Pc by ~0.37% over the straight-line 2-D model.
    # That gap is intended, not numerical (rtol and orders move it < 1e-5):
    # it grows as ~3.7e-6 k^3 at k sigma (and as 1/r_orbit), and vanishes
    # (~1e-6) with the mean-centred expansion. rel=1e-2 leaves a 2.7x margin
    # and still catches a window cut short at its initial width (~30% lost).
    assert result.value == pytest.approx(
        foster(conjunction.plane, 5.0).value, rel=1e-2, abs=0.0
    )


def slow_crossing() -> Conjunction:
    """A 0.2 m/s crossing with 5 m position and 0.1 m/s velocity sigmas."""
    covariance = 25.0 * np.eye(3)
    return Conjunction(
        object_state(LEO_R, LEO_V, covariance, velocity_sigma=0.1),
        object_state(
            LEO_R + np.array([30.0, 0.0, 0.0]),
            LEO_V + np.array([0.0, 0.0, 0.2]),
            covariance,
            velocity_sigma=0.1,
        ),
        5.0,
    )


def test_automatic_window_stops_at_half_an_orbit() -> None:
    # The rate decays slowly, and the crossing orbits meet again half an
    # orbit later.
    conjunction = slow_crossing()
    result = hall3d(conjunction)
    period = orbital_period(conjunction.primary.state)
    assert period is not None
    half_period = 0.5 * period
    assert result.details["window_end_s"] == pytest.approx(half_period, rel=1e-3)
    assert any("pass window=" in note for note in result.diagnostics)


@pytest.mark.parametrize("dt", [0.01, 1.0])
def test_states_before_closest_approach_find_the_encounter(dt: float) -> None:
    # The straight-line closest approach is dt after the states' epoch: the
    # automatic window must cover it, not only t = 0.
    conjunction = shifted_back(TYPICAL_HEAD_ON, dt)
    expected = foster(conjunction.plane, 5.0).value
    result = hall3d(conjunction)
    assert result.value == pytest.approx(expected, rel=1e-4)
    assert result.details["window_end_s"] > dt


def test_an_encounter_beyond_the_initial_window_is_searched_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Centre the window on t = 0 only: the rate is 0 there, and the window
    # keeps doubling until it reaches the encounter 1 s later.
    def at_tca(search: WindowSearch) -> WindowSearch:
        return search._replace(centre=(0.0, 0.0))

    window_search(monkeypatch, at_tca)
    conjunction = shifted_back(TYPICAL_HEAD_ON, 1.0)
    assert hall3d(conjunction).value == pytest.approx(
        foster(conjunction.plane, 5.0).value, rel=1e-4
    )


def test_a_missed_straight_line_approach_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Forbid any growth of a window about t = 0: the scan then finds nothing,
    # although the straight-line closest approach is 30 m away at t = 1 s.
    def frozen(search: WindowSearch) -> WindowSearch:
        return WindowSearch(search.half_width, search.half_width, (0.0, 0.0))

    window_search(monkeypatch, frozen)
    result = hall3d(shifted_back(TYPICAL_HEAD_ON, 1.0))
    assert result.value == 0.0
    assert result.diagnostics[0].startswith("collision rate negligible")
    assert result.diagnostics[1].startswith(
        "the straight-line closest approach (30 m, t = 1 s)"
    )


def test_a_straight_line_approach_beyond_reach_is_not_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # As above, but 1 km off: two-body motion is not what keeps them apart.
    def frozen(search: WindowSearch) -> WindowSearch:
        return WindowSearch(search.half_width, search.half_width, (0.0, 0.0))

    window_search(monkeypatch, frozen)
    result = hall3d(shifted_back(head_on(1000.0, 10.0, 20.0, 5.0), 1.0))
    assert result.value == 0.0
    (note,) = result.diagnostics
    assert note.startswith("collision rate negligible")


def test_nc_of_a_long_encounter_is_flagged_as_an_upper_bound() -> None:
    result = hall3d(slow_crossing(), window=(-60.0, 60.0))
    assert any(
        note.startswith("long encounter") and "upper bound on Pc" in note
        for note in result.diagnostics
    )


def test_floored_combined_covariance_is_reported() -> None:
    # The combined covariance is 1e-12 m^2 across the plane: raised to
    # (1e-4 R)^2, as the 2-D methods do (and report).
    primary = object_state(LEO_R, LEO_V, np.diag([100.0, 1.0, 1e-12]))
    secondary = object_state(
        LEO_R + np.array([3.0, 0.0, 1.0]), -LEO_V, np.zeros((3, 3))
    )
    conjunction = Conjunction(primary, secondary, 5.0)
    result = hall3d(conjunction)
    assert result.diagnostics == (
        (
            "combined position covariance: smallest variance 1e-12 m^2 raised to "
            "2.5e-07 m^2"
        ),
    )
    two_d = foster(conjunction.plane, 5.0)
    assert two_d.diagnostics[0].endswith("raised to 2.5e-07 m^2")
    assert result.value == pytest.approx(two_d.value, rel=1e-3)


def test_floored_inverse_keeps_a_tiny_axis_next_to_a_huge_one() -> None:
    # 100 km along-track, and a perfectly known axis floored to 2 mm:
    # rebuilding the floored matrix and inverting it would bury the floor in
    # the rounding of the 1e10 m^2 eigenvalue (~1e-6 m^2).
    rotation = Rotation.from_euler("xyz", [0.3, -0.7, 1.1]).as_matrix()
    matrix = (rotation * np.array([0.0, 1e2, 1e10])) @ rotation.T
    inverse = overlap_module._floored_inverse(matrix, 4e-6)
    narrow = rotation[:, 0]
    assert narrow @ inverse @ narrow == pytest.approx(1.0 / 4e-6, rel=1e-6)


@pytest.mark.parametrize("closest", [1e6, -1e6, 1234.5])
def test_automatic_window_never_spans_more_than_one_period(
    monkeypatch: pytest.MonkeyPatch, closest: float
) -> None:
    # A wider window would count the next revolution's encounter again.
    captured: list[tuple[float, tuple[float, float]]] = []

    class Captured(Exception):
        pass

    def capture(search: WindowSearch) -> WindowSearch:
        captured.append((search.limit, search.centre))
        raise Captured

    straight_line_closest_at(monkeypatch, closest)
    window_search(monkeypatch, capture)
    conjunction = TYPICAL_HEAD_ON
    with pytest.raises(Captured):
        hall3d(conjunction)
    periods = [
        orbital_period(item.state)
        for item in (conjunction.primary, conjunction.secondary)
    ]
    period = min(p for p in periods if p is not None)
    [(limit, (start, end))] = captured
    assert start <= 0.0 <= end
    assert end - start + 2.0 * limit == pytest.approx(period)
