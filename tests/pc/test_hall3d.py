import importlib
import math
import re

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
)
from faults import (
    Failure,
    WindowSearch,
    failing_where,
    never_converges,
    not_converging,
    raising,
    searching_with,
)
from scipy.spatial.transform import Rotation

from keepout.conjunction import Conjunction
from keepout.errors import CovarianceError, GeometryError, PcConvergenceError
from keepout.objects import ObjectState
from keepout.orbits import orbital_period
from keepout.pc import _rate as rate_module
from keepout.pc import _timeline as timeline_module
from keepout.pc import foster, hall3d
from keepout.pc._timeline import RateIntegral, RateScan, Segment

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
    conjunction = Conjunction(
        base.primary, ObjectState("S", base.secondary.state, None), 5.0
    )
    with pytest.raises(CovarianceError, match="hall3d needs both covariances"):
        hall3d(conjunction)


def test_unbound_orbit_is_rejected() -> None:
    fast = np.array([0.0, 15000.0, 0.0])
    primary = object_state(LEO_R, fast, np.eye(3))
    secondary = object_state(LEO_R + np.array([30.0, 0.0, 0.0]), LEO_V, np.eye(3))
    with pytest.raises(GeometryError, match="unbound"):
        hall3d(Conjunction(primary, secondary, 5.0))


def test_zero_relative_velocity_is_supported() -> None:
    # Co-located objects 10 m apart on the same circular orbit never meet in
    # a straight line, but their uncertainties overlap: 3-D Pc is defined.
    primary = object_state(LEO_R, LEO_V, 25.0 * np.eye(3))
    secondary = object_state(
        LEO_R + np.array([0.0, 0.0, 10.0]), LEO_V, 25.0 * np.eye(3)
    )
    result = hall3d(Conjunction(primary, secondary, 5.0), window=(-60.0, 60.0))
    # Nc is the expected number of entries into the hard body. A brute-force
    # simulation (1e6 equinoctial samples, distance every 0.5 s) counted 165
    # entries: 1.65e-4 per trial, 99% Poisson interval [1.3e-4, 2.0e-4].
    # (First entries alone give Pc ~ 8e-5: at mm/s relative speeds grazing
    # samples enter repeatedly, so Nc is well above Pc here.)
    assert 1.3e-4 < result.value < 2.0e-4
    assert result.diagnostics == (
        "relative velocity is zero: Nc counts repeated entries into the "
        "hard-body sphere, so it is an upper bound on Pc",
    )


def test_failed_iteration_where_it_matters_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rate_module, "_iterate_overlap", never_converges)
    with pytest.raises(PcConvergenceError, match="did not converge"):
        hall3d(TYPICAL_HEAD_ON)


def scan_of(times: list[float], values: list[float]) -> RateScan:
    """A scan of one segment over the whole grid."""
    return RateScan(np.array(times), np.array(values), ((0, len(times) - 1),))


def test_several_encounters_and_nc_above_one_are_reported() -> None:
    integral = RateIntegral(
        nc=1.5,
        segments=(Segment(-10.0, -5.0, 0.9), Segment(5.0, 10.0, 0.6)),
        scan=scan_of([-20.0, 0.0, 20.0], [0.0, 1.0, 0.0]),
    )
    notes = hall3d_module._integral_notes(integral, window_given=True, rtol=1e-6)
    assert notes[0].startswith("expected number of collisions Nc = 1.5 exceeds 1")
    assert notes[1].startswith("2 encounters in the window")


def test_rate_at_the_edge_of_an_automatic_window_is_reported() -> None:
    integral = RateIntegral(
        nc=0.1,
        segments=(Segment(-10.0, 10.0, 0.1),),
        scan=scan_of([-10.0, 0.0, 10.0], [1e-3, 1.0, 0.0]),
    )
    (note,) = hall3d_module._integral_notes(integral, window_given=False, rtol=1e-6)
    assert "pass window=" in note
    assert hall3d_module._integral_notes(integral, window_given=True, rtol=1e-6) == []


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
    monkeypatch.setattr(
        rate_module,
        "_iterate_overlap",
        failing_where(lambda t: (t >= 4.2e-4) & (t <= 5.0e-4), failure),
    )
    with pytest.raises(PcConvergenceError, match="did not converge"):
        hall3d(TYPICAL_HEAD_ON)


@pytest.mark.parametrize("failure", [not_converging, raising])
def test_failure_in_the_far_tail_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
    failure: Failure,
) -> None:
    conjunction = TYPICAL_HEAD_ON
    monkeypatch.setattr(
        rate_module,
        "_iterate_overlap",
        failing_where(lambda t: np.abs(t) >= 3e-3, failure),
    )
    result = hall3d(conjunction, window=(-0.02, 0.02))
    assert result.details["fallbacks"] > 0
    assert result.value == pytest.approx(foster(conjunction.plane, 5.0).value, rel=1e-4)


def test_failing_fallback_raises_a_convergence_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*_: object) -> float:
        raise np.linalg.LinAlgError("Singular matrix")

    monkeypatch.setattr(rate_module, "_iterate_overlap", never_converges)
    monkeypatch.setattr(rate_module.CollisionRate, "_rate", broken)
    with pytest.raises(PcConvergenceError, match="cannot be computed"):
        hall3d(TYPICAL_HEAD_ON)


def test_non_finite_rate_raises_a_convergence_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rate_module.CollisionRate, "_rate", lambda *_: math.nan)
    with pytest.raises(PcConvergenceError, match="not finite"):
        hall3d(TYPICAL_HEAD_ON)


@pytest.mark.parametrize("rtol", [0.0, -1e-6, math.nan, math.inf])
def test_rtol_is_checked(rtol: float) -> None:
    with pytest.raises(ValueError, match="rtol"):
        hall3d(TYPICAL_HEAD_ON, rtol=rtol)


def test_unmet_rtol_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    # Orders 4 and 8 cannot integrate a padded encounter to 1e-4.
    monkeypatch.setattr(timeline_module, "CC_START_ORDER", 4)
    monkeypatch.setattr(timeline_module, "CC_MAX_ORDER", 8)
    result = hall3d(TYPICAL_HEAD_ON, rtol=1e-15)
    (note,) = result.diagnostics
    assert re.match(
        r"the time integral did not reach rtol: its worst piece agrees to "
        r"\d\.\de[-+]\d+ of Nc \(rtol 1\.0e-15\)",
        note,
    )


def test_small_rtol_shortfall_is_not_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    # Orders 32 and 64 agree to far better than QUIET_AGREEMENT, not to 1e-15.
    monkeypatch.setattr(timeline_module, "CC_MAX_ORDER", 64)
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

    monkeypatch.setattr(rate_module, "locate_auto", searching_with(record))
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

    monkeypatch.setattr(rate_module, "locate_auto", searching_with(at_tca))
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

    monkeypatch.setattr(rate_module, "locate_auto", searching_with(frozen))
    result = hall3d(shifted_back(TYPICAL_HEAD_ON, 1.0))
    assert result.value == 0.0
    assert result.diagnostics[0].startswith("collision rate negligible")
    assert result.diagnostics[1].startswith(
        "the straight-line closest approach (30 m, t = 1 s)"
    )


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
        "combined position covariance: smallest variance 1e-12 m^2 raised to "
        "2.5e-07 m^2",
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
    inverse = rate_module._floored_inverse(matrix, 4e-6)
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

    monkeypatch.setattr(rate_module, "straight_line_closest_time", lambda _: closest)
    monkeypatch.setattr(rate_module, "locate_auto", searching_with(capture))
    conjunction = TYPICAL_HEAD_ON
    rate = rate_module.collision_rate(conjunction, "hall3d")
    with pytest.raises(Captured):
        rate_module.encounter_scan(conjunction, rate, None)
    period = min(rate.primary.period, rate.secondary.period)
    [(limit, (start, end))] = captured
    assert start <= 0.0 <= end
    assert end - start + 2.0 * limit == pytest.approx(period)
