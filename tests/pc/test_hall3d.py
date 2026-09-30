import importlib
import math
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from builders import EPOCH, LEO_R, LEO_V, head_on, object_state, shifted_back
from scipy.spatial.transform import Rotation

from keepout.conjunction import Conjunction
from keepout.covariance import Covariance
from keepout.errors import CovarianceError, GeometryError, PcConvergenceError
from keepout.objects import ObjectState
from keepout.orbits import orbital_period
from keepout.pc import compute_pc, foster, hall3d
from keepout.pc._timeline import RateIntegral, Segment

# The package exports the function hall3d under the module's name.
hall3d_module = importlib.import_module("keepout.pc.hall3d")
timeline_module = importlib.import_module("keepout.pc._timeline")


def with_velocity_sigma(item: ObjectState, sigma: float) -> ObjectState:
    """The object with a 6x6 covariance: its position block plus sigma^2 I [m/s]."""
    assert item.covariance is not None
    matrix = item.covariance.matrix.copy()
    matrix[3:, 3:] = sigma**2 * np.eye(3)
    return ObjectState(
        item.object_id, item.state, Covariance(matrix, "GCRF", EPOCH), None
    )


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
    assert result.method == "hall"
    assert result.value == pytest.approx(
        foster(conjunction.plane, radius).value, rel=1e-4
    )
    assert result.details["encounters"] == 1.0
    assert result.details["fallbacks"] == 0.0
    assert result.diagnostics == ()


def test_velocity_uncertainty_is_accepted() -> None:
    base = head_on(30.0, 10.0, 20.0, 5.0)
    conjunction = Conjunction(
        with_velocity_sigma(base.primary, 0.1),
        with_velocity_sigma(base.secondary, 0.1),
        5.0,
    )
    two_d = foster(conjunction.plane, 5.0).value
    assert hall3d(conjunction).value == pytest.approx(two_d, rel=1e-3)


def test_swapping_the_objects_changes_nothing() -> None:
    base = head_on(30.0, 10.0, 20.0, 5.0)
    swapped = Conjunction(base.secondary, base.primary, 5.0)
    assert hall3d(swapped).value == pytest.approx(hall3d(base).value, rel=1e-8)


def test_explicit_window_is_used() -> None:
    conjunction = head_on(30.0, 10.0, 20.0, 5.0)
    result = hall3d(conjunction, window=(-0.002, 0.002))
    window = (result.details["window_start_s"], result.details["window_end_s"])
    assert window == (-0.002, 0.002)


@pytest.mark.parametrize("bad", [(1.0, -1.0), (0.0, math.inf)])
def test_invalid_window_is_rejected(bad: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="window"):
        hall3d(head_on(30.0, 10.0, 20.0, 5.0), window=bad)


def test_unknown_covariance_is_rejected() -> None:
    base = head_on(30.0, 10.0, 20.0, 5.0)
    conjunction = Conjunction(
        base.primary, ObjectState("S", base.secondary.state, None), 5.0
    )
    with pytest.raises(CovarianceError, match="hall needs both covariances"):
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
    monkeypatch.setattr(hall3d_module, "_peak_overlap", lambda *_: None)
    with pytest.raises(PcConvergenceError, match="did not converge"):
        hall3d(head_on(30.0, 10.0, 20.0, 5.0))


def test_several_encounters_and_nc_above_one_are_reported() -> None:
    integral = RateIntegral(
        nc=1.5,
        segments=(Segment(-10.0, -5.0, 0.9), Segment(5.0, 10.0, 0.6)),
        window=(-20.0, 20.0),
        edge=0.0,
        peak=1.0,
    )
    notes = hall3d_module._diagnostics(integral, window_given=True)
    assert notes[0].startswith("expected number of collisions Nc = 1.5 exceeds 1")
    assert notes[1].startswith("2 encounters in the window")


def test_rate_at_the_edge_of_an_automatic_window_is_reported() -> None:
    integral = RateIntegral(
        nc=0.1,
        segments=(Segment(-10.0, 10.0, 0.1),),
        window=(-10.0, 10.0),
        edge=1e-3,
        peak=1.0,
    )
    (note,) = hall3d_module._diagnostics(integral, window_given=False)
    assert "pass window=" in note
    assert hall3d_module._diagnostics(integral, window_given=True) == []


@pytest.mark.parametrize(
    "window", [(-1000.05, 999.97), (-300.03, 700.0), (-30.0, 30.0015)]
)
def test_long_explicit_window_resolves_the_encounter(
    window: tuple[float, float],
) -> None:
    # The crossing takes 0.4 ms: a uniform grid of at most 20001 points over
    # these windows steps over it; the screen must still find and resolve it.
    conjunction = head_on(30.0, 10.0, 20.0, 5.0)
    result = hall3d(conjunction, window=window)
    assert result.value == pytest.approx(foster(conjunction.plane, 5.0).value, rel=1e-4)
    assert result.details["encounters"] == 1.0


def test_unresolvable_window_raises() -> None:
    # Days of head-on passes, each 0.4 ms long: more scan points than allowed.
    with pytest.raises(
        PcConvergenceError, match="cannot resolve the collision rate"
    ) as info:
        hall3d(head_on(30.0, 10.0, 20.0, 5.0), window=(-1e5, 1e5))
    assert "'hall', window=(a, b)" in str(info.value)


def test_window_without_an_encounter_is_reported() -> None:
    result = hall3d(head_on(30.0, 10.0, 20.0, 5.0), window=(100.0, 101.0))
    assert result.value == 0.0
    (note,) = result.diagnostics
    assert note.startswith("collision rate negligible over the whole window")


Overlap = Callable[..., object]


def failing_where(
    when: Callable[[float], bool],
    failure: Callable[[Overlap, tuple[object, ...]], object],
) -> Overlap:
    """``_peak_overlap`` that fails through ``failure`` at the times ``when``."""
    real = hall3d_module._peak_overlap

    def patched(*args: object) -> object:
        t = args[2]
        assert isinstance(t, float)
        if when(t):
            return failure(real, args)
        return real(*args)

    return patched


def not_converging(real: Overlap, args: tuple[object, ...]) -> object:
    """The real iteration cut to one step: its genuine non-convergence return."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hall3d_module, "OVERLAP_MAX_ITERATIONS", 1)
        result = real(*args)
    assert result is None
    return result


def raising(real: Overlap, args: tuple[object, ...]) -> object:
    raise GeometryError("orbit is unbound")


@pytest.mark.parametrize("failure", [not_converging, raising])
def test_failure_inside_the_encounter_raises(
    monkeypatch: pytest.MonkeyPatch,
    failure: Callable[[Overlap, tuple[object, ...]], object],
) -> None:
    # The rate peaks near -0.25 ms; at +0.42 to +0.50 ms it is 1e-16 to 1e-10
    # of its peak: inside the encounter segment, but a fallback there stays
    # below the relative tolerance of FALLBACK_TOLERANCE.
    monkeypatch.setattr(
        hall3d_module,
        "_peak_overlap",
        failing_where(lambda t: 4.2e-4 <= t <= 5.0e-4, failure),
    )
    with pytest.raises(PcConvergenceError, match="did not converge"):
        hall3d(head_on(30.0, 10.0, 20.0, 5.0))


@pytest.mark.parametrize("failure", [not_converging, raising])
def test_failure_in_the_far_tail_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
    failure: Callable[[Overlap, tuple[object, ...]], object],
) -> None:
    conjunction = head_on(30.0, 10.0, 20.0, 5.0)
    monkeypatch.setattr(
        hall3d_module, "_peak_overlap", failing_where(lambda t: abs(t) >= 3e-3, failure)
    )
    result = hall3d(conjunction, window=(-0.02, 0.02))
    assert result.details["fallbacks"] > 0
    assert result.value == pytest.approx(foster(conjunction.plane, 5.0).value, rel=1e-4)


def test_failing_fallback_raises_a_convergence_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*_: object) -> float:
        raise np.linalg.LinAlgError("Singular matrix")

    monkeypatch.setattr(hall3d_module, "_peak_overlap", lambda *_: None)
    monkeypatch.setattr(hall3d_module.CollisionRate, "_rate", broken)
    with pytest.raises(PcConvergenceError, match="cannot be computed"):
        hall3d(head_on(30.0, 10.0, 20.0, 5.0))


def test_non_finite_rate_raises_a_convergence_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hall3d_module.CollisionRate, "_rate", lambda *_: math.nan)
    with pytest.raises(PcConvergenceError, match="not finite"):
        hall3d(head_on(30.0, 10.0, 20.0, 5.0))


@pytest.mark.parametrize("rtol", [0.0, -1e-6, math.nan, math.inf])
def test_rtol_is_checked(rtol: float) -> None:
    with pytest.raises(ValueError, match="rtol"):
        hall3d(head_on(30.0, 10.0, 20.0, 5.0), rtol=rtol)


def test_unmet_rtol_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    # Orders 4 and 8 cannot integrate a padded encounter to 1e-4.
    monkeypatch.setattr(timeline_module, "START_ORDER", 4)
    monkeypatch.setattr(timeline_module, "MAX_ORDER", 8)
    result = hall3d(head_on(30.0, 10.0, 20.0, 5.0), rtol=1e-15)
    (note,) = result.diagnostics
    assert re.match(
        r"the time integral did not reach rtol: its worst piece agrees to "
        r"\d\.\de[-+]\d+ of Nc \(rtol 1\.0e-15\)",
        note,
    )


def test_small_rtol_shortfall_is_not_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    # Orders 32 and 64 agree to far better than QUIET_AGREEMENT, not to 1e-15.
    monkeypatch.setattr(timeline_module, "MAX_ORDER", 64)
    result = hall3d(head_on(30.0, 10.0, 20.0, 5.0), rtol=1e-15)
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
    real = timeline_module.locate_auto

    def spy(rate: object, half: float, *args: object) -> object:
        widths.append(half)
        return real(rate, half, *args)

    monkeypatch.setattr(hall3d_module, "locate_auto", spy)
    result = hall3d(conjunction)
    initial = widths[0]
    assert result.details["window_end_s"] > initial  # the window doubled
    assert result.diagnostics == ()
    assert result.value == pytest.approx(foster(conjunction.plane, 5.0).value, rel=1e-4)


def slow_crossing() -> Conjunction:
    """A 0.2 m/s crossing with 5 m position and 0.1 m/s velocity sigmas."""

    def obj(r: np.ndarray, v: np.ndarray) -> ObjectState:
        matrix = np.zeros((6, 6))
        matrix[:3, :3] = 25.0 * np.eye(3)
        matrix[3:, 3:] = 0.01 * np.eye(3)
        state = object_state(r, v).state
        return ObjectState("X", state, Covariance(matrix, "GCRF", EPOCH), None)

    return Conjunction(
        obj(LEO_R, LEO_V),
        obj(LEO_R + np.array([30.0, 0.0, 0.0]), LEO_V + np.array([0.0, 0.0, 0.2])),
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
    conjunction = shifted_back(head_on(30.0, 10.0, 20.0, 5.0), dt)
    expected = foster(conjunction.plane, 5.0).value
    auto = compute_pc(conjunction)
    assert auto.method == "hall"
    assert auto.value == pytest.approx(expected, rel=1e-4)
    assert auto.details["window_end_s"] > dt


def test_an_encounter_beyond_the_initial_window_is_searched_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Centre the window on t = 0 only: the rate is 0 there, and the window
    # keeps doubling until it reaches the encounter 1 s later.
    real = timeline_module.locate_auto

    def at_tca(*args: Any) -> Any:
        return real(*args[:4], (0.0, 0.0))

    monkeypatch.setattr(hall3d_module, "locate_auto", at_tca)
    conjunction = shifted_back(head_on(30.0, 10.0, 20.0, 5.0), 1.0)
    assert hall3d(conjunction).value == pytest.approx(
        foster(conjunction.plane, 5.0).value, rel=1e-4
    )


def test_a_missed_straight_line_approach_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Forbid any growth of a window about t = 0: the scan then finds nothing,
    # although the straight-line closest approach is 30 m away at t = 1 s.
    real = timeline_module.locate_auto

    def frozen(rate: Any, half: float, limit: float, screen: Any, _: Any) -> Any:
        return real(rate, half, half, screen, (0.0, 0.0))

    monkeypatch.setattr(hall3d_module, "locate_auto", frozen)
    result = hall3d(shifted_back(head_on(30.0, 10.0, 20.0, 5.0), 1.0))
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
    inverse = hall3d_module._floored_inverse(matrix, 4e-6)
    narrow = rotation[:, 0]
    assert narrow @ inverse @ narrow == pytest.approx(1.0 / 4e-6, rel=1e-6)


@pytest.mark.parametrize("closest", [1e6, -1e6, 1234.5])
def test_automatic_window_never_spans_more_than_one_period(
    monkeypatch: pytest.MonkeyPatch, closest: float
) -> None:
    # A wider window would count the next revolution's encounter again.
    captured: dict[str, Any] = {}

    class Captured(Exception):
        pass

    def capture(
        rate: Any, half: float, limit: float, screen: Any, centre: tuple[float, float]
    ) -> Any:
        captured.update(limit=limit, centre=centre)
        raise Captured

    monkeypatch.setattr(hall3d_module, "straight_line_closest_time", lambda _: closest)
    monkeypatch.setattr(hall3d_module, "locate_auto", capture)
    conjunction = head_on(30.0, 10.0, 20.0, 5.0)
    rate = hall3d_module.collision_rate(conjunction)
    with pytest.raises(Captured):
        hall3d_module.encounter_scan(conjunction, rate, None)
    period = min(rate.primary.period, rate.secondary.period)
    start, end = captured["centre"]
    assert start <= 0.0 <= end
    assert end - start + 2.0 * captured["limit"] == pytest.approx(period)
