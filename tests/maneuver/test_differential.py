"""The effect of a burn at TCA, batched over a grid, and the new closest approach."""

import math
from collections.abc import Callable
from typing import Final

import numpy as np
import pytest
from astropy.time import Time
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from keepout._types import FloatArray
from keepout.errors import GeometryError, PropagationError
from keepout.maneuver import BurnEffects, burn_effects, closest_approach_shift
from keepout.orbits import EARTH_MU
from keepout.propagation.forces import EARTH_RADIUS, ForceModel
from keepout.state import RefFrame, StateVector
from keepout.units import HOUR, KM

EPOCH: Final = Time("2026-10-06T00:00:00", scale="utc")
POLE: Final = np.array([0.0, 0.0, 1.0])
TWO_BODY: Final = ForceModel(POLE, degree=0)
FULL: Final = ForceModel(POLE)
DRAG: Final = ForceModel(POLE, drag_term=0.02)
SEMI_MAJOR_AXIS: Final = EARTH_RADIUS + 500 * KM
PERIOD: Final = 2.0 * math.pi * math.sqrt(SEMI_MAJOR_AXIS**3 / EARTH_MU)
LIMIT: Final = PERIOD / 20.0
SEARCH: Final = 100.0
"""How far from a shift to look for the true minimum [s]."""


def circular(radius: float = SEMI_MAJOR_AXIS, frame: RefFrame = "GCRF") -> StateVector:
    """A circular orbit inclined by 0.9 rad, at its node."""
    speed = math.sqrt(EARTH_MU / radius)
    velocity = speed * np.array([0.0, math.cos(0.9), math.sin(0.9)])
    return StateVector(EPOCH, [radius, 0.0, 0.0], velocity, frame)


def across_a_band_edge() -> StateVector:
    """An ellipse whose altitude goes through the density table's 500 km edge."""
    radius = EARTH_RADIUS + 490 * KM
    speed = math.sqrt(EARTH_MU * (2.0 / radius - 1.0 / (radius + 15 * KM)))
    velocity = speed * np.array([0.0, math.cos(0.9), math.sin(0.9)])
    return StateVector(EPOCH, [radius, 0.0, 0.0], velocity, "GCRF")


def unit(vector: FloatArray) -> FloatArray:
    unit_vector: FloatArray = vector / np.linalg.norm(vector)
    return unit_vector


@pytest.mark.parametrize(
    "delta_v",
    [[0.0], [0.0, 0.0], [0.0, 0.5], [-0.5, 0.0], [-0.5, 0.0, 0.5], [-0.5, 0.5, 0.0]],
)
@pytest.mark.parametrize("force", [FULL, DRAG], ids=["gravity", "drag"])
def test_zero_delta_v_changes_nothing(delta_v: list[float], force: ForceModel) -> None:
    # Anywhere in the batch: equal states do not stay equal in one, as the BLAS
    # product in the solver's step rounds the last components of a vector
    # apart, about 1e-7 m here.
    leads = np.array([6.0, 3.0, 1.5]) * HOUR
    changes = np.array(delta_v)
    effects = burn_effects(circular(), leads, changes, force)
    # These assert the Δv = 0 mask; the tiny-burn test below carries the real check.
    assert np.all(effects.dr[:, changes == 0.0] == 0.0)
    assert np.all(effects.dv[:, changes == 0.0] == 0.0)
    assert np.all(np.linalg.norm(effects.dr[:, changes != 0.0], axis=2) > 1.0)


def test_a_tiny_burn_has_a_tiny_effect_at_every_lead() -> None:
    # A later row must start where the reference is, not where the integration
    # back from TCA put the burn: those are about 4 cm apart after 20 h.
    leads = np.array([20.0, 7.0, 1.5]) * HOUR
    delta_v = 1e-7
    effects = burn_effects(circular(), leads, np.array([delta_v]), FULL)
    separation = np.linalg.norm(effects.dr[:, 0], axis=1)
    assert np.all(separation < 2.0 * 3.0 * delta_v * leads)


@pytest.mark.parametrize("orbits", [1, 3, 15])
def test_separation_grows_as_three_dv_dt(orbits: int) -> None:
    lead = orbits * PERIOD
    delta_v = 0.01
    effects = burn_effects(circular(), np.array([lead]), np.array([delta_v]), TWO_BODY)
    assert np.linalg.norm(effects.dr[0, 0]) == pytest.approx(
        3.0 * delta_v * lead, rel=0.03
    )


def test_prograde_arrives_later_than_retrograde() -> None:
    state = circular()
    leads = np.array([3.0, 2.0, 1.0]) * PERIOD
    effects = burn_effects(state, leads, np.array([-0.1, 0.1]), FULL)
    along_velocity = effects.dr @ unit(state.v)
    assert np.all(along_velocity[:, 1] < 0.0)
    assert np.all(along_velocity[:, 0] > 0.0)


def test_effect_is_nearly_antisymmetric() -> None:
    effects = burn_effects(
        circular(), np.array([10.0 * PERIOD]), np.array([-0.01, 0.01]), FULL
    )
    retrograde, prograde = effects.dr[0]
    assert np.linalg.norm(prograde + retrograde) < 0.01 * np.linalg.norm(prograde)
    retrograde, prograde = effects.dv[0]
    assert np.linalg.norm(prograde + retrograde) < 0.01 * np.linalg.norm(prograde)


@pytest.mark.parametrize(
    ("state", "force"),
    [
        pytest.param(circular(), FULL, id="gravity"),
        pytest.param(
            across_a_band_edge(),
            DRAG,
            id="drag across a band edge",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_grid_matches_one_burn_at_a_time(state: StateVector, force: ForceModel) -> None:
    leads = np.array([25.0, 11.0, 6.0]) * HOUR
    delta_v = np.array([-1.0, -0.01, 0.1, 1.0])
    grid = burn_effects(state, leads, delta_v, force)
    np.testing.assert_array_equal(grid.burn_leads, leads)
    for row in range(len(leads)):
        for column, change in enumerate(delta_v):
            alone = burn_effects(state, leads[[row]], np.array([change]), force)
            assert grid.dr[row, column] == pytest.approx(alone.dr[0, 0], abs=1.0)
            assert grid.dv[row, column] == pytest.approx(alone.dv[0, 0], abs=1e-3)


def test_a_burn_at_tca_changes_the_velocity_only() -> None:
    state = circular()
    effects = burn_effects(state, np.array([HOUR, 0.0]), np.array([0.25]), FULL)
    assert np.all(effects.dr[1] == 0.0)
    assert effects.dv[1, 0] == pytest.approx(0.25 * unit(state.v), abs=1e-6)


def test_the_axes_and_shapes_are_kept() -> None:
    leads = np.array([3.0, 2.0, 1.0]) * HOUR
    delta_v = np.array([-0.1, 0.1, 0.2, 0.3])
    effects = burn_effects(circular(), leads, delta_v, TWO_BODY)
    assert isinstance(effects, BurnEffects)
    assert effects.dr.shape == effects.dv.shape == (3, 4, 3)
    np.testing.assert_array_equal(effects.burn_leads, leads)
    np.testing.assert_array_equal(effects.delta_v, delta_v)


def test_the_arrays_are_read_only_copies() -> None:
    leads = np.array([2.0, 1.0]) * HOUR
    delta_v = np.array([0.1])
    effects = burn_effects(circular(), leads, delta_v, TWO_BODY)
    leads[0] = 0.0
    assert effects.burn_leads[0] == 2.0 * HOUR
    for array in (effects.burn_leads, effects.delta_v, effects.dr, effects.dv):
        assert not array.flags.writeable


@pytest.mark.parametrize(
    "dr_shape",
    [(2, 1, 2), (1, 1, 3), (2, 2, 3)],
)
def test_burn_effects_shapes_must_match_the_axes(dr_shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=r"dr and dv"):
        BurnEffects(
            np.array([2.0, 1.0]),
            np.array([0.1]),
            np.zeros(dr_shape),
            np.zeros((2, 1, 3)),
        )


@pytest.mark.parametrize(
    "leads",
    [
        [],
        [[3600.0]],
        [3600.0, 3600.0],
        [3600.0, 7200.0],
        [3600.0, -1.0],
        [math.nan],
        [math.inf],
    ],
)
def test_burn_leads_must_be_descending_and_not_after_tca(leads: list[float]) -> None:
    with pytest.raises(ValueError, match="burn_leads"):
        burn_effects(circular(), np.array(leads), np.array([0.1]), TWO_BODY)


@pytest.mark.parametrize("delta_v", [[], [[0.1]], [math.nan], [math.inf]])
def test_delta_v_must_be_finite_numbers(delta_v: list[float]) -> None:
    with pytest.raises(ValueError, match="delta_v"):
        burn_effects(circular(), np.array([HOUR]), np.array(delta_v), TWO_BODY)


def test_only_a_gcrf_state_can_burn() -> None:
    with pytest.raises(GeometryError, match="GCRF"):
        burn_effects(
            circular(frame="ITRF"), np.array([HOUR]), np.array([0.1]), TWO_BODY
        )


def test_a_burn_into_the_earth_is_an_error() -> None:
    with pytest.raises(PropagationError, match="surface"):
        burn_effects(
            circular(EARTH_RADIUS + 100 * KM),
            np.array([HOUR]),
            np.array([-1_000.0]),
            TWO_BODY,
        )


def test_shift_of_a_straight_line() -> None:
    rng = np.random.default_rng(1)
    r = rng.normal(size=(4, 5, 3)) * 1_000.0
    v = rng.normal(size=(4, 5, 3)) * 100.0
    shift = closest_approach_shift(r, v, np.zeros_like(r), limit=1e6)
    assert shift.shape == (4, 5)
    expected = -np.sum(r * v, axis=-1) / np.sum(v * v, axis=-1)
    assert shift == pytest.approx(expected, rel=1e-12)


def test_shift_finds_the_minimum_of_a_curved_path() -> None:
    r = np.array([500.0, 200.0, 30.0])
    v = np.array([-100.0, 3.0, 1.0])
    a = np.array([0.5, -0.2, 0.01])
    time = closest_approach_shift(r, v, a, limit=LIMIT)
    best = minimize_scalar(
        lambda t: float(np.linalg.norm(r + v * t + 0.5 * a * t**2)),
        bounds=(0.0, 20.0),
        method="bounded",
        options={"xatol": 1e-9},
    )
    assert time.shape == ()
    assert float(time) == pytest.approx(best.x, abs=1e-5)


def test_shift_of_objects_at_rest_is_nan() -> None:
    zero = np.zeros(3)
    resting = np.array([100.0, 0.0, 0.0])
    assert np.isnan(closest_approach_shift(resting, zero, zero, limit=LIMIT))
    assert np.isnan(closest_approach_shift(zero, zero, zero, limit=LIMIT))


def test_shift_from_a_concave_start_is_nan() -> None:
    # A Newton step from here climbs to the maximum, which is not a minimum.
    r = np.array([1_000.0, 0.0, 0.0])
    v = np.array([10.0, 20.0, 0.0])
    a = np.array([-1.0, 0.0, 0.0])
    assert np.sum(v * v) + np.sum(r * a) < 0.0
    assert np.isnan(closest_approach_shift(r, v, a, limit=LIMIT))


def test_shift_outside_the_limit_is_nan() -> None:
    r = np.array([[1_000.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    v = np.array([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    shift = closest_approach_shift(r, v, np.zeros_like(r), limit=500.0)
    assert np.isnan(shift[0])
    assert shift[1] == pytest.approx(100.0)


def test_shift_that_has_not_converged_is_nan() -> None:
    # A triple root at t = 10 s: Newton gains only a third per iteration.
    r = np.array([100.0, 0.0, 0.0])
    v = np.array([-20.0, 0.0, 0.0])
    a = np.array([2.0, 0.0, 0.0])
    assert np.isnan(closest_approach_shift(r, v, a, limit=LIMIT))


def test_shift_of_nan_is_nan() -> None:
    r = np.array([math.nan, 0.0, 0.0])
    v = np.array([1.0, 0.0, 0.0])
    assert np.isnan(closest_approach_shift(r, v, np.zeros(3), limit=LIMIT))


def test_finite_shifts_are_minima_within_the_limit() -> None:
    rng = np.random.default_rng(7)
    count = 200_000

    def vectors(lowest: float, highest: float) -> FloatArray:
        sizes = 10.0 ** rng.uniform(lowest, highest, (count, 1))
        directions = rng.normal(size=(count, 3))
        scaled: FloatArray = (
            sizes * directions / np.linalg.norm(directions, axis=1, keepdims=True)
        )
        return scaled

    r, v, a = vectors(0.0, 6.0), vectors(-2.0, 4.0), vectors(-6.0, 1.0)
    shift = closest_approach_shift(r, v, a, limit=LIMIT)
    found = np.isfinite(shift)
    assert 0 < np.count_nonzero(found) < count

    r, v, a, t = r[found], v[found], a[found], shift[found][:, None]
    position = r + v * t + 0.5 * a * t**2
    rate = v + a * t
    curvature = np.sum(rate * rate + position * a, axis=1)
    slope = np.sum(position * rate, axis=1)
    assert np.all(np.abs(t) <= LIMIT)
    assert np.all(curvature > 0.0)
    assert np.all(np.abs(slope) <= 1e-6 * curvature)
    assert np.all(
        np.linalg.norm(position, axis=1) <= np.linalg.norm(r, axis=1) * (1.0 + 1e-12)
    )


@pytest.mark.parametrize("limit", [0.0, -1.0, math.nan, math.inf])
def test_limit_must_be_positive(limit: float) -> None:
    with pytest.raises(ValueError, match="limit"):
        closest_approach_shift(np.ones(3), np.ones(3), np.ones(3), limit=limit)


@pytest.mark.parametrize(
    ("r", "v"),
    [
        (np.ones(2), np.ones(2)),
        (np.ones(4), np.ones(3)),
        (np.ones((2, 3)), np.ones((3, 3))),
    ],
)
def test_shift_needs_vectors_that_broadcast(r: FloatArray, v: FloatArray) -> None:
    with pytest.raises(ValueError, match="shape"):
        closest_approach_shift(r, v, np.ones(3), limit=LIMIT)


def _motion(
    force: ForceModel, state: FloatArray, span: tuple[float, float]
) -> Callable[[float], FloatArray]:
    """A dense solve_ivp of one state, far tighter than the code under test."""
    solution = solve_ivp(
        lambda _, y: force.derivative(y),
        span,
        state,
        method="DOP853",
        rtol=1e-13,
        atol=np.repeat((1e-6, 1e-9), 3),
        dense_output=True,
    )
    assert solution.success
    assert solution.sol is not None
    return solution.sol


def _burned_motion(
    primary: StateVector, lead: float, delta_v: float, end: float
) -> Callable[[float], FloatArray]:
    """The primary's true path after a burn, up to ``end`` after TCA."""
    start = np.concatenate((primary.r, primary.v))
    burn = _motion(FULL, start, (0.0, -lead))(-lead)
    burn[3:] += delta_v * unit(burn[3:])
    return _motion(FULL, burn, (-lead, end))


def _true_range(
    burned: Callable[[float], FloatArray],
    secondary: FloatArray,
    window: float,
) -> Callable[[float], float]:
    """The true range to a secondary given at TCA, within ``window`` of TCA."""
    after = _motion(FULL, secondary, (0.0, window))
    before = _motion(FULL, secondary, (0.0, -window))

    def true_range(t: float) -> float:
        there = after(t) if t >= 0.0 else before(t)
        return float(np.linalg.norm(there[:3] - burned(t)[:3]))

    return true_range


def _relative(
    moved: FloatArray, secondary: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Relative position, velocity and acceleration of two states ``(6,)``."""
    states = np.stack((secondary, moved))
    accelerations = FULL.acceleration(states[:, :3], states[:, 3:])
    r, v = secondary[:3] - moved[:3], secondary[3:] - moved[3:]
    return r, v, accelerations[0] - accelerations[1]


@pytest.mark.parametrize(
    ("offset", "closing"),
    [
        (np.array([0.0, 0.0, 200.0]), np.array([0.0, -9_000.0, 6_000.0])),
        (np.array([300.0, 0.0, 0.0]), np.array([0.0, 150.0, -90.0])),
    ],
    ids=["fast", "slow"],
)
def test_shift_agrees_with_propagation(offset: FloatArray, closing: FloatArray) -> None:
    primary = circular()
    lead, delta_v = 3.0 * HOUR, 0.1
    effects = burn_effects(primary, np.array([lead]), np.array([delta_v]), FULL)
    moved = np.concatenate((primary.r + effects.dr[0, 0], primary.v + effects.dv[0, 0]))
    secondary = np.concatenate((primary.r + offset, primary.v + closing))
    r, v, a = _relative(moved, secondary)
    shift = float(closest_approach_shift(r, v, a, limit=LIMIT))
    assert math.isfinite(shift)
    predicted = float(np.linalg.norm(r + v * shift + 0.5 * a * shift**2))

    window = 2.0 * abs(shift) + 60.0
    burned = _burned_motion(primary, lead, delta_v, window)
    truth = minimize_scalar(
        _true_range(burned, secondary, window),
        bounds=(shift - 5.0, shift + 5.0),
        method="bounded",
        options={"xatol": 1e-7},
    )
    assert predicted == pytest.approx(truth.fun, abs=1.0)


def test_slow_encounter_gives_nan_not_a_false_minimum() -> None:
    primary = circular()
    lead, delta_v = 24.0 * HOUR, 0.01
    effects = burn_effects(primary, np.array([lead]), np.array([delta_v]), FULL)
    moved = np.concatenate((primary.r + effects.dr[0, 0], primary.v + effects.dv[0, 0]))
    span = LIMIT + 2.0 * SEARCH
    burned = _burned_motion(primary, lead, delta_v, span)

    rng = np.random.default_rng(3)
    shifts = []
    for _ in range(40):
        closing = rng.normal(size=3)
        closing *= 3.0 / np.linalg.norm(closing)
        miss = rng.normal(size=3)
        miss -= miss @ closing / (closing @ closing) * closing
        miss *= rng.uniform(100.0, 2_000.0) / np.linalg.norm(miss)
        secondary = np.concatenate((primary.r + miss, primary.v + closing))
        r, v, a = _relative(moved, secondary)
        shift = float(closest_approach_shift(r, v, a, limit=LIMIT))
        shifts.append(shift)
        if math.isnan(shift):
            continue
        # A true minimum, against propagation that shares nothing with the
        # code. Its range is what a Pc needs; a shallow one can lie far in time.
        assert abs(shift) <= LIMIT
        truth = minimize_scalar(
            _true_range(burned, secondary, span),
            bounds=(shift - SEARCH, shift + SEARCH),
            method="bounded",
            options={"xatol": 1e-4},
        )
        assert abs(truth.x - shift) < 0.9 * SEARCH
        predicted = float(np.linalg.norm(r + v * shift + 0.5 * a * shift**2))
        assert predicted == pytest.approx(truth.fun, abs=10.0)
    assert 0 < sum(math.isnan(shift) for shift in shifts) < len(shifts)
