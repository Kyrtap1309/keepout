"""An orbit after a burn: SGP4 plus the difference of two Cowell orbits."""

import pickle
from typing import Final

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time
from screening_data import FIXTURE_START, fixture_by_number

from keepout._types import FloatArray
from keepout.errors import PropagationError
from keepout.frames import frame_to_gcrf, teme_to_gcrf
from keepout.propagation import (
    CowellPropagator,
    Ephemeris,
    ForceModel,
    ManeuveredPropagator,
    Propagator,
    Sgp4Propagator,
    earth_pole,
    integrate,
    propagate_in_chunks,
)
from keepout.units import DAY, HOUR

START: Final = FIXTURE_START
BY_NUMBER: Final = fixture_by_number()
LEO: Final = Sgp4Propagator([BY_NUMBER[68377]])
GRID: Final = np.arange(0.0, 7 * DAY, 317.0)
"""Offsets from :data:`START` that miss the 60 s nodes."""
HEO: Final = 30580
"""An orbit of eccentricity 0.83."""
HEO_LOW: Final = 30797
"""An orbit of eccentricity 0.83, with its perigee 1400 km up."""
HEO_WORST: Final = 270358
"""An orbit of eccentricity 0.74, the worst of the fixture's for a difference."""
FAILING: Final = 45711
"""SGP4 fails for it 23 h 12 min after :data:`START`."""


def maneuvered(
    delta_v: float = 0.1,
    *,
    base: Propagator = LEO,
    burn_time: Time = START,
    horizon: float = 7 * DAY,
) -> ManeuveredPropagator:
    return ManeuveredPropagator(base, burn_time, delta_v, horizon)


def assert_same_states(left: Ephemeris, right: Ephemeris) -> None:
    assert np.array_equal(left.r, right.r, equal_nan=True)
    assert np.array_equal(left.v, right.v, equal_nan=True)
    assert np.array_equal(left.errors, right.errors)


def cowell_difference(
    base: Propagator, burn_time: Time, delta_v: float, offsets: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Burned minus nominal in GCRF at ``offsets`` from the burn, each orbit in
    an integration of its own. Far tighter than the defaults, which on the
    eccentric orbits are themselves 0.64 m and 4.2 m off after a week of 1 m/s."""
    at_burn = base.propagate(burn_time, 0.0)
    r, v = frame_to_gcrf(base.frame, burn_time, at_burn.r[0, 0], at_burn.v[0, 0])
    nominal = np.concatenate((r, v))
    burned = nominal.copy()
    burned[3:] += delta_v * v / np.linalg.norm(v)
    force = ForceModel(earth_pole(burn_time))
    rows = [
        integrate(
            force,
            state[None],
            offsets,
            rtol=3e-14,
            atol_position=1e-7,
            atol_velocity=1e-10,
        )[:, 0]
        for state in (nominal, burned)
    ]
    difference = rows[1] - rows[0]
    return difference[:, :3], difference[:, 3:]


def test_is_a_propagator_in_the_frame_of_its_base() -> None:
    propagator = maneuvered(horizon=HOUR)
    assert isinstance(propagator, Propagator)
    assert propagator.frame == "TEME"
    assert len(propagator) == 1
    assert repr(propagator) == "ManeuveredPropagator(1 objects)"


def test_states_before_the_burn_equal_base() -> None:
    burn_time = START + 2 * u.day
    propagator = maneuvered(burn_time=burn_time)
    before = np.arange(-DAY, 2 * DAY, 311.0)
    assert_same_states(
        propagator.propagate(START, before), LEO.propagate(START, before)
    )


@pytest.mark.parametrize("delta_v", [0.25, -0.25])
def test_the_burn_changes_the_velocity_at_its_time_only(delta_v: float) -> None:
    burn_time = START + DAY * u.s
    at_burn = maneuvered(delta_v, burn_time=burn_time, horizon=HOUR).propagate(
        burn_time, 0.0
    )
    base = LEO.propagate(burn_time, 0.0)
    assert np.array_equal(at_burn.r, base.r)
    change = (at_burn.v - base.v)[0, 0]
    assert np.linalg.norm(change) == pytest.approx(abs(delta_v), abs=1e-9)
    # Along the velocity, whose GCRF direction is a hair from the TEME one.
    assert np.dot(change, base.v[0, 0]) / np.linalg.norm(base.v) == pytest.approx(
        delta_v, abs=1e-5
    )


def test_zero_delta_v_equals_base_everywhere() -> None:
    propagator = maneuvered(0.0, burn_time=START + DAY * u.s)
    assert_same_states(propagator.propagate(START, GRID), LEO.propagate(START, GRID))
    bounds, base = (
        propagator.radius_bounds(START, 7 * DAY),
        LEO.radius_bounds(START, 7 * DAY),
    )
    assert np.array_equal(bounds.lowest, base.lowest)
    assert np.array_equal(bounds.highest, base.highest)


def test_a_tiny_burn_is_no_zero_burn_and_has_no_step() -> None:
    # 1 um/s takes the spline path, not the zero burn's shortcut: the base
    # before the burn, and after it a difference growing as dv t (the orbit
    # bends it by under 5 % in 10 min), from nothing at the burn.
    delta_v = 1e-6
    burn_time = START + DAY * u.s
    propagator = maneuvered(delta_v, burn_time=burn_time, horizon=HOUR)
    before = np.arange(-600.0, 0.0, 7.0)
    assert_same_states(
        propagator.propagate(burn_time, before), LEO.propagate(burn_time, before)
    )
    after = np.arange(1.0, 600.0, 7.0)
    moved, base = (
        propagator.propagate(burn_time, after),
        LEO.propagate(burn_time, after),
    )
    drift = np.linalg.norm(moved.r - base.r, axis=2)[0] / (delta_v * after)
    assert np.all((drift > 0.95) & (drift < 1.0 + 1e-3))
    kick = np.linalg.norm(moved.v - base.v, axis=2)[0] / delta_v
    assert np.all((kick > 0.9) & (kick < 1.0 + 1e-3))


def difference_errors(
    propagator: Propagator,
    base: Propagator,
    burn_time: Time,
    delta_v: float,
    offsets: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """How far ``propagator`` is from ``base`` plus the difference of two
    orbits integrated apart, position [m] and velocity [m/s], at each of
    ``offsets`` from the burn."""
    r_expected, v_expected = cowell_difference(base, burn_time, delta_v, offsets)
    after, nominal = (p.propagate(burn_time, offsets) for p in (propagator, base))
    times = burn_time + offsets * u.s
    r_actual, _ = frame_to_gcrf(
        base.frame, times, (after.r - nominal.r)[0], np.zeros(3)
    )
    v_actual, _ = frame_to_gcrf(
        base.frame, times, (after.v - nominal.v)[0], np.zeros(3)
    )
    return (
        np.linalg.norm(r_actual - r_expected, axis=1),
        np.linalg.norm(v_actual - v_expected, axis=1),
    )


@pytest.mark.parametrize(
    ("number", "delta_v", "span", "limit"),
    [
        pytest.param(68377, -0.1, 7 * DAY, 1.0, marks=pytest.mark.slow),
        (68377, 1.0, DAY, 1.0),
        (HEO, 1.0, DAY, 1.0),
        pytest.param(68377, 1.0, 7 * DAY, 1.0, marks=pytest.mark.slow),
        # e > 0.3 at the top of the grid: the integrator and the 60 s spline at
        # perigee reach 2.24 m, the worst of the 38 such orbits of the fixture
        # (HEO_WORST), against a volume of kilometres. 11 to 16 s each; the
        # day of HEO above stands for them in a quick run.
        pytest.param(HEO, 1.0, 7 * DAY, 3.0, marks=pytest.mark.slow),
        pytest.param(HEO_LOW, 1.0, 7 * DAY, 3.0, marks=pytest.mark.slow),
        pytest.param(HEO_WORST, 1.0, 7 * DAY, 3.0, marks=pytest.mark.slow),
    ],
)
def test_difference_matches_cowell(
    number: int, delta_v: float, span: float, limit: float
) -> None:
    base = Sgp4Propagator([BY_NUMBER[number]])
    burn_time = START + HOUR * u.s
    propagator = maneuvered(delta_v, base=base, burn_time=burn_time, horizon=span)
    middles = np.arange(30.0, span, 60.0)
    position, velocity = difference_errors(
        propagator, base, burn_time, delta_v, middles
    )
    assert position.max() < limit
    # The velocity is the spline's slope: 5 mm/s off at the perigee of the
    # eccentric orbit, between nodes.
    assert velocity.max() < 1e-2


def test_a_base_in_gcrf_is_not_rotated() -> None:
    at_start = LEO.propagate(START, 0.0)
    r, v = teme_to_gcrf(START, at_start.r[0, 0], at_start.v[0, 0])
    force = ForceModel(earth_pole(START))
    state = np.concatenate((r, v))[None]
    base = CowellPropagator(START, state, [force], (0.0, 2 * DAY))
    burn_time, delta_v = START + HOUR * u.s, 1.0
    propagator = maneuvered(delta_v, base=base, burn_time=burn_time, horizon=DAY)
    assert propagator.frame == "GCRF"
    before = np.arange(0.0, HOUR, 97.0)
    assert_same_states(
        propagator.propagate(START, before), base.propagate(START, before)
    )
    middles = np.arange(30.0, DAY, 60.0)
    position, _ = difference_errors(propagator, base, burn_time, delta_v, middles)
    assert position.max() < 0.05


def dense_radii(propagator: Propagator, duration: float) -> FloatArray:
    """Radii every 2 s: a bound holds them, and is within metres of them."""
    offsets = np.arange(0.0, duration + 1.0, 2.0)
    positions = propagator.propagate(START, offsets).r[0]
    radii: FloatArray = np.linalg.norm(positions, axis=1)
    return radii


@pytest.mark.parametrize(
    ("number", "delta_v"),
    [(68377, 1.0), (68377, -1.0), (68377, 0.01), (HEO_WORST, 1.0)],
)
def test_radius_bounds_hold_the_states_closely(number: int, delta_v: float) -> None:
    base = Sgp4Propagator([BY_NUMBER[number]])
    propagator = maneuvered(delta_v, base=base)
    bounds = propagator.radius_bounds(START, 7 * DAY)
    radii = dense_radii(propagator, 7 * DAY)
    assert bounds.lowest[0] <= radii.min()
    assert radii.max() <= bounds.highest[0]
    # Not |d| wide: most of a difference lies along the track (1800 km at
    # 1 m/s in a week on the LEO orbit).
    assert radii.min() - bounds.lowest[0] < 500.0
    assert bounds.highest[0] - radii.max() < 500.0


def test_radius_bounds_follow_their_span() -> None:
    propagator = maneuvered(0.1, burn_time=START + DAY * u.s)
    before = propagator.radius_bounds(START, DAY / 2.0)
    base = LEO.radius_bounds(START, DAY / 2.0)
    assert np.array_equal(before.lowest, base.lowest)
    assert np.array_equal(before.highest, base.highest)
    short = propagator.radius_bounds(START, 2 * DAY)
    long = propagator.radius_bounds(START, 7 * DAY)
    assert long.lowest[0] <= short.lowest[0] < short.highest[0] <= long.highest[0]


def test_radius_bounds_hold_the_spline_between_its_nodes() -> None:
    # A step as long as a third of an orbit leaves peaks between the nodes.
    propagator = ManeuveredPropagator(LEO, START, 1.0, DAY, step=1800.0)
    bounds = propagator.radius_bounds(START, DAY)
    radii = dense_radii(propagator, DAY)
    assert bounds.lowest[0] <= radii.min()
    assert radii.max() <= bounds.highest[0]


def test_radius_bounds_hold_a_burn_that_reshapes_the_orbit() -> None:
    bounds = maneuvered(2000.0).radius_bounds(START, 7 * DAY)
    radii = dense_radii(maneuvered(2000.0), 7 * DAY)
    assert 0.0 <= bounds.lowest[0] <= radii.min()
    assert radii.max() <= bounds.highest[0]
    assert bounds.highest[0] > LEO.radius_bounds(START, 7 * DAY).highest[0] + 1e6


def test_radius_bounds_fall_back_to_the_difference_where_the_base_fails() -> None:
    base = Sgp4Propagator([BY_NUMBER[FAILING]])
    propagator = maneuvered(1.0, base=base, horizon=2 * DAY)
    bounds = propagator.radius_bounds(START, 2 * DAY)
    widening = propagator._largest_difference(0.0, 2 * DAY)
    nominal = base.radius_bounds(START, 2 * DAY)
    assert bounds.lowest[0] == max(nominal.lowest[0] - widening, 0.0)
    assert bounds.highest[0] == nominal.highest[0] + widening


def test_errors_of_base_pass_through() -> None:
    base = Sgp4Propagator([BY_NUMBER[FAILING]])
    propagator = maneuvered(0.1, base=base, horizon=2 * DAY)
    offsets = np.arange(0.0, 2 * DAY, 600.0)
    states, expected = (
        propagator.propagate(START, offsets),
        base.propagate(START, offsets),
    )
    assert expected.errors.any()
    assert (~expected.valid).any()
    assert np.array_equal(states.errors, expected.errors)
    assert np.isnan(states.r[~states.valid]).all()
    assert np.isnan(states.v[~states.valid]).all()
    assert np.isfinite(states.r[states.valid]).all()
    kept = offsets[states.valid[0]]
    position, velocity = difference_errors(propagator, base, START, 0.1, kept)
    assert position.max() < 1.0
    assert velocity.max() < 1e-2


def test_a_failed_state_at_the_burn_is_refused() -> None:
    base = Sgp4Propagator([BY_NUMBER[FAILING]])
    with pytest.raises(PropagationError, match="no state to burn from"):
        maneuvered(0.1, base=base, burn_time=START + 2 * u.day, horizon=DAY)


def test_an_orbit_brought_down_by_the_burn_is_refused() -> None:
    with pytest.raises(PropagationError, match="fell below the Earth's surface"):
        maneuvered(-3000.0)


def test_subset_and_chunks_give_identical_numbers() -> None:
    propagator = maneuvered(0.1, burn_time=START + DAY * u.s)
    whole = propagator.propagate(START, GRID)
    same = propagator.subset(np.array([0]))
    assert isinstance(same, ManeuveredPropagator)
    assert_same_states(same.propagate(START, GRID), whole)
    doubled = propagator.subset(np.array([0, 0]))
    assert len(doubled) == 2
    twice = doubled.propagate(START, GRID)
    assert np.array_equal(twice.r[1:], whole.r)
    assert np.array_equal(twice.v[0:1], whole.v)
    empty = propagator.subset(np.array([], dtype=np.intp))
    assert empty.propagate(START, GRID).r.shape == (0, len(GRID), 3)
    assert len(empty.radius_bounds(START, DAY)) == 0
    chunks = list(propagate_in_chunks([propagator], START, GRID, max_states=97))
    assert len(chunks) > 1
    assert np.array_equal(np.concatenate([c[0].r for c in chunks], axis=1), whole.r)
    assert np.array_equal(np.concatenate([c[0].v for c in chunks], axis=1), whole.v)
    one_by_one = [propagator.propagate(START, t) for t in GRID[::97]]
    assert np.array_equal(
        np.concatenate([e.r for e in one_by_one], axis=1), whole.r[:, ::97]
    )


def test_subset_refuses_what_its_base_refuses() -> None:
    propagator = maneuvered(horizon=HOUR)
    with pytest.raises(IndexError):
        propagator.subset(np.array([1]))
    with pytest.raises(ValueError, match="integers"):
        propagator.subset(np.array([0.5]))


def test_maneuvered_propagator_pickles() -> None:
    propagator = maneuvered(0.1, burn_time=START + DAY * u.s)
    copy = pickle.loads(pickle.dumps(propagator))
    assert_same_states(copy.propagate(START, GRID), propagator.propagate(START, GRID))
    bounds = propagator.radius_bounds(START, 7 * DAY)
    again = copy.radius_bounds(START, 7 * DAY)
    assert np.array_equal(again.highest, bounds.highest)
    zero = pickle.loads(pickle.dumps(maneuvered(0.0)))
    assert_same_states(zero.propagate(START, GRID), LEO.propagate(START, GRID))


def test_offset_beyond_horizon_is_refused() -> None:
    propagator = maneuvered(0.1, burn_time=START + HOUR * u.s, horizon=DAY)
    last = HOUR + DAY
    propagator.propagate(START, last)
    with pytest.raises(ValueError, match="after the horizon"):
        propagator.propagate(START, last + 1e-3)
    with pytest.raises(ValueError, match="after the horizon"):
        maneuvered(0.0, horizon=DAY).propagate(START, DAY + 1.0)


def test_times_at_the_horizon_survive_rounding() -> None:
    # Times around this one land on either side of the end, as Time differences
    # round; one that rounds past it by less than a microsecond is the end.
    burn_time = Time("2026-10-06T07:13:21.123456", scale="utc")
    propagator = maneuvered(0.1, burn_time=burn_time, horizon=DAY)
    for lead in (1.1, 3333.3, 12345.678):
        epoch = burn_time - lead * u.s
        states = propagator.propagate(epoch, lead + DAY)
        assert states.valid.all()
    end = propagator.propagate(burn_time, DAY)
    past = propagator.propagate(burn_time, DAY + 5e-7)
    base = LEO.propagate(burn_time, [DAY, DAY + 5e-7])
    assert np.array_equal((past.r - base.r[:, 1:]), (end.r - base.r[:, :1]))


def test_radius_bounds_of_a_span_past_the_horizon_are_refused() -> None:
    propagator = maneuvered(0.1, horizon=DAY)
    propagator.radius_bounds(START, DAY)
    with pytest.raises(ValueError, match="after the horizon"):
        propagator.radius_bounds(START, DAY + 1.0)


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        ({"delta_v": float("nan")}, ValueError, "delta_v must be finite"),
        ({"delta_v": float("inf")}, ValueError, "delta_v must be finite"),
        ({"horizon": 0.0}, ValueError, "horizon must be finite and > 0"),
        ({"horizon": float("inf")}, ValueError, "horizon must be finite and > 0"),
        ({"step": 0.0}, ValueError, "step must be finite and > 0"),
        (
            {"burn_time": START + np.array([0.0, 1.0]) * u.s},
            ValueError,
            r"burn_time must be a single time",
        ),
        (
            {"base": Sgp4Propagator([BY_NUMBER[68377], BY_NUMBER[56156]])},
            ValueError,
            "base must hold one object, got 2",
        ),
    ],
)
def test_checks_its_arguments(
    change: dict[str, object], error: type[Exception], message: str
) -> None:
    arguments: dict[str, object] = {
        "base": LEO,
        "burn_time": START,
        "delta_v": 0.1,
        "horizon": DAY,
    }
    with pytest.raises(error, match=message):
        ManeuveredPropagator(**(arguments | change))  # type: ignore[arg-type]


def test_a_shorter_step_gives_the_same_difference() -> None:
    fine = ManeuveredPropagator(LEO, START, 1.0, DAY, step=20.0)
    coarse = maneuvered(1.0, horizon=DAY)
    offsets = np.arange(1.0, DAY, 317.0)
    gap = fine.propagate(START, offsets).r - coarse.propagate(START, offsets).r
    assert np.linalg.norm(gap, axis=2).max() < 0.05
