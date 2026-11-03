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
    ManoeuvredPropagator,
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


def manoeuvred(
    delta_v: float = 0.1,
    *,
    base: Propagator = LEO,
    burn_time: Time = START,
    horizon: float = 7 * DAY,
) -> ManoeuvredPropagator:
    return ManoeuvredPropagator(base, burn_time, delta_v, horizon)


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
    propagator = manoeuvred()
    assert isinstance(propagator, Propagator)
    assert propagator.frame == "TEME"
    assert len(propagator) == 1
    assert repr(propagator) == "ManoeuvredPropagator(1 objects)"


def test_states_before_the_burn_equal_base() -> None:
    burn_time = START + 2 * u.day
    propagator = manoeuvred(burn_time=burn_time)
    before = np.arange(-DAY, 2 * DAY, 311.0)
    assert_same_states(
        propagator.propagate(START, before), LEO.propagate(START, before)
    )


def test_the_burn_changes_the_velocity_at_its_time_only() -> None:
    delta_v = 0.25
    burn_time = START + DAY * u.s
    at_burn = manoeuvred(delta_v, burn_time=burn_time).propagate(burn_time, 0.0)
    base = LEO.propagate(burn_time, 0.0)
    assert np.array_equal(at_burn.r, base.r)
    change = (at_burn.v - base.v)[0, 0]
    assert np.linalg.norm(change) == pytest.approx(delta_v, abs=1e-9)
    # Along the velocity, whose GCRF direction is a hair from the TEME one.
    assert np.dot(change, base.v[0, 0]) / np.linalg.norm(base.v) == pytest.approx(
        delta_v, abs=1e-5
    )


def test_a_negative_delta_v_goes_against_the_velocity() -> None:
    at_burn = manoeuvred(-0.25).propagate(START, 0.0)
    base = LEO.propagate(START, 0.0)
    assert np.dot((at_burn.v - base.v)[0, 0], base.v[0, 0]) < 0.0


def test_zero_delta_v_equals_base_everywhere() -> None:
    propagator = manoeuvred(0.0, burn_time=START + DAY * u.s)
    assert_same_states(propagator.propagate(START, GRID), LEO.propagate(START, GRID))
    bounds, base = (
        propagator.radius_bounds(START, 7 * DAY),
        LEO.radius_bounds(START, 7 * DAY),
    )
    assert np.array_equal(bounds.lowest, base.lowest)
    assert np.array_equal(bounds.highest, base.highest)


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
        (68377, -0.1, 7 * DAY, 1.0),
        (68377, 1.0, DAY, 1.0),
        (HEO, 1.0, DAY, 1.0),
        (68377, 1.0, 7 * DAY, 1.0),
        # e > 0.3 at the top of the grid: the integrator and the 60 s spline at
        # perigee reach 2.24 m, the worst of the 38 such orbits of the fixture
        # (HEO_WORST), against a volume of kilometres.
        (HEO, 1.0, 7 * DAY, 3.0),
        (HEO_LOW, 1.0, 7 * DAY, 3.0),
        (HEO_WORST, 1.0, 7 * DAY, 3.0),
    ],
)
def test_difference_matches_cowell(
    number: int, delta_v: float, span: float, limit: float
) -> None:
    base = Sgp4Propagator([BY_NUMBER[number]])
    burn_time = START + HOUR * u.s
    propagator = manoeuvred(delta_v, base=base, burn_time=burn_time, horizon=span)
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
    propagator = manoeuvred(delta_v, base=base, burn_time=burn_time, horizon=DAY)
    assert propagator.frame == "GCRF"
    before = np.arange(0.0, HOUR, 97.0)
    assert_same_states(
        propagator.propagate(START, before), base.propagate(START, before)
    )
    middles = np.arange(30.0, DAY, 60.0)
    position, _ = difference_errors(propagator, base, burn_time, delta_v, middles)
    assert position.max() < 0.05


@pytest.mark.parametrize("delta_v", [1.0, -1.0, 0.01])
def test_radius_bounds_hold_the_states(delta_v: float) -> None:
    propagator = manoeuvred(delta_v)
    bounds = propagator.radius_bounds(START, 7 * DAY)
    radii = np.linalg.norm(propagator.propagate(START, GRID).r, axis=2)
    assert bounds.lowest[0] <= radii.min()
    assert radii.max() <= bounds.highest[0]
    base = LEO.radius_bounds(START, 7 * DAY)
    assert bounds.lowest[0] < base.lowest[0]
    assert base.highest[0] < bounds.highest[0]


def test_radius_bounds_widen_by_the_difference_of_their_span_only() -> None:
    delta_v = 0.1
    propagator = manoeuvred(delta_v, burn_time=START + DAY * u.s)
    before = propagator.radius_bounds(START, DAY / 2.0)
    base = LEO.radius_bounds(START, DAY / 2.0)
    assert np.array_equal(before.lowest, base.lowest)
    assert np.array_equal(before.highest, base.highest)
    short = propagator.radius_bounds(START, 2 * DAY)
    long = propagator.radius_bounds(START, 7 * DAY)
    near = LEO.radius_bounds(START, 2 * DAY)
    far = LEO.radius_bounds(START, 7 * DAY)
    assert short.highest[0] - near.highest[0] < long.highest[0] - far.highest[0]


def test_radius_bounds_hold_the_spline_between_its_nodes() -> None:
    # A step as long as a third of an orbit leaves peaks between the nodes.
    propagator = ManoeuvredPropagator(LEO, START, 1.0, DAY, step=1800.0)
    reference = LEO.propagate(START, np.arange(0.0, DAY, 7.0))
    states = propagator.propagate(START, reference.offsets)
    largest = np.linalg.norm(states.r - reference.r, axis=2).max()
    bounds, base = propagator.radius_bounds(START, DAY), LEO.radius_bounds(START, DAY)
    assert bounds.highest[0] - base.highest[0] >= largest


def test_lowest_bound_is_never_negative() -> None:
    # Half an orbit's worth of 2 km/s moves the object by more than its radius.
    bounds = manoeuvred(2000.0).radius_bounds(START, 7 * DAY)
    assert bounds.lowest[0] == 0.0
    assert bounds.highest[0] > LEO.radius_bounds(START, 7 * DAY).highest[0]


def test_errors_of_base_pass_through() -> None:
    base = Sgp4Propagator([BY_NUMBER[FAILING]])
    propagator = manoeuvred(0.1, base=base, horizon=2 * DAY)
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
        manoeuvred(0.1, base=base, burn_time=START + 2 * u.day, horizon=DAY)


def test_an_orbit_brought_down_by_the_burn_is_refused() -> None:
    with pytest.raises(PropagationError, match="fell below the Earth's surface"):
        manoeuvred(-3000.0)


def test_subset_and_chunks_give_identical_numbers() -> None:
    propagator = manoeuvred(0.1, burn_time=START + DAY * u.s)
    whole = propagator.propagate(START, GRID)
    same = propagator.subset(np.array([0]))
    assert isinstance(same, ManoeuvredPropagator)
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
    with pytest.raises(IndexError):
        manoeuvred().subset(np.array([1]))
    with pytest.raises(ValueError, match="integers"):
        manoeuvred().subset(np.array([0.5]))


def test_manoeuvred_propagator_pickles() -> None:
    propagator = manoeuvred(0.1, burn_time=START + DAY * u.s)
    copy = pickle.loads(pickle.dumps(propagator))
    assert_same_states(copy.propagate(START, GRID), propagator.propagate(START, GRID))
    bounds = propagator.radius_bounds(START, 7 * DAY)
    again = copy.radius_bounds(START, 7 * DAY)
    assert np.array_equal(again.highest, bounds.highest)
    zero = pickle.loads(pickle.dumps(manoeuvred(0.0)))
    assert_same_states(zero.propagate(START, GRID), LEO.propagate(START, GRID))


def test_offset_beyond_horizon_is_refused() -> None:
    propagator = manoeuvred(0.1, burn_time=START + HOUR * u.s, horizon=DAY)
    last = HOUR + DAY
    propagator.propagate(START, last)
    with pytest.raises(ValueError, match="after the horizon"):
        propagator.propagate(START, last + 1e-3)
    with pytest.raises(ValueError, match="after the horizon"):
        manoeuvred(0.0, horizon=DAY).propagate(START, DAY + 1.0)


def test_times_at_the_horizon_survive_rounding() -> None:
    # Times around this one land on either side of the end, as Time differences
    # round; one that rounds past it by less than a microsecond is the end.
    burn_time = Time("2026-10-06T07:13:21.123456", scale="utc")
    propagator = manoeuvred(0.1, burn_time=burn_time, horizon=DAY)
    for lead in (1.1, 3333.3, 12345.678):
        epoch = burn_time - lead * u.s
        states = propagator.propagate(epoch, lead + DAY)
        assert states.valid.all()
    end = propagator.propagate(burn_time, DAY)
    past = propagator.propagate(burn_time, DAY + 5e-7)
    base = LEO.propagate(burn_time, [DAY, DAY + 5e-7])
    assert np.array_equal((past.r - base.r[:, 1:]), (end.r - base.r[:, :1]))


def test_radius_bounds_of_a_span_past_the_horizon_are_refused() -> None:
    propagator = manoeuvred(0.1, horizon=DAY)
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
        ManoeuvredPropagator(**(arguments | change))  # type: ignore[arg-type]


def test_a_shorter_step_gives_the_same_difference() -> None:
    fine = ManoeuvredPropagator(LEO, START, 1.0, DAY, step=20.0)
    coarse = manoeuvred(1.0, horizon=DAY)
    offsets = np.arange(1.0, DAY, 317.0)
    gap = fine.propagate(START, offsets).r - coarse.propagate(START, offsets).r
    assert np.linalg.norm(gap, axis=2).max() < 0.05
