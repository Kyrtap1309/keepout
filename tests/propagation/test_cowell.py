"""Cowell integration and the propagator built on it."""

import math
import pickle
from typing import Any, Final

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time
from builders import EARTH_ROTATION_RATE, loaded_after
from screening_data import FIXTURE_START, fixture_by_number

from keepout._types import FloatArray
from keepout.errors import PropagationError
from keepout.frames import teme_to_gcrf
from keepout.orbits import EARTH_MU
from keepout.propagation import (
    DECAYED,
    ERROR_MESSAGES,
    NO_ERROR,
    Ephemeris,
    Propagator,
    Sgp4Propagator,
    propagate_in_chunks,
)
from keepout.propagation._atmosphere import density
from keepout.propagation.cowell import CowellPropagator, integrate
from keepout.propagation.forces import EARTH_RADIUS, ForceModel, earth_pole
from keepout.units import DAY, HOUR, KM

EPOCH: Final = Time("2026-10-06T00:00:00", scale="utc")
POLE: Final = np.array([0.0, 0.0, 1.0])
TWO_BODY: Final = ForceModel(POLE, degree=0)
J2_ONLY: Final = ForceModel(POLE, degree=2)
FULL: Final = ForceModel(POLE)
J2: Final = 0.484165143790815e-3 * math.sqrt(5.0)
"""J2 of EGM2008, from the normalised coefficient."""


def state_of(
    semi_major_axis: float,
    eccentricity: float = 0.0,
    inclination: float = 0.9,
    raan: float = 0.3,
    argument: float = 0.5,
    anomaly: float = 0.2,
) -> FloatArray:
    """A state ``(6,)`` of classical elements [m, rad]."""
    p = semi_major_axis * (1.0 - eccentricity**2)
    radius = p / (1.0 + eccentricity * math.cos(anomaly))
    speed = math.sqrt(EARTH_MU / p)
    in_plane_r = radius * np.array([math.cos(anomaly), math.sin(anomaly), 0.0])
    in_plane_v = speed * np.array(
        [-math.sin(anomaly), eccentricity + math.cos(anomaly), 0.0]
    )
    rotation = _rotation_z(raan) @ _rotation_x(inclination) @ _rotation_z(argument)
    return np.concatenate((rotation @ in_plane_r, rotation @ in_plane_v))


def _rotation_z(angle: float) -> FloatArray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotation_x(angle: float) -> FloatArray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


LEO: Final = state_of(EARTH_RADIUS + 500 * KM)
ELLIPSE: Final = state_of(EARTH_RADIUS + 1_000 * KM, 0.05)
REENTERING: Final = np.array([EARTH_RADIUS + 200 * KM, 0.0, 0.0, 0.0, 6_500.0, 2_000.0])
"""A state whose perigee is far below the surface, reached in under an hour."""


def energy(states: FloatArray) -> FloatArray:
    """Specific two-body energy of states ``(..., 6)``."""
    radius = np.linalg.norm(states[..., :3], axis=-1)
    speed = np.linalg.norm(states[..., 3:], axis=-1)
    energies: FloatArray = 0.5 * speed**2 - EARTH_MU / radius
    return energies


def elements(states: FloatArray) -> dict[str, FloatArray]:
    """Osculating elements of states ``(T, 6)``, angles unwrapped."""
    r, v = states[:, :3], states[:, 3:]
    h = np.cross(r, v)
    node = np.cross(POLE, h)
    radius = np.linalg.norm(r, axis=1, keepdims=True)
    e_vector = np.cross(v, h) / EARTH_MU - r / radius
    eccentricity = np.linalg.norm(e_vector, axis=1)
    semi_major_axis = -EARTH_MU / (2.0 * energy(states))
    inclination = np.arccos(h[:, 2] / np.linalg.norm(h, axis=1))
    in_plane_y = np.cross(h / np.linalg.norm(h, axis=1, keepdims=True), node)
    argument = np.arctan2(
        np.sum(e_vector * in_plane_y, axis=1) / np.linalg.norm(in_plane_y, axis=1),
        np.sum(e_vector * node, axis=1) / np.linalg.norm(node, axis=1),
    )
    return {
        "a": semi_major_axis,
        "e": eccentricity,
        "i": inclination,
        "raan": np.unwrap(np.arctan2(node[:, 1], node[:, 0])),
        "argument": np.unwrap(argument),
    }


def test_two_body_conserves_energy_and_returns_after_one_period() -> None:
    semi_major_axis = EARTH_RADIUS + 1_000 * KM
    period = 2.0 * math.pi * math.sqrt(semi_major_axis**3 / EARTH_MU)
    rows = integrate(TWO_BODY, ELLIPSE[None], np.array([period / 2.0, period]))
    assert rows.shape == (2, 1, 6)
    assert np.all(abs(energy(rows) / energy(ELLIPSE) - 1.0) < 1e-10)
    assert np.linalg.norm(rows[1, 0, :3] - ELLIPSE[:3]) < 1.0


def test_j2_secular_rates_match_analytical() -> None:
    """Rates from elements averaged over whole orbits, on a high orbit: the
    first-order rates are off by about 2.4e-3 (R / p)², the second order in
    J2, which is 7e-4 here and 2e-3 in LEO."""
    samples = 120
    semi_major_axis = EARTH_RADIUS + 5_000 * KM
    period = 2.0 * math.pi * math.sqrt(semi_major_axis**3 / EARTH_MU)
    state = state_of(semi_major_axis, 0.1, 0.6, anomaly=0.0)
    orbits = int(DAY / period)
    times = (np.arange(orbits * samples) + 1.0) * period / samples
    rows = integrate(J2_ONLY, state[None], times)[:, 0]
    osculating = elements(rows)
    a = float(np.mean(osculating["a"]))
    e = float(np.mean(osculating["e"]))
    i = float(np.mean(osculating["i"]))
    factor = math.sqrt(EARTH_MU / a**3) * J2 * (EARTH_RADIUS / (a * (1.0 - e**2))) ** 2
    expected = {
        "raan": -1.5 * factor * math.cos(i),
        "argument": 0.75 * factor * (5.0 * math.cos(i) ** 2 - 1.0),
    }
    for name, rate in expected.items():
        first = np.mean(osculating[name][:samples])
        last = np.mean(osculating[name][-samples:])
        assert (last - first) / ((orbits - 1) * period) == pytest.approx(rate, rel=1e-3)


def test_back_and_forth_round_trip() -> None:
    rows = integrate(FULL, LEO[None], np.array([25.0 * HOUR, 0.0]))
    assert np.linalg.norm(rows[1, 0, :3] - LEO[:3]) < 0.1


def test_stops_may_start_behind_the_direction_of_travel() -> None:
    """Burn leads come as increasing negative stops: back first, then forth."""
    stops = np.array([-2.0 * HOUR, -HOUR, 1_000.0])
    rows = integrate(FULL, LEO[None], stops)
    alone = integrate(FULL, LEO[None], np.array([-2.0 * HOUR]))
    forth = integrate(FULL, LEO[None], np.array([1_000.0]))
    assert np.linalg.norm(rows[0, 0, :3] - alone[0, 0, :3]) < 1e-3
    assert np.linalg.norm(rows[2, 0, :3] - forth[0, 0, :3]) < 1e-2


def test_a_stop_at_time_zero_is_the_given_state() -> None:
    rows = integrate(FULL, LEO[None], np.array([0.0, 60.0]))
    np.testing.assert_array_equal(rows[0, 0], LEO)


def test_batch_matches_single_integration() -> None:
    generator = np.random.default_rng(7)
    count = 64
    states = np.array(
        [
            state_of(
                EARTH_RADIUS + generator.uniform(400, 1_500) * KM,
                generator.uniform(0.0, 0.02),
                generator.uniform(0.1, 3.0),
                *generator.uniform(0.0, 2.0 * math.pi, 3),
            )
            for _ in range(count)
        ]
    )
    stops = np.array([12.0 * HOUR, 25.0 * HOUR])
    batch = integrate(FULL, states, stops)
    assert batch.shape == (2, count, 6)
    for index in range(0, count, 9):
        single = integrate(FULL, states[index : index + 1], stops)
        assert np.linalg.norm(batch[-1, index, :3] - single[-1, 0, :3]) < 1.0


def test_integrate_reports_reentry() -> None:
    with pytest.raises(PropagationError, match="below"):
        integrate(TWO_BODY, REENTERING[None], np.array([HOUR, 2.0 * HOUR]))


def test_integrate_reports_reentry_of_one_object_in_a_batch() -> None:
    with pytest.raises(PropagationError, match="below"):
        integrate(TWO_BODY, np.stack((LEO, REENTERING)), np.array([2.0 * HOUR]))


def test_integrate_reports_a_state_that_starts_below_the_surface() -> None:
    inside = LEO.copy()
    inside[:3] *= 0.5
    with pytest.raises(PropagationError, match="below"):
        integrate(TWO_BODY, inside[None], np.array([60.0]))


def test_stop_at_surface_leaves_nan_after_the_event() -> None:
    stops = np.linspace(60.0, 2.0 * HOUR, 120)
    rows = integrate(TWO_BODY, REENTERING[None], stops, stop_at_surface=True)
    finite = np.isfinite(rows[:, 0, 0])
    assert 0 < np.count_nonzero(finite) < len(stops)
    count = np.count_nonzero(finite)
    assert finite[:count].all()
    assert np.isnan(rows[count:]).all()
    assert np.all(np.linalg.norm(rows[:count, 0, :3], axis=1) >= EARTH_RADIUS)


def test_stop_at_surface_of_a_state_below_it_is_all_nan() -> None:
    inside = LEO.copy()
    inside[:3] *= 0.5
    rows = integrate(TWO_BODY, inside[None], np.array([60.0]), stop_at_surface=True)
    assert np.isnan(rows).all()


def test_stop_at_surface_with_a_reentry_before_the_first_stop() -> None:
    rows = integrate(
        TWO_BODY,
        REENTERING[None],
        np.array([3.0 * HOUR, 4.0 * HOUR]),
        stop_at_surface=True,
    )
    assert np.isnan(rows).all()


def test_stop_at_surface_takes_one_object() -> None:
    with pytest.raises(ValueError, match="one object"):
        integrate(
            TWO_BODY, np.stack((LEO, LEO)), np.array([60.0]), stop_at_surface=True
        )


class _FailingForce:
    """A growth that reaches infinity in finite time, where steps cannot go."""

    def derivative(self, state: FloatArray) -> FloatArray:
        return state**2


def test_integrator_failure_is_a_propagation_error() -> None:
    force: Any = _FailingForce()
    with pytest.raises(PropagationError, match="integration failed"):
        integrate(force, LEO[None], np.array([60.0]))


@pytest.mark.parametrize(
    ("states", "stops", "message"),
    [
        (LEO, [60.0], "states"),
        (np.zeros((1, 5)), [60.0], "states"),
        (np.empty((0, 6)), [60.0], "states"),
        (np.full((1, 6), np.nan), [60.0], "states"),
        (LEO[None], [], "stops"),
        (LEO[None], [[60.0]], "stops"),
        (LEO[None], [60.0, 60.0], "monotonic"),
        (LEO[None], [60.0, 120.0, 100.0], "monotonic"),
        (LEO[None], [60.0, np.nan], "stops"),
    ],
)
def test_integrate_refuses_bad_arrays(
    states: FloatArray, stops: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        integrate(TWO_BODY, states, np.array(stops))


SPAN: Final = (0.0, 3.0 * HOUR)
OBJECTS: Final = np.stack(
    (LEO, ELLIPSE, state_of(EARTH_RADIUS + 800 * KM, 0.01, 1.7, 2.0))
)
FORCES: Final = (FULL, FULL, ForceModel(POLE, drag_term=0.01))
PROPAGATOR: Final = CowellPropagator(EPOCH, OBJECTS, FORCES, SPAN)
OFFSETS: Final = np.linspace(0.0, 3.0 * HOUR, 137)
WHOLE: Final = PROPAGATOR.propagate(EPOCH, OFFSETS)


def test_cowell_propagator_is_a_propagator() -> None:
    propagator: Propagator = PROPAGATOR
    assert isinstance(propagator, Propagator)
    assert len(PROPAGATOR) == 3
    assert PROPAGATOR.frame == "GCRF"
    assert repr(PROPAGATOR) == "CowellPropagator(3 objects)"
    assert isinstance(WHOLE, Ephemeris)
    assert WHOLE.frame == "GCRF"
    assert WHOLE.r.shape == (3, 137, 3)
    assert np.all(WHOLE.errors == NO_ERROR)


def test_cowell_propagator_starts_at_the_given_states() -> None:
    first = PROPAGATOR.propagate(EPOCH, 0.0)
    np.testing.assert_array_equal(first.r[:, 0], OBJECTS[:, :3])
    np.testing.assert_array_equal(first.v[:, 0], OBJECTS[:, 3:])


def test_cowell_gives_no_radius_bounds() -> None:
    bounds = PROPAGATOR.radius_bounds(EPOCH, 3.0 * HOUR)
    np.testing.assert_array_equal(bounds.lowest, np.zeros(3))
    np.testing.assert_array_equal(bounds.highest, np.full(3, np.inf))


def test_radius_bounds_check_the_duration() -> None:
    with pytest.raises(ValueError, match="duration"):
        PROPAGATOR.radius_bounds(EPOCH, -1.0)


def test_subset_gives_identical_numbers() -> None:
    subset = PROPAGATOR.subset(np.array([2, 0]))
    ephemeris = subset.propagate(EPOCH, OFFSETS)
    np.testing.assert_array_equal(ephemeris.r, WHOLE.r[[2, 0]])
    np.testing.assert_array_equal(ephemeris.v, WHOLE.v[[2, 0]])


def test_an_empty_subset_propagates_no_object() -> None:
    ephemeris = PROPAGATOR.subset(np.array([], dtype=int)).propagate(EPOCH, OFFSETS)
    assert ephemeris.r.shape == (0, 137, 3)


@pytest.mark.parametrize(
    ("indices", "error"), [([[0]], ValueError), ([1.5], ValueError), ([3], IndexError)]
)
def test_subset_refuses_bad_indices(indices: list[Any], error: type[Exception]) -> None:
    with pytest.raises(error):
        PROPAGATOR.subset(indices)


def test_part_of_the_grid_gives_identical_numbers() -> None:
    chunks = [
        chunk
        for (chunk,) in propagate_in_chunks([PROPAGATOR], EPOCH, OFFSETS, max_states=30)
    ]
    assert len(chunks) > 1
    np.testing.assert_array_equal(np.hstack([c.r for c in chunks]), WHOLE.r)
    np.testing.assert_array_equal(np.hstack([c.v for c in chunks]), WHOLE.v)
    np.testing.assert_array_equal(
        PROPAGATOR.propagate(EPOCH, OFFSETS[::-1]).r, WHOLE.r[:, ::-1]
    )


def test_an_empty_grid_gives_no_states() -> None:
    assert PROPAGATOR.propagate(EPOCH, np.empty(0)).r.shape == (3, 0, 3)


def test_offsets_count_from_the_epoch_of_the_call() -> None:
    later = EPOCH + 1_000.0 * u.s
    ephemeris = PROPAGATOR.propagate(later, np.array([-500.0, 0.0, 500.0]))
    expected = PROPAGATOR.propagate(EPOCH, np.array([500.0, 1_000.0, 1_500.0]))
    np.testing.assert_allclose(ephemeris.r, expected.r, atol=1e-3, rtol=0.0)


def test_interpolation_error_is_small() -> None:
    span = (0.0, 2.0 * HOUR)
    propagator = CowellPropagator(EPOCH, LEO[None], (FULL,), span)
    between = np.arange(5.0, span[1], 10.0)
    interpolated = propagator.propagate(EPOCH, between)
    exact = integrate(FULL, LEO[None], between)[:, 0]
    assert np.linalg.norm(interpolated.r[0] - exact[:, :3], axis=1).max() < 1e-3
    assert np.linalg.norm(interpolated.v[0] - exact[:, 3:], axis=1).max() < 1e-6


def test_the_span_may_start_before_the_epoch() -> None:
    span = (-HOUR, HOUR)
    propagator = CowellPropagator(EPOCH, LEO[None], (FULL,), span)
    ephemeris = propagator.propagate(EPOCH, np.array([-HOUR, 0.0, HOUR]))
    assert np.linalg.norm(ephemeris.r[0, 1] - LEO[:3]) < 1e-2
    exact = integrate(FULL, LEO[None], np.array([HOUR]))[0, 0]
    assert np.linalg.norm(ephemeris.r[0, 2] - exact[:3]) < 1e-2
    assert np.all(ephemeris.errors == NO_ERROR)


def test_the_span_may_end_before_the_epoch() -> None:
    span = (-2.0 * HOUR, -HOUR)
    propagator = CowellPropagator(EPOCH, LEO[None], (FULL,), span)
    ephemeris = propagator.propagate(EPOCH, np.array([-2.0 * HOUR, -HOUR]))
    exact = integrate(FULL, LEO[None], np.array([-2.0 * HOUR, -HOUR]))[:, 0]
    assert np.linalg.norm(ephemeris.r[0] - exact[:, :3], axis=1).max() < 1e-2


@pytest.mark.parametrize(
    "offsets", [[-1.0], [3.0 * HOUR + 1.0], [0.0, 3.0 * HOUR + 1.0]]
)
def test_offset_outside_span_is_refused(offsets: list[float]) -> None:
    with pytest.raises(ValueError, match="span"):
        PROPAGATOR.propagate(EPOCH, offsets)


def test_the_ends_of_the_span_are_inside_it() -> None:
    ephemeris = PROPAGATOR.propagate(EPOCH, np.array(SPAN))
    assert np.all(ephemeris.errors == NO_ERROR)


@pytest.mark.parametrize("shift", [1_234.567, 0.1, 7_777.7, 10_000.3])
def test_the_ends_of_the_span_are_inside_it_from_a_shifted_epoch(
    shift: float,
) -> None:
    """The ``Time`` difference rounds: an end must not fall out by an ulp."""
    shifted = EPOCH + shift * u.s
    ephemeris = PROPAGATOR.propagate(shifted, np.array(SPAN) - shift)
    assert np.all(ephemeris.errors == NO_ERROR)
    exact = PROPAGATOR.propagate(EPOCH, np.array(SPAN))
    np.testing.assert_allclose(ephemeris.r, exact.r, atol=1e-2, rtol=0.0)


def test_many_shifted_epochs_reach_the_end_of_the_span() -> None:
    generator = np.random.default_rng(11)
    for shift in generator.uniform(0.0, SPAN[1], 300):
        ephemeris = PROPAGATOR.propagate(EPOCH + shift * u.s, SPAN[1] - shift)
        assert np.all(ephemeris.errors == NO_ERROR)


def test_a_microsecond_is_the_slack_of_the_span() -> None:
    PROPAGATOR.propagate(EPOCH, SPAN[1] + 5e-7)
    with pytest.raises(ValueError, match="span"):
        PROPAGATOR.propagate(EPOCH, SPAN[1] + 2e-6)


def test_propagate_checks_the_epoch_and_the_offsets() -> None:
    with pytest.raises(ValueError, match="finite"):
        PROPAGATOR.propagate(EPOCH, [np.nan])


@pytest.mark.parametrize(
    ("states", "forces", "span", "step", "message"),
    [
        (OBJECTS, FORCES[:2], SPAN, 60.0, "forces"),
        (LEO, (FULL,), SPAN, 60.0, "states"),
        (OBJECTS, FORCES, (10.0, 10.0), 60.0, "span"),
        (OBJECTS, FORCES, (10.0, 0.0), 60.0, "span"),
        (OBJECTS, FORCES, (0.0, np.inf), 60.0, "span"),
        (OBJECTS, FORCES, SPAN, 0.0, "step"),
        (OBJECTS, FORCES, SPAN, np.nan, "step"),
    ],
)
def test_constructor_refuses_bad_arguments(
    states: FloatArray,
    forces: tuple[ForceModel, ...],
    span: tuple[float, float],
    step: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CowellPropagator(EPOCH, states, forces, span, step=step)


def test_constructor_takes_no_objects() -> None:
    propagator = CowellPropagator(EPOCH, np.empty((0, 6)), (), SPAN)
    assert len(propagator) == 0
    assert propagator.propagate(EPOCH, OFFSETS).r.shape == (0, 137, 3)


def test_the_end_of_the_span_is_the_integrations_own() -> None:
    span = (0.0, 1_000.0)
    propagator = CowellPropagator(EPOCH, LEO[None], (FULL,), span)
    exact = integrate(FULL, LEO[None], np.array([1_000.0]))[0, 0]
    last = propagator.propagate(EPOCH, 1_000.0)
    np.testing.assert_array_equal(last.r[0, 0], exact[:3])
    np.testing.assert_array_equal(last.v[0, 0], exact[3:])


def test_a_longest_step_makes_the_solver_keep_more_steps() -> None:
    # DOP853 steps about 3 min in LEO at these tolerances.
    span = (0.0, HOUR)
    free = CowellPropagator(EPOCH, LEO[None], (FULL,), span)
    held = CowellPropagator(EPOCH, LEO[None], (FULL,), span, step=10.0)
    assert len(pickle.dumps(held)) > 5 * len(pickle.dumps(free))
    times = np.linspace(0.0, HOUR, 50)
    gap = held.propagate(EPOCH, times).r - free.propagate(EPOCH, times).r
    assert np.abs(gap).max() < 1e-3


def test_a_step_longer_than_the_span_gives_its_two_ends() -> None:
    propagator = CowellPropagator(EPOCH, LEO[None], (FULL,), (0.0, 30.0), step=60.0)
    ephemeris = propagator.propagate(EPOCH, np.array([0.0, 15.0, 30.0]))
    assert np.all(ephemeris.errors == NO_ERROR)
    later = integrate(FULL, LEO[None], np.array([15.0, 30.0]))[:, 0]
    exact = np.concatenate((LEO[None], later))
    gap = np.linalg.norm(ephemeris.r[0] - exact[:, :3], axis=1)
    # The ends are the integration's; in between DOP853's own interpolant is
    # within its tolerances (0.1 mm), where a cubic between the ends was off
    # by 2 cm.
    assert gap[[0, 2]].max() < 1e-6
    assert gap[1] < 1e-4


def _decayed() -> tuple[CowellPropagator, Ephemeris, FloatArray]:
    states = np.stack((LEO, REENTERING, ELLIPSE))
    forces = (TWO_BODY, TWO_BODY, TWO_BODY)
    span = (0.0, 2.0 * HOUR)
    grid = np.arange(0.0, span[1] + 1.0, 30.0)
    propagator = CowellPropagator(EPOCH, states, forces, span, step=60.0)
    return propagator, propagator.propagate(EPOCH, grid), states


def test_reentered_object_has_error_code_six() -> None:
    propagator, ephemeris, states = _decayed()
    failed = ephemeris.errors[1] != NO_ERROR
    count = np.count_nonzero(~failed)
    assert 0 < count < len(failed)
    assert failed[count:].all()
    # Valid as far as an integration of the object alone reaches.
    reached = integrate(
        TWO_BODY, states[1:2], ephemeris.offsets[1:], stop_at_surface=True
    )[:, 0]
    assert count == 1 + np.count_nonzero(np.isfinite(reached[:, 0]))
    assert np.all(ephemeris.errors[1, count:] == DECAYED)
    assert DECAYED in ERROR_MESSAGES
    assert np.isnan(ephemeris.r[1, count:]).all()
    assert np.isnan(ephemeris.v[1, count:]).all()
    assert np.isfinite(ephemeris.r[1, :count]).all()
    assert np.all(np.linalg.norm(ephemeris.r[1, :count], axis=1) >= EARTH_RADIUS)
    # The others are unaffected, in number as in code.
    assert np.all(ephemeris.errors[[0, 2]] == NO_ERROR)
    alone = CowellPropagator(
        EPOCH, states[[0, 2]], (TWO_BODY, TWO_BODY), (0.0, 2.0 * HOUR), step=60.0
    ).propagate(EPOCH, ephemeris.offsets)
    np.testing.assert_array_equal(alone.r, ephemeris.r[[0, 2]])
    np.testing.assert_array_equal(
        propagator.subset(np.array([1])).propagate(EPOCH, ephemeris.offsets).r,
        ephemeris.r[[1]],
    )


def test_an_object_gone_before_the_second_node_has_no_valid_state() -> None:
    inside = LEO.copy()
    inside[:3] *= 0.5
    propagator = CowellPropagator(EPOCH, inside[None], (TWO_BODY,), (0.0, HOUR))
    ephemeris = propagator.propagate(EPOCH, np.array([0.0, 60.0]))
    assert np.all(ephemeris.errors == DECAYED)
    assert np.isnan(ephemeris.r).all()


def test_close_to_sgp4_over_a_short_arc() -> None:
    omm = fixture_by_number()[900]
    sgp4 = Sgp4Propagator((omm,))
    offsets = np.linspace(0.0, 2.0 * HOUR, 25)
    reference = sgp4.propagate(FIXTURE_START, offsets)
    times = reference.times
    r, v = teme_to_gcrf(times, reference.r[0], reference.v[0])
    state = np.concatenate((r[0], v[0]))[None]
    force = ForceModel(earth_pole(FIXTURE_START))
    propagator = CowellPropagator(FIXTURE_START, state, (force,), (0.0, 2.0 * HOUR))
    cowell = propagator.propagate(FIXTURE_START, offsets)
    # 182 m apart after 2 h (SGP4's theory and drag against these forces):
    # 0.5 km allows for that, and a TEME state taken for GCRF is 6 to 48 km off.
    assert np.linalg.norm(cowell.r[0] - r, axis=1).max() < 0.5 * KM


def test_cowell_propagator_pickles() -> None:
    copy = pickle.loads(pickle.dumps(PROPAGATOR))
    assert isinstance(copy, CowellPropagator)
    ephemeris = copy.propagate(EPOCH, OFFSETS)
    np.testing.assert_array_equal(ephemeris.r, WHOLE.r)
    np.testing.assert_array_equal(ephemeris.v, WHOLE.v)
    with pytest.raises(ValueError, match="span"):
        copy.propagate(EPOCH, [-1.0])


def test_a_decayed_object_pickles() -> None:
    propagator, ephemeris, _ = _decayed()
    copy = pickle.loads(pickle.dumps(propagator))
    again = copy.propagate(EPOCH, ephemeris.offsets)
    np.testing.assert_array_equal(again.r, ephemeris.r)
    np.testing.assert_array_equal(again.errors, ephemeris.errors)


def test_the_propagator_is_built_with_each_object_alone() -> None:
    """One object's presence does not change another's numbers."""
    alone = CowellPropagator(EPOCH, OBJECTS[:1], FORCES[:1], SPAN)
    np.testing.assert_array_equal(alone.propagate(EPOCH, OFFSETS).r, WHOLE.r[:1])


def test_drag_lowers_the_orbit_through_the_density_table() -> None:
    """A low orbit crossing the table's bands, where the density steps."""
    state = state_of(EARTH_RADIUS + 300 * KM, 0.01)
    stops = np.array([HOUR, DAY])
    with_drag = integrate(ForceModel(POLE, drag_term=0.02), state[None], stops)
    without = integrate(FULL, state[None], stops)
    assert energy(with_drag[-1, 0]) < energy(without[-1, 0])


@pytest.mark.parametrize("altitude", [300 * KM, 450 * KM])
def test_drag_decays_a_circular_orbit_at_the_analytic_rate(altitude: float) -> None:
    """da/dt = -rho B sqrt(mu a) (1 - omega a / v)^2 on a circular equatorial
    orbit in an atmosphere turning with the Earth, over an hour, in which a and
    the density barely change."""
    semi_major_axis = EARTH_RADIUS + altitude
    term = 0.02
    state = state_of(semi_major_axis, inclination=0.0)
    end = integrate(
        ForceModel(POLE, degree=0, drag_term=term), state[None], np.array([HOUR])
    )[-1, 0]
    decay = -EARTH_MU / (2.0 * energy(end)) - semi_major_axis
    speed = math.sqrt(EARTH_MU / semi_major_axis)
    rho = float(density(np.array(altitude)))
    co_rotation = (1.0 - EARTH_ROTATION_RATE * semi_major_axis / speed) ** 2
    rate = -rho * term * math.sqrt(EARTH_MU * semi_major_axis) * co_rotation
    assert decay == pytest.approx(rate * HOUR, rel=0.02)


def test_importing_the_module_does_not_load_scipy() -> None:
    """scipy.integrate takes 0.2 s to import, on every command that screens."""
    assert loaded_after("import keepout.propagation.cowell", ["scipy"]).loaded == []
