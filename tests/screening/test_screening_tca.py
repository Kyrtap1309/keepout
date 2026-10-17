"""Close approaches on a time grid: exact on straight-line motion, the same
however the grid is cut into chunks, and on SGP4 every minimum the
brute-force oracle and SOCRATES find."""

import dataclasses
import functools
import math
import pickle
from dataclasses import dataclass, field
from typing import Self

import astropy.units as u
import numpy as np
import numpy.typing as npt
import pytest
from astropy.time import Time
from hypothesis import given, settings, target
from hypothesis import strategies as st
from screening_data import FAST_SPEED, load_fixture, socrates_approaches
from screening_oracle import brute_force, positions, refine, states

from keepout.orbits import EARTH_MU
from keepout.propagation import Ephemeris, RadiusBounds, Sgp4Propagator, time_offsets
from keepout.screening import (
    DEFAULT_VOLUME,
    ScreeningVolume,
    VolumeRule,
    apogee_perigee_filter,
    volume_for,
)
from keepout.screening.tca import (
    DEFAULT_STEP,
    GRAVITY_MARGIN,
    MIN_RADIUS,
    PERTURBATION,
    SPEED_MARGIN,
    ApproachSearch,
    ObjectFailure,
    RangeMinimum,
    SearchResult,
    speed_bounds,
    time_grid,
)
from keepout.state import RefFrame
from keepout.units import KM

HOUR = 3600.0
REFERENCE = Time("2026-10-06T00:00:00", scale="utc")
"""Time zero of the straight-line objects."""

FIXTURE = load_fixture()
BY_NUMBER = {omm.catalog_number: omm for omm in FIXTURE}
COLUMN = {omm.catalog_number: index for index, omm in enumerate(FIXTURE)}
PRIMARIES = FIXTURE[::9]
"""25 primaries spread over the fixture (LEO, MEO, GEO, HEO)."""
SOCRATES = [
    approach
    for approach in socrates_approaches()
    if approach.primary in BY_NUMBER and approach.secondary in BY_NUMBER
]
"""The SOCRATES approaches of two fixture objects (31 fast, 1002 slow)."""
CLOSE = sorted(
    {omm.catalog_number for omm in PRIMARIES}
    | {approach.primary for approach in SOCRATES}
    | {approach.secondary for approach in SOCRATES}
)
"""Primaries for the oracle tests: PRIMARIES and the objects of SOCRATES's
approaches, which have many minima."""
WEEK_START = Time("2026-10-06T00:00:00", scale="utc")


@dataclass(frozen=True)
class StraightLines:
    """Objects in straight-line motion from REFERENCE, plus a circle of radius
    ``circle[i, 0]`` [m] in the x-z plane at ``circle[i, 1]`` rad/s (from
    phase ``circle[i, 2]``); object i fails (code 6) from ``fail[i, 0]`` to
    ``fail[i, 1]`` s after REFERENCE."""

    r0: npt.NDArray[np.float64]
    v0: npt.NDArray[np.float64]
    fail: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 2)))
    circle: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 3)))
    frame: RefFrame = "TEME"

    def __post_init__(self) -> None:
        if len(self.fail) == 0:
            object.__setattr__(self, "fail", np.full((len(self.r0), 2), np.inf))
        if len(self.circle) == 0:
            object.__setattr__(self, "circle", np.zeros((len(self.r0), 3)))

    def __len__(self) -> int:
        return len(self.r0)

    def subset(self, indices: npt.ArrayLike) -> Self:
        chosen = np.asarray(indices, dtype=np.intp)
        return type(self)(
            self.r0[chosen], self.v0[chosen], self.fail[chosen], self.circle[chosen]
        )

    def seconds(self, epoch: Time, offsets: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return float((epoch - REFERENCE).to_value(u.s)) + time_offsets(epoch, offsets)

    def propagate(self, epoch: Time, offsets: npt.ArrayLike) -> Ephemeris:
        times = self.seconds(epoch, offsets)
        r = self.r0[:, None, :] + self.v0[:, None, :] * times[None, :, None]
        v = np.broadcast_to(self.v0[:, None, :], r.shape).copy()
        radius, rate, phase = (self.circle[:, i : i + 1] for i in range(3))
        angle = rate * times + phase
        r[..., 0] += radius * np.cos(angle)
        r[..., 2] += radius * np.sin(angle)
        v[..., 0] -= radius * rate * np.sin(angle)
        v[..., 2] += radius * rate * np.cos(angle)
        failed = (times >= self.fail[:, :1]) & (times < self.fail[:, 1:])
        r[failed] = np.nan
        v[failed] = np.nan
        errors = np.where(failed, 6, 0).astype(np.uint8)
        return Ephemeris(epoch, time_offsets(epoch, offsets), r, v, errors, self.frame)

    def radius_bounds(self, start: Time, duration: float) -> RadiusBounds:
        # The closest and farthest points of the line, widened by the circle.
        first = self.seconds(start, 0.0)[0]
        along = -np.sum(self.r0 * self.v0, axis=1) / np.sum(self.v0 * self.v0, axis=1)
        closest = np.clip(along, first, first + duration)

        def radius(times: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            distances: npt.NDArray[np.float64] = np.linalg.norm(
                self.r0 + self.v0 * times[:, None], axis=1
            )
            return distances

        ends = np.maximum(
            radius(np.full(len(self), first)),
            radius(np.full(len(self), first + duration)),
        )
        return RadiusBounds(
            radius(closest) - self.circle[:, 0], ends + self.circle[:, 0]
        )


PRIMARY = (np.array([7000 * KM, 0.0, 0.0]), np.array([0.0, 1000.0, 0.0]))
"""A primary 7000 km from the centre at time zero, moving at 1 km/s along y.
(Straight lines are not orbits: the sieves assume no object is faster than an
orbit through its radii, so the objects here stay slower than that.)"""

FAST = np.array([0.0, -5000.0, 6000.0])
"""A relative velocity of 7.8 km/s (the other object moves at 7.2 km/s)."""

SLOW = np.array([0.0, 0.3, 0.4])
"""A relative velocity of 0.5 m/s."""


def passing(
    miss: npt.ArrayLike, relative: npt.ArrayLike, tca: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """An object whose position relative to PRIMARY is ``miss + relative (t -
    tca)``: closest at ``tca`` (s after REFERENCE) if ``miss`` is
    perpendicular to ``relative``."""
    miss, relative = np.asarray(miss, float), np.asarray(relative, float)
    return PRIMARY[0] + miss - relative * tca, PRIMARY[1] + relative


def lines(
    *objects: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
) -> StraightLines:
    return StraightLines(
        np.array([o[0] for o in objects]), np.array([o[1] for o in objects])
    )


def search(
    catalog: StraightLines,
    duration: float = 2 * HOUR,
    volume: ScreeningVolume = DEFAULT_VOLUME,
    *,
    start: Time = REFERENCE,
    primaries: StraightLines | None = None,
    pairs: npt.ArrayLike | None = None,
    step: float = DEFAULT_STEP,
) -> ApproachSearch:
    """A search of every pair (or of ``pairs``), with the objects' own radius
    bounds."""
    primaries = lines(PRIMARY) if primaries is None else primaries
    if pairs is None:
        pairs = np.ones((len(primaries), len(catalog)), dtype=bool)
    return ApproachSearch(
        primaries,
        catalog,
        start,
        duration,
        volume,
        pairs,
        step=step,
        primary_bounds=primaries.radius_bounds(start, duration),
        catalog_bounds=catalog.radius_bounds(start, duration),
    )


def summary(result: SearchResult) -> list[object]:
    """Everything in a result, exact, for comparisons."""
    minima: list[object] = [
        (
            m.primary_index,
            m.secondary_index,
            m.offset,
            m.distance,
            m.r_primary.tolist(),
            m.v_primary.tolist(),
            m.r_secondary.tolist(),
            m.v_secondary.tolist(),
        )
        for m in result.minima
    ]
    return [*minima, dict(result.primary_failures), dict(result.catalog_failures)]


# The time grid.


@pytest.mark.parametrize(
    ("duration", "step", "samples"),
    [
        (600.0, 120.0, 6),
        (601.0, 120.0, 7),
        (100.0, 120.0, 2),
        (7 * 86400.0, 120.0, 5041),
    ],
)
def test_the_grid_spans_the_window_at_most_a_step_apart(
    duration: float, step: float, samples: int
) -> None:
    grid = time_grid(duration, step)
    assert len(grid) == samples
    assert (grid[0], grid[-1]) == (0.0, duration)
    assert np.all(np.diff(grid) <= step)
    assert np.allclose(np.diff(grid), duration / (samples - 1))
    assert not grid.flags.writeable


def test_an_empty_window_has_one_sample() -> None:
    assert time_grid(0.0).tolist() == [0.0]


@pytest.mark.parametrize(
    ("duration", "step", "message"),
    [
        (-1.0, 60.0, "duration must be finite and >= 0, got -1.0"),
        (math.nan, 60.0, "duration must be finite"),
        (600.0, 0.0, "step must be finite and > 0, got 0.0"),
        (600.0, math.inf, "step must be finite and > 0"),
    ],
)
def test_the_grid_needs_a_window_and_a_step(
    duration: float, step: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        time_grid(duration, step)


# Bounds on speed and acceleration.


def test_speed_bounds_are_the_fastest_orbit_within_the_radii() -> None:
    bounds = RadiusBounds(
        np.array([7000 * KM, 7000 * KM, 7000 * KM, 6000 * KM]),
        np.array([7000 * KM, 42000 * KM, math.inf, 7000 * KM]),
    )
    mu = EARTH_MU * GRAVITY_MARGIN
    circular = math.sqrt(mu / (7000 * KM))
    eccentric = math.sqrt(2 * mu * 42000 / (7000 * KM * 49000))
    escape = math.sqrt(2 * mu / (7000 * KM))
    floored = math.sqrt(2 * mu * 7000 * KM / (MIN_RADIUS * (MIN_RADIUS + 7000 * KM)))
    np.testing.assert_allclose(
        speed_bounds(bounds),
        np.array([circular, eccentric, escape, floored]) * (1 + SPEED_MARGIN),
        rtol=1e-15,
    )


def j2_gravity(r: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Point mass and J2 (WGS 72, SGP4's constants)."""
    mu, radius, j2 = 398600.8e9, 6378.135e3, 0.001082616
    distance = np.linalg.norm(r, axis=-1, keepdims=True)
    z2 = (r[..., 2:] / distance) ** 2
    factor = -1.5 * j2 * mu * radius**2 / distance**5
    j2_part = factor * r * np.concatenate([1 - 5 * z2, 1 - 5 * z2, 3 - 5 * z2], axis=-1)
    gravity: npt.NDArray[np.float64] = -mu * r / distance**3 + j2_part
    return gravity


def test_sgp4_moves_within_the_bounds_the_sieves_assume() -> None:
    # Over the week after the download, every 5 minutes: speed and the
    # acceleration beyond point mass and J2, both from positions.
    propagator = Sgp4Propagator(FIXTURE)
    week = 7 * 86400.0
    centres = np.arange(0.0, week, 300.0)
    delta = 1.0
    ephemeris = propagator.propagate(
        WEEK_START, np.concatenate([centres - delta, centres, centres + delta])
    )
    r = ephemeris.r.reshape(len(FIXTURE), 3, len(centres), 3)
    valid = ephemeris.valid.reshape(len(FIXTURE), 3, len(centres)).all(axis=1)
    speed = np.linalg.norm(r[:, 2] - r[:, 0], axis=-1) / (2 * delta)
    bounds = speed_bounds(propagator.radius_bounds(WEEK_START, week))
    assert np.all(speed[valid] <= np.broadcast_to(bounds[:, None], speed.shape)[valid])
    acceleration = (r[:, 2] - 2 * r[:, 1] + r[:, 0]) / delta**2
    beyond = np.linalg.norm(acceleration - j2_gravity(r[:, 1]), axis=-1)
    assert np.max(beyond[valid]) < PERTURBATION


# Straight lines: exact geometry.


@pytest.mark.parametrize(
    ("relative", "precision"),
    # Positions thousands of km from the centre are rounded to ~4e-9 m, and
    # a 0.5 m/s pass has a flat range: its TCA is known to ~10 us.
    [(FAST, 1e-6), (SLOW, 1e-4)],
    ids=["fast", "slow"],
)
@pytest.mark.parametrize("tca", [1800.0, 1830.25, 3599.0])
def test_a_pass_inside_the_volume_is_found_at_its_tca(
    relative: npt.NDArray[np.float64], precision: float, tca: float
) -> None:
    miss = np.cross(relative, [1.0, 0.0, 0.0])
    miss *= 3 * KM / np.linalg.norm(miss)
    catalog = lines(passing(miss, relative, tca))
    result = search(catalog).run()
    (minimum,) = result.minima
    assert (minimum.primary_index, minimum.secondary_index) == (0, 0)
    assert minimum.offset == pytest.approx(tca, abs=precision)
    assert minimum.distance == pytest.approx(3 * KM, abs=1e-6)
    expected = (
        PRIMARY[0] + PRIMARY[1] * minimum.offset,
        PRIMARY[1],
        catalog.r0[0] + catalog.v0[0] * minimum.offset,
        catalog.v0[0],
    )
    for state, value in zip(
        (
            minimum.r_primary,
            minimum.v_primary,
            minimum.r_secondary,
            minimum.v_secondary,
        ),
        expected,
        strict=True,
    ):
        np.testing.assert_allclose(state, value, rtol=1e-15, atol=1e-9)
    assert result.primary_failures == result.catalog_failures == {}


@pytest.mark.parametrize("relative", [FAST, SLOW], ids=["fast", "slow"])
def test_only_passes_inside_the_volume_count(relative: npt.NDArray[np.float64]) -> None:
    across = np.cross(relative, [1.0, 0.0, 0.0])
    across /= np.linalg.norm(across)
    catalog = lines(
        passing(across * (5 * KM - 1.0), relative, 1000.0),
        passing(across * (5 * KM + 1.0), relative, 2000.0),
        passing(across * 100.0, relative, 3000.0),
    )
    found = search(catalog).run().minima
    assert [m.secondary_index for m in found] == [0, 2]


def rtn_axes(tca: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """PRIMARY's radial and in-track directions at ``tca`` (its normal is z)."""
    radial = PRIMARY[0] + PRIMARY[1] * tca
    radial /= np.linalg.norm(radial)
    return radial, np.cross([0.0, 0.0, 1.0], radial)


def test_an_ellipsoid_counts_the_position_at_tca() -> None:
    # 1 km radial, 10 km in-track and cross-track; the passes cross along N.
    # 3 km off along R at TCA is outside, though the pass crosses the volume's
    # cross-track extent; 3 km in-track, or 600 m radial, is inside.
    volume = ScreeningVolume(1 * KM, 10 * KM, 10 * KM)
    across = np.array([0.0, 0.0, 6000.0])
    catalog = lines(
        passing(rtn_axes(1000.0)[0] * 3 * KM, across, 1000.0),
        passing(rtn_axes(2000.0)[1] * 3 * KM, across, 2000.0),
        passing(rtn_axes(3000.0)[0] * 600.0, across, 3000.0),
    )
    found = search(catalog, volume=volume).run().minima
    assert [m.secondary_index for m in found] == [1, 2]


def test_minima_outside_the_window_are_not_found() -> None:
    # The range is still falling at the end, or already rising at the start.
    miss = np.cross(FAST, [1.0, 0.0, 0.0])
    miss *= 1 * KM / np.linalg.norm(miss)
    catalog = lines(
        passing(miss, FAST, -0.5),
        passing(miss, FAST, 0.5),
        passing(miss, FAST, 2 * HOUR - 0.5),
        passing(miss, FAST, 2 * HOUR + 0.5),
    )
    found = search(catalog).run().minima
    assert [m.secondary_index for m in found] == [1, 2]


@pytest.mark.parametrize(
    ("relative", "precision"), [(FAST, 1e-6), (SLOW, 1e-4)], ids=["fast", "slow"]
)
def test_consecutive_windows_find_the_minima_of_the_whole(
    relative: npt.NDArray[np.float64], precision: float
) -> None:
    across = np.cross(relative, [1.0, 0.0, 0.0])
    across *= 2 * KM / np.linalg.norm(across)
    catalog = lines(
        *(passing(across, relative, tca) for tca in (100.0, 2999.9, 3000.1, 5000.0))
    )
    whole = search(catalog, 2 * HOUR).run().minima
    first = search(catalog, 3000.0).run().minima
    later = REFERENCE + 3000.0 * u.s
    second = search(catalog, 2 * HOUR - 3000.0, start=later).run().minima
    assert [m.secondary_index for m in whole] == [0, 1, 2, 3]
    assert [m.secondary_index for m in first] == [0, 1]
    assert [m.secondary_index for m in second] == [2, 3]
    joined = [m.offset for m in first] + [3000.0 + m.offset for m in second]
    np.testing.assert_allclose(
        [m.offset for m in whole], joined, rtol=0, atol=precision
    )


def test_an_object_at_the_same_position_has_no_minimum() -> None:
    assert search(lines(PRIMARY)).run().minima == ()


@pytest.mark.parametrize("radius", [5 * KM, 1.2 * KM])
def test_minima_closer_than_a_step_are_told_apart(radius: float) -> None:
    # A companion 2 km off along x circles 1 km around that point every 200 s
    # (1 m/s^2, 31 m/s): the range has a minimum every 200 s and a maximum
    # 100 s from each, closer than the 120 s step. The intervals are
    # resampled every SLOW_STEP, so every minimum is found. With a 1.2 km
    # sphere the chords between samples miss the volume: the chord sieve's
    # pad keeps them.
    rate, phase = 2 * math.pi / 200.0, 0.3
    companion = StraightLines(
        (PRIMARY[0] + [2 * KM, 0.0, 0.0])[None],
        PRIMARY[1][None],
        circle=np.array([[1 * KM, rate, phase]]),
    )
    volume = ScreeningVolume.sphere(radius)
    found = search(companion, HOUR, volume).run().minima
    expected = (math.pi - phase) / rate + 200.0 * np.arange(18)
    assert DEFAULT_STEP == 120.0
    np.testing.assert_allclose([m.offset for m in found], expected, rtol=0, atol=1e-5)
    np.testing.assert_allclose([m.distance for m in found], 1 * KM, rtol=1e-12)


def test_every_pair_of_several_primaries_is_searched() -> None:
    other = (np.array([0.0, 7000 * KM, 0.0]), np.array([-1000.0, 0.0, 0.0]))
    primaries = lines(PRIMARY, other)
    miss = np.cross(FAST, [1.0, 0.0, 0.0])
    miss *= 1 * KM / np.linalg.norm(miss)
    across = np.array([5000.0, 3000.0, 0.0])
    near_other = (other[0] + [0.0, 0.0, 2 * KM] - across * 1500.0, other[1] + across)
    catalog = lines(passing(miss, FAST, 900.0), near_other, passing(miss, FAST, 300.0))
    pairs = np.array([[True, True, False], [True, True, True]])
    found = search(catalog, primaries=primaries, pairs=pairs).run().minima
    assert [(m.primary_index, m.secondary_index) for m in found] == [(0, 0), (1, 1)]
    assert [m.offset for m in found] == pytest.approx([900.0, 1500.0], abs=1e-6)


@pytest.mark.parametrize(
    ("relative", "precision"), [(FAST, 1e-6), (SLOW, 1e-4)], ids=["fast", "slow"]
)
@pytest.mark.parametrize(
    ("tca", "fail"),
    [(0.5, (-np.inf, -1.0)), (2 * HOUR - 0.5, (2 * HOUR + 1.0, np.inf))],
    ids=["start", "end"],
)
def test_a_failure_just_outside_the_window_is_not_the_objects(
    relative: npt.NDArray[np.float64],
    precision: float,
    tca: float,
    fail: tuple[float, float],
) -> None:
    # The central difference at the window's ends reaches 2 s outside it:
    # an object failing a second outside is fine in the window, and its
    # minimum half a second from the end is found as accurately.
    miss = np.cross(relative, [1.0, 0.0, 0.0])
    miss *= 1 * KM / np.linalg.norm(miss)
    fine = lines(passing(miss, relative, tca))
    failing = StraightLines(fine.r0, fine.v0, np.array([fail]))
    result = search(failing).run()
    assert result.primary_failures == result.catalog_failures == {}
    (minimum,) = result.minima
    assert minimum.offset == pytest.approx(tca, abs=precision)
    assert minimum.distance == pytest.approx(1 * KM, abs=1e-6)
    # Every evaluation in the window, failing or not, is the same object's.
    assert search(fine).run().minima[0].offset == pytest.approx(
        minimum.offset, abs=precision
    )


def test_failed_objects_are_listed_with_their_first_failure() -> None:
    miss = np.cross(FAST, [1.0, 0.0, 0.0])
    miss *= 1 * KM / np.linalg.norm(miss)
    catalog = lines(
        passing(miss, FAST, 1000.0),
        passing(miss, FAST, 1000.0),
        passing(miss, FAST, 5000.0),
        passing(miss, FAST, 3000.0),
    )
    fail = np.full((4, 2), np.inf)
    fail[0] = (4000.0, np.inf)  # fails on the grid after its minimum
    fail[1] = (999.5, 1000.5)  # fails between two samples, at its TCA
    fail[2] = (-np.inf, 600.0)  # fails at the start
    catalog = StraightLines(catalog.r0, catalog.v0, fail)
    result = search(catalog).run()
    assert [(m.secondary_index, round(m.offset, 6)) for m in result.minima] == [
        (0, 1000.0),
        (2, 5000.0),
        (3, 3000.0),
    ]
    assert result.primary_failures == {}
    assert dict(result.catalog_failures) == {
        0: ObjectFailure(4080.0, 6),
        1: ObjectFailure(result.catalog_failures[1].offset, 6),
        2: ObjectFailure(0.0, 6),
    }
    assert result.catalog_failures[1].offset == pytest.approx(1000.0, abs=1e-6)
    # A primary failing on the grid, and one failing at a TCA.
    for window, offset in [((600.0, 700.0), 600.0), ((999.5, 1000.5), 1000.0)]:
        failing = StraightLines(PRIMARY[0][None], PRIMARY[1][None], np.array([window]))
        (failure,) = search(catalog, primaries=failing).run().primary_failures.values()
        assert failure.offset == pytest.approx(offset, abs=1e-6)


# The same result however the grid is cut.

MISSES = lines(
    *(
        passing(
            np.cross(relative, axis) * scale / np.linalg.norm(np.cross(relative, axis)),
            relative,
            tca,
        )
        for relative, axis, scale, tca in [
            (FAST, [1.0, 0.0, 0.0], 1 * KM, 700.0),
            (FAST, [0.0, 1.0, 0.0], 4 * KM, 2400.0),
            (SLOW, [1.0, 0.0, 0.0], 2 * KM, 3700.0),
            (FAST, [1.0, 0.0, 0.0], 6 * KM, 5000.0),
            (FAST * 0.05, [1.0, 0.0, 0.0], 3 * KM, 6100.0),
        ]
    )
)
CIRCLING = 2 * math.pi / 200.0
FAILING = StraightLines(
    np.concatenate([MISSES.r0, (PRIMARY[0] + [2 * KM, 0.0, 0.0])[None]]),
    np.concatenate([MISSES.v0, PRIMARY[1][None]]),
    np.concatenate([MISSES.fail, [[2900.5, 2901.5]]]),
    np.concatenate([MISSES.circle, [[1 * KM, CIRCLING, 0.3]]]),
)
"""MISSES and a companion circling 1 km around a point 2 km from PRIMARY
(a minimum every 200 s, at 90.45 s + 200 k), failing for a second in the
interval of the minimum at 2890.45 s."""


@functools.cache
def failing_whole() -> SearchResult:
    """FAILING searched in one chunk (computed once: it is slow)."""
    return search(FAILING).run()


def test_a_failure_loses_the_minimum_of_its_own_interval_only() -> None:
    whole = failing_whole()
    companion = [m.offset for m in whole.minima if m.secondary_index == 5]
    expected = (math.pi - 0.3) / CIRCLING + 200.0 * np.arange(36)
    np.testing.assert_allclose(companion, np.delete(expected, 14), rtol=0, atol=1e-5)
    assert len(whole.minima) == 4 + 35
    assert dict(whole.catalog_failures) == {5: ObjectFailure(2901.0, 6)}


@settings(deadline=None, max_examples=30)
@given(max_states=st.integers(min_value=1, max_value=40))
def test_chunks_do_not_change_the_result(max_states: int) -> None:
    # 7 objects: chunks of 2 to 5 of the 61 samples, or all of them.
    chunked = search(FAILING).run(max_states=max_states)
    assert summary(chunked) == summary(failing_whole())


def test_catalog_parts_merge_into_the_whole() -> None:
    # Screening searches parts of the catalog apart, numbered alike.
    whole = search(FAILING).run()
    parts = [
        search(FAILING, pairs=np.isin(np.arange(len(FAILING)), part)[None]).run()
        for part in ([0, 2, 4], [1, 3, 5])
    ]
    assert summary(SearchResult.merge(parts)) == summary(whole)


def test_a_result_pickles_with_its_failures() -> None:
    fail = np.full((len(MISSES), 2), np.inf)
    fail[1] = (3000.0, np.inf)
    catalog = StraightLines(MISSES.r0, MISSES.v0, fail)
    result = search(catalog).run()
    assert dict(result.catalog_failures) == {1: ObjectFailure(3000.0, 6)}
    copy = pickle.loads(pickle.dumps(result))
    assert summary(copy) == summary(result)


def test_the_search_keeps_its_arguments() -> None:
    found = search(MISSES, 600.0, step=200.0)
    assert (found.start, found.duration) == (REFERENCE, 600.0)
    assert len(found.primaries) == 1
    assert len(found.catalog) == 5


def test_nothing_to_search_finds_nothing() -> None:
    empty = SearchResult((), {}, {})
    nothing = np.zeros((1, 5), dtype=bool)
    assert summary(search(MISSES, HOUR, pairs=nothing).run()) == summary(empty)
    assert summary(search(MISSES, 0.0).run()) == summary(empty)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"start": Time([REFERENCE, REFERENCE])}, "start must be a single time"),
        ({"pairs": np.ones((1, 5))}, r"pairs must be booleans of shape \(1, 5\)"),
        ({"pairs": np.ones((2, 5), dtype=bool)}, "got bool"),
        ({"volumes": [DEFAULT_VOLUME] * 2}, "2 volumes given for 1 primaries"),
        ({"duration": -1.0}, "duration must be finite and >= 0"),
        (
            {"catalog_bounds": RadiusBounds(np.ones(2), np.ones(2))},
            "radius bounds of 1 primaries and 2 catalog objects given for 1 and 5",
        ),
        (
            {
                "primaries": StraightLines(
                    PRIMARY[0][None], PRIMARY[1][None], frame="GCRF"
                )
            },
            "share a frame, got GCRF and TEME",
        ),
    ],
)
def test_a_search_needs_consistent_arguments(
    change: dict[str, object], message: str
) -> None:
    arguments: dict[str, object] = {
        "primaries": lines(PRIMARY),
        "catalog": MISSES,
        "start": REFERENCE,
        "duration": HOUR,
        "volumes": DEFAULT_VOLUME,
        "pairs": np.ones((1, 5), dtype=bool),
        "primary_bounds": lines(PRIMARY).radius_bounds(REFERENCE, HOUR),
        "catalog_bounds": MISSES.radius_bounds(REFERENCE, HOUR),
    } | change
    with pytest.raises(ValueError, match=message):
        ApproachSearch(**arguments)  # type: ignore[arg-type]


def test_a_run_propagates_at_least_one_state_at_once() -> None:
    with pytest.raises(ValueError, match="max_states must be at least 1, got 0"):
        search(MISSES).run(max_states=0)


def test_results_are_sorted_and_merge_the_earliest_failures() -> None:
    def minimum(primary: int, secondary: int, offset: float) -> RangeMinimum:
        return RangeMinimum(primary, secondary, offset, 1.0, *np.eye(4, 3)[:4])

    first = SearchResult(
        (minimum(1, 0, 5.0), minimum(0, 3, 9.0), minimum(0, 3, 2.0)),
        {2: ObjectFailure(50.0, 6), 0: ObjectFailure(10.0, 1)},
        {7: ObjectFailure(30.0, 6)},
    )
    assert [(m.primary_index, m.secondary_index, m.offset) for m in first.minima] == [
        (0, 3, 2.0),
        (0, 3, 9.0),
        (1, 0, 5.0),
    ]
    assert list(first.primary_failures) == [0, 2]
    second = SearchResult(
        (minimum(0, 0, 1.0),), {2: ObjectFailure(40.0, 3)}, {7: ObjectFailure(31.0, 1)}
    )
    merged = SearchResult.merge([first, second])
    assert [(m.primary_index, m.secondary_index) for m in merged.minima] == [
        (0, 0),
        (0, 3),
        (0, 3),
        (1, 0),
    ]
    assert dict(merged.primary_failures) == {
        0: ObjectFailure(10.0, 1),
        2: ObjectFailure(40.0, 3),
    }
    assert dict(merged.catalog_failures) == {7: ObjectFailure(30.0, 6)}
    # The failures are copies of those given.
    given = {7: ObjectFailure(30.0, 6)}
    copied = SearchResult((), {}, given)
    given.clear()
    assert dict(copied.catalog_failures) == {7: ObjectFailure(30.0, 6)}


def test_a_minimum_holds_read_only_states() -> None:
    r = np.array([1.0, 2.0, 3.0])
    minimum = RangeMinimum(0, 1, 10.0, 5.0, r, r, r, r)
    r[0] = 9.0
    assert minimum.r_primary.tolist() == [1.0, 2.0, 3.0]
    assert not minimum.v_secondary.flags.writeable
    with pytest.raises(
        ValueError, match=r"v_primary must have shape \(3,\), got \(2,\)"
    ):
        RangeMinimum(0, 1, 10.0, 5.0, r, r[:2], r, r)


# SGP4 on the frozen fixture: the brute-force oracle and SOCRATES.


def fixture_search(
    primaries: list[int],
    start: Time,
    duration: float,
    volume: ScreeningVolume | VolumeRule,
    step: float = DEFAULT_STEP,
) -> tuple[ApproachSearch, list[ScreeningVolume]]:
    """A search of the fixture with some of its objects as primaries, as
    screening does: the apogee/perigee filter, no self pairs. Returns it with
    the primaries' volumes."""
    omms = [BY_NUMBER[number] for number in primaries]
    own, catalog = Sgp4Propagator(omms), Sgp4Propagator(FIXTURE)
    own_bounds = own.radius_bounds(start, duration)
    catalog_bounds = catalog.radius_bounds(start, duration)
    volumes = [volume_for(volume, omm) for omm in omms]
    pairs = apogee_perigee_filter(own_bounds, catalog_bounds, volumes)
    for row, number in enumerate(primaries):
        pairs[row, COLUMN[number]] = False
    found = ApproachSearch(
        own,
        catalog,
        start,
        duration,
        volumes,
        pairs,
        step=step,
        primary_bounds=own_bounds,
        catalog_bounds=catalog_bounds,
    )
    return found, volumes


def range_at(
    primary: int, secondary: int, start: Time, offsets: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    r, _ = positions([BY_NUMBER[primary], BY_NUMBER[secondary]], start, offsets)
    distance: npt.NDArray[np.float64] = np.linalg.norm(r[1] - r[0], axis=-1)
    return distance


def in_rtn_ellipsoid(
    volume: ScreeningVolume,
    r: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    r_other: npt.NDArray[np.float64],
) -> bool:
    """Whether ``r_other`` lies in ``volume`` around a primary at ``r``, ``v``:
    the RTN axes and the ellipsoid worked out here, not by keepout."""
    radial = r / np.linalg.norm(r)
    normal = np.cross(r, v)
    normal /= np.linalg.norm(normal)
    offset = r_other - r
    rtn = np.array(
        [offset @ radial, offset @ np.cross(normal, radial), offset @ normal]
    )
    semi_axes = np.array([volume.radial, volume.transverse, volume.normal])
    return float(np.sum((rtn / semi_axes) ** 2)) <= 1.0


def check_against_oracle(
    primaries: list[int],
    start: Time,
    duration: float,
    volume: ScreeningVolume | VolumeRule,
    step: float,
) -> int:
    """Every minimum the oracle finds inside the volume is found, at the same
    TCA and range; every minimum found is one, inside the volume. Returns the
    number of the oracle's minima inside the volumes."""
    search_, volumes = fixture_search(primaries, start, duration, volume, step)
    result = search_.run()
    omms = [BY_NUMBER[number] for number in primaries]
    # A pass at up to 20 km/s moves 100 km between two samples 10 s apart.
    reach = [v.bounding_radius + 100 * KM for v in volumes]
    oracle = brute_force(omms, FIXTURE, start, duration, reach)
    failed = {primaries[i] for i in result.primary_failures} | {
        FIXTURE[i].catalog_number for i in result.catalog_failures
    }
    assert failed <= oracle.failed
    found: dict[tuple[int, int], list[RangeMinimum]] = {}
    for minimum in result.minima:
        pair = (
            primaries[minimum.primary_index],
            FIXTURE[minimum.secondary_index].catalog_number,
        )
        found.setdefault(pair, []).append(minimum)
    row = {number: i for i, number in enumerate(primaries)}
    inside = 0
    for sampled in oracle.minima:
        if {sampled.primary, sampled.secondary} & oracle.failed:
            continue
        if sampled.distance == 0.0:
            continue  # the same position throughout: one object, no minimum
        volume_ = volumes[row[sampled.primary]]
        if sampled.distance > math.hypot(volume_.bounding_radius, 100 * KM):
            continue
        best = refine(
            BY_NUMBER[sampled.primary], BY_NUMBER[sampled.secondary], oracle, sampled
        )
        if not 1e-3 < best.offset < duration - 1e-3:
            continue  # the range is still falling at an edge of the window
        r, v, _ = states(
            [BY_NUMBER[sampled.primary], BY_NUMBER[sampled.secondary]],
            start,
            [best.offset],
        )
        r, v = r[:, 0], v[:, 0]
        if not in_rtn_ellipsoid(volume_, r[0], v[0], r[1]):
            continue
        inside += 1
        candidates = found.get((sampled.primary, sampled.secondary), [])
        match = min(candidates, key=lambda m: abs(m.offset - best.offset), default=None)
        assert match is not None, best
        # The oracle's TCA of a slow pass is as flat as its range (~30 ms).
        fast = np.linalg.norm(v[1] - v[0]) >= 100.0
        assert match.offset == pytest.approx(best.offset, abs=1e-3 if fast else 0.5)
        assert match.distance == pytest.approx(best.distance, abs=1e-3), best
    for (primary, secondary), minima in found.items():
        for minimum in minima:
            # Far enough to see the range rise: 1e-5 m at 5 km.
            speed = float(np.linalg.norm(minimum.v_secondary - minimum.v_primary))
            delta = min(60.0, max(0.5, 0.5 / speed))
            around = range_at(
                primary,
                secondary,
                start,
                minimum.offset + np.array([-delta, 0.0, delta]),
            )
            assert around[1] == pytest.approx(minimum.distance, abs=1e-6)
            assert around[0] > minimum.distance < around[2]
            assert 0.0 < minimum.offset <= duration
    return inside


volumes = st.one_of(
    st.builds(ScreeningVolume.sphere, st.floats(min_value=1 * KM, max_value=200 * KM)),
    st.builds(
        ScreeningVolume,
        st.floats(min_value=200.0, max_value=200 * KM),
        st.floats(min_value=200.0, max_value=200 * KM),
        st.floats(min_value=200.0, max_value=200 * KM),
    ),
    st.just(ScreeningVolume.sfs_handbook),
)


@settings(deadline=None, max_examples=20)
@given(
    chosen=st.lists(st.sampled_from(CLOSE), min_size=1, max_size=3, unique=True),
    start=st.floats(min_value=-2 * 86400.0, max_value=5 * 86400.0),
    duration=st.floats(min_value=0.5 * HOUR, max_value=12 * HOUR),
    step=st.floats(min_value=30.0, max_value=600.0),
    volume=volumes,
)
def test_no_minimum_the_oracle_finds_is_missed(
    chosen: list[int],
    start: float,
    duration: float,
    step: float,
    volume: ScreeningVolume | VolumeRule,
) -> None:
    inside = check_against_oracle(
        chosen, WEEK_START + start * u.s, duration, volume, step
    )
    # Steer towards windows holding minima, which check the main direction.
    target(float(inside), label="oracle minima inside the volumes")


def test_a_window_with_minima_against_the_oracle() -> None:
    # The examples above may hold no minimum inside the volume; these 6 h
    # hold 22 (formation flyers among them), so the main direction is checked.
    primaries = [44476, 45098, 47826, 52939, 56153]
    inside = check_against_oracle(
        primaries, WEEK_START, 6 * HOUR, DEFAULT_VOLUME, DEFAULT_STEP
    )
    assert inside == 22


def test_every_socrates_approach_of_the_fixture_is_found() -> None:
    # SOCRATES prints TCA to the millisecond and the range to the metre; the
    # TCA of a slow pair is ill-conditioned (a flat minimum of the range).
    first = min(approach.tca for approach in SOCRATES)
    start = first - HOUR * u.s
    duration = (max(approach.tca for approach in SOCRATES) - start).sec + HOUR
    primaries = sorted({approach.primary for approach in SOCRATES})
    result = fixture_search(primaries, start, duration, DEFAULT_VOLUME)[0].run()
    found: dict[tuple[int, int], list[RangeMinimum]] = {}
    for minimum in result.minima:
        pair = (
            primaries[minimum.primary_index],
            FIXTURE[minimum.secondary_index].catalog_number,
        )
        found.setdefault(pair, []).append(minimum)
    assert len(SOCRATES) == 1033
    for approach in SOCRATES:
        offset = (approach.tca - start).sec
        candidates = found[(approach.primary, approach.secondary)]
        match = min(candidates, key=lambda m: abs(m.offset - offset))
        fast = approach.relative_speed >= FAST_SPEED
        assert match.offset == pytest.approx(offset, abs=1e-3 if fast else 0.5)
        assert match.distance == pytest.approx(approach.miss_distance, abs=0.5)


def test_a_docked_spacecraft_has_no_minimum() -> None:
    # CelesTrak gives docked spacecraft their station's element set.
    station = PRIMARIES[0]
    docked = dataclasses.replace(station, catalog_number=99999)
    own, catalog = Sgp4Propagator([station]), Sgp4Propagator([docked, PRIMARIES[1]])
    result = ApproachSearch(
        own,
        catalog,
        WEEK_START,
        86400.0,
        DEFAULT_VOLUME,
        np.array([[True, False]]),
        primary_bounds=own.radius_bounds(WEEK_START, 86400.0),
        catalog_bounds=catalog.radius_bounds(WEEK_START, 86400.0),
    ).run()
    assert summary(result) == summary(SearchResult((), {}, {}))


def test_sgp4_results_do_not_depend_on_the_chunks() -> None:
    # Formation flyers (many slow minima) and a few fast passes.
    primaries = [68377, 56156, 58199, *(omm.catalog_number for omm in PRIMARIES[:3])]
    search_, _ = fixture_search(primaries, WEEK_START, 86400.0, DEFAULT_VOLUME)
    whole = search_.run()
    assert len(whole.minima) > 50
    for max_states in (5000, 1):
        assert summary(search_.run(max_states=max_states)) == summary(whole)


@pytest.mark.validation
def test_a_week_of_the_fixture_against_the_oracle() -> None:
    # Offline, about 20 s: 25 primaries, every 10 s of the week.
    inside = check_against_oracle(
        [omm.catalog_number for omm in PRIMARIES],
        WEEK_START,
        7 * 86400.0,
        DEFAULT_VOLUME,
        DEFAULT_STEP,
    )
    assert inside == 267
