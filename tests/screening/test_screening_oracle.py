"""The brute-force oracle: SGP4's positions, every sampled minimum, and
SOCRATES's fast approaches to the precision the report prints."""

import gzip

import astropy.units as u
import numpy as np
import pytest
import screening_snapshot as snapshot
from astropy.time import Time
from screening_oracle import (
    ORACLE_STEP,
    brute_force,
    local_minima,
    positions,
    refine,
)

from keepout.omm import load
from keepout.propagation import Sgp4Propagator

FIXTURE = load(snapshot.DATA / snapshot.FIXTURE)
BY_NUMBER = {omm.catalog_number: omm for omm in FIXTURE}
START = Time("2026-10-06T00:00:00", scale="utc")
FAILING = {45665, 45711, 270365}
"""Fixture objects SGP4 fails on within a week of START."""

FAST = [
    approach
    for approach in snapshot.read_socrates(
        gzip.decompress((snapshot.DATA / snapshot.SOCRATES).read_bytes()).decode()
    )
    if approach.primary in BY_NUMBER
    and approach.secondary in BY_NUMBER
    and approach.relative_speed >= snapshot.FAST
]
"""The fast (1 km/s or more) SOCRATES approaches of two fixture objects."""


def test_positions_are_sgp4s() -> None:
    omms = FIXTURE[:40] + tuple(BY_NUMBER[number] for number in sorted(FAILING))
    offsets = np.linspace(-86400.0, 7 * 86400.0, 33)
    r, errors = positions(omms, START, offsets)
    ephemeris = Sgp4Propagator(omms).propagate(START, offsets)
    np.testing.assert_array_equal(errors, ephemeris.errors)
    np.testing.assert_allclose(r, ephemeris.r, rtol=0, atol=1e-5)
    assert np.all(np.isnan(r[errors != 0]))
    empty, no_errors = positions([], START, offsets)
    assert empty.shape == (0, 33, 3)
    assert no_errors.shape == (0, 33)


@pytest.mark.parametrize(
    ("distances", "minima"),
    [
        ([3.0, 1.0, 2.0, 0.0, 4.0], [False, True, False, True, False]),
        ([1.0, 2.0, 3.0], [True, False, False]),
        ([3.0, 2.0, 1.0], [False, False, True]),
        ([3.0, 1.0, 1.0, 2.0], [False, True, False, False]),
        ([5.0], [True]),
    ],
)
def test_local_minima(distances: list[float], minima: list[bool]) -> None:
    np.testing.assert_array_equal(local_minima(np.array(distances)), minima)
    stacked = np.array([distances, distances])
    np.testing.assert_array_equal(local_minima(stacked), [minima, minima])


def test_the_fixture_holds_fast_socrates_approaches() -> None:
    assert len(FAST) == 31


@pytest.mark.parametrize(
    "approach", FAST, ids=lambda a: f"{a.primary}-{a.secondary}-{a.tca.isot}"
)
def test_refined_minima_are_socrates_approaches(
    approach: snapshot.Approach,
) -> None:
    # SOCRATES prints TCA to the millisecond and the range to the metre.
    primary = BY_NUMBER[approach.primary]
    secondary = BY_NUMBER[approach.secondary]
    start = approach.tca - 600.0 * u.s
    oracle = brute_force([primary], [secondary], start, 1200.0, 6e3)
    (minimum,) = oracle.minima
    assert not minimum.at_edge
    assert minimum.distance >= approach.miss_distance - 1.0
    refined = refine(primary, secondary, oracle, minimum)
    assert abs(refined.offset - 600.0) <= 1e-3
    assert refined.distance == pytest.approx(approach.miss_distance, abs=1.0)
    assert refined.distance <= minimum.distance
    assert (refined.primary, refined.secondary, refined.sample) == (
        minimum.primary,
        minimum.secondary,
        minimum.sample,
    )


def test_refined_minima_are_minima_late_in_a_window() -> None:
    # Bounded Brent's tolerance grows with its variable (sqrt(eps) |x|, 8 ms
    # at 6 days): refining in the time since the earlier neighbour keeps it
    # at the microsecond. A pass at 14.8 km/s, 6 days into the window.
    primary, secondary = BY_NUMBER[100054], BY_NUMBER[60746]
    oracle = brute_force([primary], [secondary], START, 7 * 86400.0, 100e3)
    (minimum,) = (m for m in oracle.minima if abs(m.offset - 526453.6) < 10.0)
    refined = refine(primary, secondary, oracle, minimum)
    assert refined.distance == pytest.approx(2479.5205, abs=1e-4)
    r, _ = positions(
        [primary, secondary], START, refined.offset + np.array([-1e-3, 1e-3])
    )
    assert np.all(np.linalg.norm(r[1] - r[0], axis=-1) > refined.distance)


def test_failing_objects_and_self_pairs_are_left_out() -> None:
    primaries = [BY_NUMBER[45665], FIXTURE[0], FIXTURE[100]]
    oracle = brute_force(primaries, FIXTURE, START, 7 * 86400.0, 1e9, step=3600.0)
    assert oracle.failed == FAILING
    assert len(oracle.offsets) == 169
    pairs = oracle.pairs
    assert {pair[0] for pair in pairs} == {
        FIXTURE[0].catalog_number,
        FIXTURE[100].catalog_number,
    }
    assert not any(secondary in FAILING for _, secondary in pairs)
    assert not any(primary == secondary for primary, secondary in pairs)
    # Every other pair has a closest sample, a minimum within any reach.
    assert len(pairs) == 2 * (len(FIXTURE) - len(FAILING) - 1)


def test_each_primary_has_its_own_reach() -> None:
    primaries = [FIXTURE[0], FIXTURE[100]]
    oracle = brute_force(primaries, FIXTURE, START, 86400.0, [1e6, 5e6], step=600.0)
    reach = {FIXTURE[0].catalog_number: 1e6, FIXTURE[100].catalog_number: 5e6}
    assert oracle.minima
    assert all(m.distance <= reach[m.primary] for m in oracle.minima)
    assert oracle.within(1e6) == tuple(m for m in oracle.minima if m.distance <= 1e6)
    assert oracle.minima == tuple(
        sorted(oracle.minima, key=lambda m: (m.primary, m.secondary, m.sample))
    )


def test_minima_at_the_edges_of_the_window() -> None:
    approach = FAST[0]
    primary = BY_NUMBER[approach.primary]
    secondary = BY_NUMBER[approach.secondary]
    # The window opens at TCA: the distance only grows after it.
    oracle = brute_force([primary], [secondary], approach.tca, 600.0, 1e5)
    (minimum,) = oracle.minima
    assert (minimum.sample, minimum.at_edge) == (0, True)
    assert oracle.offsets[1] - oracle.offsets[0] == pytest.approx(ORACLE_STEP)
    # One sample: the pair is a minimum there, and refining keeps it.
    instant = brute_force([primary], [secondary], approach.tca, 0.0, 1e5)
    (alone,) = instant.minima
    assert (len(instant.offsets), alone.at_edge) == (1, True)
    assert refine(primary, secondary, instant, alone) == alone
