"""Screening end to end: the search's minima as close approaches in GCRF, the
element sets left out and why, and the same result in any number of
processes."""

import dataclasses
import functools
import math
from collections.abc import Callable, Sequence

import astropy.units as u
import numpy as np
import numpy.typing as npt
import pytest
from astropy.time import Time
from screening_data import (
    DATA,
    FAST_SPEED,
    FIXTURE_START,
    SGP4_FAILING,
    SNAPSHOT,
    fixture_by_number,
    load_fixture,
    socrates_approaches,
)

from keepout.errors import GeometryError, PropagationError, UnsupportedElements
from keepout.frames import teme_to_gcrf
from keepout.omm import Omm, load
from keepout.propagation import (
    CowellPropagator,
    ForceModel,
    ManeuveredPropagator,
    Propagator,
    RadiusBounds,
    Sgp4Propagator,
    describe_error,
    earth_pole,
    satrec_of,
)
from keepout.screening import (
    DEFAULT_DURATION,
    DEFAULT_VOLUME,
    MAX_AGE,
    MAX_LEAD,
    ApproachSearch,
    RangeMinimum,
    ScreeningResult,
    ScreeningVolume,
    SkipReason,
    apogee_perigee_filter,
    screen,
)
from keepout.screening.tca import DERIVATIVE_STEP, PERTURBATION
from keepout.units import DAY, HOUR, KM

START = FIXTURE_START
FIXTURE = load_fixture()
BY_NUMBER = fixture_by_number()
FORMATION = [68377, 56156, 58199]
"""Formation flyers: many slow approaches."""
SPREAD = [omm.catalog_number for omm in FIXTURE[::9][:5]]
"""LEO, GEO and HEO objects of the fixture."""
FAILING = [45665, 45711]
"""SGP4 fails for them 41 h 34 min 43 s and 23 h 12 min after START (error 1);
each is the other's only neighbour."""
STALE = 32360
"""Its elements are 29.3 days old at START."""
DENSE_STEP = 5.0
"""Time between the samples of a dense scan of a pair [s]."""
REENTERING = 270365
"""An orbit with its perigee 113 km up: SGP4's drag on it exceeds the search's
bound on perturbations from 8 days after its epoch; SGP4 fails at 10.5 days."""


def elements(numbers: Sequence[int]) -> list[Omm]:
    return [BY_NUMBER[number] for number in numbers]


def without(*numbers: int) -> list[Omm]:
    return [omm for omm in FIXTURE if omm.catalog_number not in numbers]


def approaches_of(result: ScreeningResult) -> list[object]:
    """The approaches of a result, exact, for comparisons."""
    return [
        (
            a.primary_number,
            a.secondary_number,
            a.tca.jd1,
            a.tca.jd2,
            a.primary_state.r.tolist(),
            a.primary_state.v.tolist(),
            a.secondary_state.r.tolist(),
            a.secondary_state.v.tolist(),
        )
        for a in result.approaches
    ]


def summary(result: ScreeningResult) -> list[object]:
    """Everything in a result, exact, for comparisons."""
    skipped = [(s.elements, s.role, s.reason, s.detail) for s in result.skipped]
    return [
        *approaches_of(result),
        skipped,
        dict(result.volumes),
        result.catalog_count,
        result.duration,
    ]


def test_approaches_are_the_minima_of_the_search_in_gcrf() -> None:
    primaries = elements([*FORMATION, *SPREAD])
    result = screen(primaries, FIXTURE, START, DAY)
    # The search, by hand, on the catalog screening kept.
    left_out = {s.catalog_number for s in result.skipped if s.role == "catalog"}
    assert left_out == {STALE}
    catalog = without(*left_out)
    own, others = Sgp4Propagator(primaries), Sgp4Propagator(catalog)
    own_bounds = own.radius_bounds(START, DAY)
    other_bounds = others.radius_bounds(START, DAY)
    pairs = apogee_perigee_filter(own_bounds, other_bounds, DEFAULT_VOLUME)
    numbers = [omm.catalog_number for omm in catalog]
    for row, omm in enumerate(primaries):
        pairs[row, numbers.index(omm.catalog_number)] = False
    found = ApproachSearch(
        own,
        others,
        START,
        DAY,
        DEFAULT_VOLUME,
        pairs,
        primary_bounds=own_bounds,
        catalog_bounds=other_bounds,
    ).run()

    def ids(m: RangeMinimum) -> tuple[int, int]:
        return primaries[m.primary_index].catalog_number, numbers[m.secondary_index]

    minima = sorted(found.minima, key=lambda m: (m.offset, *ids(m)))
    assert len(result.approaches) == len(minima) == 122
    for approach, minimum in zip(result.approaches, minima, strict=True):
        assert (approach.primary_number, approach.secondary_number) == ids(minimum)
        tca = START + minimum.offset * u.s
        assert abs((approach.tca - tca).to_value(u.s)) < 1e-9
        assert 0.0 < minimum.offset <= DAY
        r, v = teme_to_gcrf(
            tca,
            np.array([minimum.r_primary, minimum.r_secondary]),
            np.array([minimum.v_primary, minimum.v_secondary]),
        )
        for state, r_gcrf, v_gcrf in zip(
            (approach.primary_state, approach.secondary_state), r, v, strict=True
        ):
            assert state.frame == "GCRF"
            assert state.epoch is approach.primary_state.epoch
            np.testing.assert_allclose(state.r, r_gcrf, rtol=0.0, atol=1e-8)
            np.testing.assert_allclose(state.v, v_gcrf, rtol=0.0, atol=1e-11)
        assert approach.range_at_tca == pytest.approx(minimum.range_at_tca, abs=1e-8)
    assert dict(result.volumes) == dict.fromkeys(
        (omm.catalog_number for omm in primaries), DEFAULT_VOLUME
    )
    assert result.catalog_count == len(FIXTURE) - 1
    assert (result.start, result.duration) == (START, DAY)


def test_a_primary_is_not_screened_against_itself() -> None:
    # Not even against another element set of its own: by catalog number.
    moved = dataclasses.replace(
        BY_NUMBER[68377], mean_anomaly=BY_NUMBER[68377].mean_anomaly + 1e-4
    )
    for primary in (BY_NUMBER[68377], moved):
        result = screen([primary], FIXTURE, START, DAY)
        assert {a.secondary_number for a in result.approaches} == {68378}
        assert all(a.primary_number == 68377 for a in result.approaches)


def test_the_newest_of_several_element_sets_is_screened() -> None:
    older = [
        dataclasses.replace(
            BY_NUMBER[number],
            epoch=BY_NUMBER[number].epoch - 0.5 * u.day,
            mean_anomaly=0.5,
        )
        for number in (56156, 58199)
    ]
    plain = screen(elements(FORMATION), FIXTURE, START, DAY)
    result = screen(
        [older[1], *elements(FORMATION), BY_NUMBER[68377]],
        [*FIXTURE, older[0], BY_NUMBER[68377]],
        START,
        DAY,
    )
    duplicates = [s for s in result.skipped if s.reason is SkipReason.DUPLICATE]
    assert [(s.role, s.catalog_number) for s in duplicates] == [
        ("primary", 58199),
        ("catalog", 56156),
    ]
    others = [s for s in result.skipped if s.reason is not SkipReason.DUPLICATE]
    assert others == list(plain.skipped)
    duplicate = duplicates[1]
    assert duplicate.elements is older[0]
    assert duplicate.detail == (
        f"epoch {older[0].epoch.isot}; the element set of epoch "
        f"{BY_NUMBER[56156].epoch.isot} is used"
    )
    assert approaches_of(result) == approaches_of(plain)
    assert list(result.volumes) == [58199, 68377, 56156]  # as first given


def test_element_sets_too_far_from_the_window_are_left_out() -> None:
    max_age, max_lead = 7 * DAY, 0.2 * DAY
    end = START + DAY * u.s
    ages = {o.catalog_number: (end - o.epoch).to_value(u.s) for o in FIXTURE}
    leads = {o.catalog_number: (o.epoch - START).to_value(u.s) for o in FIXTURE}
    stale = {n for n, age in ages.items() if age > max_age}
    ahead = {n for n, lead in leads.items() if lead > max_lead} - stale
    assert (len(stale), len(ahead)) == (19, 46)
    assert all(abs(age - max_age) > 1.0 for age in ages.values())
    assert all(abs(lead - max_lead) > 1.0 for lead in leads.values())
    primaries = elements([STALE, *FORMATION])
    result = screen(primaries, FIXTURE, START, DAY, max_age=max_age, max_lead=max_lead)
    catalog = [s for s in result.skipped if s.role == "catalog"]
    assert {s.catalog_number for s in catalog if s.reason is SkipReason.STALE} == stale
    assert {
        s.catalog_number for s in catalog if s.reason is SkipReason.FUTURE_EPOCH
    } == ahead
    assert {s.reason for s in catalog} <= {
        SkipReason.STALE,
        SkipReason.FUTURE_EPOCH,
        SkipReason.PROPAGATION_FAILED,
    }
    (first, *_) = result.skipped
    assert (first.role, first.catalog_number, first.reason) == (
        "primary",
        STALE,
        SkipReason.STALE,
    )
    assert first.detail == (
        f"the window ends {ages[STALE] / DAY:.2f} days after the epoch "
        f"{BY_NUMBER[STALE].epoch.isot} (max_age 7 days)"
    )
    future = next(s for s in catalog if s.reason is SkipReason.FUTURE_EPOCH)
    assert future.detail == (
        f"the window starts {leads[future.catalog_number] / DAY:.2f} days before "
        f"the epoch {future.elements.epoch.isot} (max_lead 0.2 days)"
    )
    assert STALE not in result.volumes
    left_out = stale | ahead
    assert not {a.secondary_number for a in result.approaches} & left_out
    assert result.approaches


@pytest.mark.parametrize(
    ("limit", "reason"),
    [("max_age", SkipReason.STALE), ("max_lead", SkipReason.FUTURE_EPOCH)],
)
def test_an_element_set_exactly_at_a_limit_is_screened(
    limit: str, reason: SkipReason
) -> None:
    # Left out only when further than the limit: a day's window from the
    # epoch ends a day after it, one from a day before starts a day ahead
    # (whole days of a Time, so exactly 86400 s).
    primary = BY_NUMBER[FORMATION[0]]
    start = primary.epoch if limit == "max_age" else primary.epoch - 1 * u.day
    assert (primary.epoch - start).to_value(u.s) in (0.0, DAY)
    loose = {"max_age": 30 * DAY, "max_lead": 30 * DAY}
    catalog = elements(FORMATION[1:])

    def skipped(at: float) -> list[SkipReason]:
        limits = loose | {limit: at}
        result = screen([primary], catalog, start, DAY, **limits)  # type: ignore[arg-type]
        return [s.reason for s in result.skipped if s.role == "primary"]

    assert skipped(DAY) == []
    assert skipped(math.nextafter(DAY, 0.0)) == [reason]


def test_default_ages_keep_sgp4_within_the_searchs_assumption() -> None:
    # Acceleration beyond point mass and J2 from MAX_LEAD before each epoch
    # to MAX_AGE after it, every 12 hours, from positions 1 s apart, until
    # SGP4 fails (a window holding the failure skips the object).
    mu, re, j2 = 398600.8e9, 6378.135e3, 0.001082616  # SGP4's WGS 72
    worst: dict[int, float] = {}
    for omm in FIXTURE:
        satrec = satrec_of(omm)
        worst[omm.catalog_number] = 0.0
        for days in np.arange(-MAX_LEAD / DAY, MAX_AGE / DAY + 0.25, 0.5):
            minutes = days * 1440.0 + np.array([-1.0, 0.0, 1.0]) / 60.0
            states = [satrec.sgp4_tsince(minute) for minute in minutes]
            if any(error for error, _, _ in states):
                break
            r = [np.array(position) * 1e3 for _, position, _ in states]
            distance = np.linalg.norm(r[1])
            z2 = (r[1][2] / distance) ** 2
            zonal = np.array([1 - 5 * z2, 1 - 5 * z2, 3 - 5 * z2])
            gravity = -mu * r[1] / distance**3
            gravity -= 1.5 * j2 * mu * re**2 / distance**5 * r[1] * zonal
            beyond = float(np.linalg.norm(r[2] - 2 * r[1] + r[0] - gravity))
            worst[omm.catalog_number] = max(worst[omm.catalog_number], beyond)
    assert worst.pop(REENTERING) > PERTURBATION
    assert max(worst.values()) < 0.25 * PERTURBATION


def test_objects_sgp4_fails_for_are_left_out_with_their_approaches() -> None:
    result = screen(elements([*FORMATION, *FAILING]), FIXTURE, START, 2 * DAY)
    failed = [s for s in result.skipped if s.reason is SkipReason.PROPAGATION_FAILED]
    when = ["2026-10-07T17:36:00.000", "2026-10-06T23:12:00.000"]
    assert [(s.role, s.catalog_number, s.detail) for s in failed] == [
        (role, number, f"{describe_error(1)}, at {at}")
        for role in ("primary", "catalog")
        for number, at in zip(FAILING, when, strict=True)
    ]
    assert not set(FAILING) & set(result.volumes)
    rest = screen(elements(FORMATION), without(*FAILING), START, 2 * DAY)
    assert approaches_of(result) == approaches_of(rest)
    assert result.catalog_count == rest.catalog_count


def test_an_object_failing_just_after_the_window_is_screened() -> None:
    # SGP4 first fails for 45711 between 83478.5 and 83478.75 s after START;
    # the search's derivative at the window's end reaches 2 s past it. With
    # a sphere wide enough to hold the pair, its minimum 0.23 s before the
    # end is found, a minimum of the range, and nothing is left out.
    duration = 83477.5
    result = screen(
        elements(FAILING[:1]),
        elements(FAILING[1:]),
        START,
        duration,
        ScreeningVolume.sphere(5000e3),
    )
    assert result.skipped == ()
    assert len(result.approaches) == 33
    last = result.approaches[-1]
    offset = (last.tca - START).to_value(u.s)
    assert duration - 1.0 < offset <= duration
    ephemeris = Sgp4Propagator(elements(FAILING)).propagate(
        START, offset + np.array([-1e-3, 0.0, 1e-3])
    )
    assert ephemeris.valid.all()
    distance = np.linalg.norm(ephemeris.r[1] - ephemeris.r[0], axis=-1)
    assert distance[0] > distance[1] < distance[2]
    assert np.linalg.norm(
        last.secondary_state.r - last.primary_state.r
    ) == pytest.approx(distance[1], abs=1e-6)


def test_element_sets_sgp4_cannot_take_are_left_out() -> None:
    big = dataclasses.replace(BY_NUMBER[68378], catalog_number=400_000)
    result = screen([big, *elements(FORMATION)], [*FIXTURE, big], START, DAY)
    unsupported = [s for s in result.skipped if s.reason is SkipReason.UNSUPPORTED]
    assert [(s.role, s.elements) for s in unsupported] == [
        ("primary", big),
        ("catalog", big),
    ]
    assert unsupported[0].detail == (
        "catalog number 400000 is above 339999, the largest an SGP4 record holds"
    )
    plain = screen(elements(FORMATION), FIXTURE, START, DAY)
    assert approaches_of(result) == approaches_of(plain)
    assert result.catalog_count == plain.catalog_count


def picky(error: Exception) -> Callable[[Omm], ScreeningVolume]:
    def rule(primary: Omm) -> ScreeningVolume:
        if primary.catalog_number == 56156:
            raise error
        return ScreeningVolume.nasa_gp_data()

    return rule


@pytest.mark.parametrize(
    "error",
    [GeometryError("no volume for 56156"), UnsupportedElements("no volume for 56156")],
)
def test_a_primary_without_a_volume_is_left_out(error: Exception) -> None:
    result = screen(elements(FORMATION), FIXTURE, START, DAY, picky(error))
    (refused,) = [s for s in result.skipped if s.role == "primary"]
    assert (refused.catalog_number, refused.reason, refused.detail) == (
        56156,
        SkipReason.NO_VOLUME,
        "no volume for 56156",
    )
    assert dict(result.volumes) == {
        68377: ScreeningVolume.nasa_gp_data(),
        58199: ScreeningVolume.nasa_gp_data(),
    }
    others = screen(
        elements([68377, 58199]), FIXTURE, START, DAY, ScreeningVolume.nasa_gp_data()
    )
    assert approaches_of(result) == approaches_of(others)


@pytest.mark.parametrize(
    "error", [ZeroDivisionError("division by zero"), ValueError("a bug in the rule")]
)
def test_a_volume_rule_that_fails_otherwise_stops_the_screening(
    error: Exception,
) -> None:
    # Only a KeepoutError says the rule has no volume for a primary; any
    # other exception is a bug in the rule, not a reason to skip.
    with pytest.raises(type(error), match=str(error)):
        screen(elements(FORMATION), FIXTURE, START, DAY, picky(error))


def test_a_volume_rule_must_give_volumes() -> None:
    with pytest.raises(TypeError, match="ScreeningVolume"):
        screen(elements(FORMATION), FIXTURE, START, DAY, lambda primary: 5e3)  # type: ignore[arg-type,return-value]


def test_every_socrates_approach_of_the_fixture_is_found_in_gcrf() -> None:
    # About 4 s. SOCRATES prints TCA to the millisecond, range and
    # relative speed to the metre (per second); a slow pair's TCA is
    # ill-conditioned.
    socrates = [
        approach
        for approach in socrates_approaches()
        if approach.primary_number in BY_NUMBER
        and approach.secondary_number in BY_NUMBER
    ]
    start = min(approach.tca for approach in socrates) - 1 * u.hour
    duration = (max(approach.tca for approach in socrates) - start).sec + 3600.0
    primaries = sorted({approach.primary_number for approach in socrates})
    result = screen(elements(primaries), FIXTURE, start, duration)
    found: dict[tuple[int, int], list[npt.NDArray[np.float64]]] = {}
    for a in result.approaches:
        row = [(a.tca - start).to_value(u.s), a.range_at_tca, a.relative_speed]
        found.setdefault((a.primary_number, a.secondary_number), []).append(
            np.array(row)
        )
    assert len(socrates) == 1033
    for approach in socrates:
        offset = (approach.tca - start).sec
        match = min(
            found[approach.primary_number, approach.secondary_number],
            key=lambda m: abs(m[0] - offset),
        )
        fast = approach.relative_speed >= FAST_SPEED
        assert match[0] == pytest.approx(offset, abs=1e-3 if fast else 0.5)
        assert match[1] == pytest.approx(approach.range_at_tca, abs=0.5)
        assert match[2] == pytest.approx(approach.relative_speed, abs=0.5)


def mixed() -> tuple[list[Omm], list[Omm]]:
    """Primaries and a catalog with every reason to leave an element set out."""
    big = dataclasses.replace(BY_NUMBER[68378], catalog_number=400_000)
    older = dataclasses.replace(
        BY_NUMBER[56156], epoch=BY_NUMBER[56156].epoch - 1 * u.day, mean_anomaly=1.0
    )
    primaries = elements([*FORMATION, *SPREAD, *FAILING, STALE])
    return [big, *primaries], [older, *FIXTURE, big]


@pytest.mark.parametrize(
    ("workers", "max_states"), [(1, 700), (2, 1_000_000), (3, 5000)]
)
def test_the_result_does_not_depend_on_workers_or_chunks(
    workers: int, max_states: int
) -> None:
    primaries, catalog = mixed()
    window = (START, 2 * DAY)
    whole = screen(primaries, catalog, *window)
    assert {s.reason for s in whole.skipped} == set(SkipReason) - {
        SkipReason.NO_VOLUME,
        SkipReason.FUTURE_EPOCH,
    }
    assert len(whole.approaches) > 150
    split = screen(primaries, catalog, *window, workers=workers, max_states=max_states)
    assert summary(split) == summary(whole)


def frame_to_gcrf_of(primaries: Propagator) -> tuple[npt.NDArray[np.float64], ...]:
    """Positions and velocities at START in GCRF, each ``(N, 3)``."""
    states = primaries.propagate(START, 0.0)
    return teme_to_gcrf(START, states.r[:, 0], states.v[:, 0])


def hook_burning(
    delta_v: float, burn_time: Time, horizon: float
) -> Callable[[Propagator], Propagator]:
    """A hook for one primary: ``delta_v`` along its velocity at ``burn_time``."""
    return functools.partial(
        ManeuveredPropagator, burn_time=burn_time, delta_v=delta_v, horizon=horizon
    )


@pytest.mark.parametrize("workers", [1, 2])
def test_screen_with_identity_hook_equals_screen(workers: int) -> None:
    primaries, catalog = mixed()
    window = (START, 2 * DAY)
    whole = screen(primaries, catalog, *window)
    assert len(whole.approaches) > 150
    same = screen(primaries, catalog, *window, workers=workers, adjust_primaries=_same)
    assert summary(same) == summary(whole)


def _same(primaries: Propagator) -> Propagator:
    return primaries


@pytest.mark.parametrize("burn_at", [0.0, DAY / 2.0])
@pytest.mark.parametrize("workers", [1, 2])
def test_screen_with_zero_burn_equals_screen(burn_at: float, workers: int) -> None:
    primary = elements(FORMATION[:1])
    whole = screen(primary, FIXTURE, START, DAY)
    assert len(whole.approaches) > 5
    hook = hook_burning(0.0, START + burn_at * u.s, 2 * DAY)
    burned = screen(
        primary, FIXTURE, START, DAY, workers=workers, adjust_primaries=hook
    )
    assert summary(burned) == summary(whole)


@pytest.mark.parametrize("workers", [1, 2])
def test_a_burn_moves_the_primary_in_the_approaches(workers: int) -> None:
    primary = elements(FORMATION[:1])
    delta_v = 0.05
    horizon = DAY + 2 * DERIVATIVE_STEP
    hook = hook_burning(delta_v, START, horizon)
    burned = screen(
        primary, FIXTURE, START, DAY, workers=workers, adjust_primaries=hook
    )
    assert burned.approaches
    plain = Sgp4Propagator(primary)
    moved = ManeuveredPropagator(plain, START, delta_v, horizon)
    shifts = []
    for approach in burned.approaches:
        offset = (approach.tca - START).to_value(u.s)
        state = moved.propagate(START, offset)
        r, v = teme_to_gcrf(approach.tca, state.r[0, 0], state.v[0, 0])
        assert approach.primary_state.r == pytest.approx(r, abs=1e-3)
        assert approach.primary_state.v == pytest.approx(v, abs=1e-6)
        unburned = plain.propagate(START, offset)
        r, _ = teme_to_gcrf(approach.tca, unburned.r[0, 0], unburned.v[0, 0])
        shifts.append(np.linalg.norm(approach.primary_state.r - r))
    # The burn's along-track drift, 3 dv t: 5 km by the last approach, 9 h in.
    assert max(shifts) > 4 * KM


def dense_minima(
    primary: Propagator,
    catalog: Sequence[Omm],
    duration: float,
    volume: ScreeningVolume = DEFAULT_VOLUME,
) -> list[tuple[int, float, float]]:
    """Every local minimum of the range inside ``volume`` in a window, by
    scanning each pair: catalog number, TCA and range, without the search."""
    others = Sgp4Propagator(catalog)
    grid = np.arange(0.0, duration + DENSE_STEP, DENSE_STEP)
    r_primary = primary.propagate(START, grid).r[0]
    ranges = np.linalg.norm(others.propagate(START, grid).r - r_primary, axis=2)
    reach = volume.bounding_radius + DENSE_STEP * 8_000.0  # half a step at 16 km/s
    falling = ranges[:, :-2] > ranges[:, 1:-1]
    rising = ranges[:, 1:-1] <= ranges[:, 2:]
    rows, samples = np.nonzero(falling & rising & (ranges[:, 1:-1] < reach))
    found = []
    for row, sample in zip(rows.tolist(), (samples + 1).tolist(), strict=True):
        middle = float(grid[sample])
        fine = np.linspace(middle - DENSE_STEP, middle + DENSE_STEP, 2001)
        own = primary.propagate(START, fine)
        other = others.subset(np.array([row])).propagate(START, fine)
        distance = np.linalg.norm(other.r[0] - own.r[0], axis=1)
        best = int(np.argmin(distance))
        tca = float(fine[best])
        if 0.0 < tca <= duration and volume.contains_position(
            own.r[0, best], own.v[0, best], other.r[0, best]
        ):
            found.append((catalog[row].catalog_number, tca, float(distance[best])))
    return sorted(found, key=lambda minimum: minimum[1])


@pytest.mark.parametrize(
    ("number", "delta_v", "burn_at"),
    [
        (FORMATION[0], 0.05, 0.0),
        (FORMATION[0], 0.05, DAY / 2.0),
        (FORMATION[0], -1.0, DAY / 2.0 + 37.0),
        (FORMATION[1], 0.5, 3 * HOUR + 11.0),
        (FORMATION[2], 1.0, DAY / 3.0),
    ],
)
def test_a_burned_screening_misses_no_minimum_of_a_dense_scan(
    number: int, delta_v: float, burn_at: float
) -> None:
    primary = elements([number])
    catalog = [o for o in without(STALE, *SGP4_FAILING) if o.catalog_number != number]
    burn_time = START + burn_at * u.s
    moved = ManeuveredPropagator(
        Sgp4Propagator(primary), burn_time, delta_v, DAY + 2 * DERIVATIVE_STEP
    )
    result = screen(primary, catalog, START, DAY, adjust_primaries=lambda _: moved)
    assert [s for s in result.skipped if s.role == "catalog"] == []
    assert_same_minima(result, dense_minima(moved, catalog, DAY))


def assert_same_minima(
    result: ScreeningResult, expected: Sequence[tuple[int, float, float]]
) -> None:
    assert expected, "no approach to check"
    found = sorted(
        (a.secondary_number, float((a.tca - START).to_value(u.s)), a.range_at_tca)
        for a in result.approaches
    )
    assert [minimum[0] for minimum in found] == sorted(m[0] for m in expected)
    for (_, tca, distance), (_, tca_dense, range_dense) in zip(
        found, sorted(expected), strict=True
    ):
        assert tca == pytest.approx(tca_dense, abs=5.0)
        assert distance == pytest.approx(range_dense, abs=1.0)


def test_a_burn_brings_in_objects_the_unburned_orbit_cannot_reach() -> None:
    number, delta_v, burn_at = 56156, 10.0, DAY / 3.0
    # Only objects whose radii keep them from the orbit without the burn but
    # not from the one with it: if the bounds did not widen, all are filtered.
    volume = ScreeningVolume(radial=30e3, transverse=30e3, normal=30e3)
    duration = 2 * DAY
    catalog = load(DATA / SNAPSHOT)
    primary = next(omm for omm in catalog if omm.catalog_number == number)
    base = Sgp4Propagator([primary])
    burn_time = START + burn_at * u.s
    horizon = duration + 2 * DERIVATIVE_STEP
    moved = ManeuveredPropagator(base, burn_time, delta_v, horizon)
    others, _, _ = Sgp4Propagator.from_omms(catalog)
    their_bounds = others.radius_bounds(START, duration)
    radii = np.linalg.norm(
        moved.propagate(START, np.arange(0.0, duration, 20.0)).r, axis=2
    )
    actual = RadiusBounds(radii.min(axis=1), radii.max(axis=1))
    reachable = apogee_perigee_filter(actual, their_bounds, volume)[0]
    before = apogee_perigee_filter(
        base.radius_bounds(START, duration), their_bounds, volume
    )[0]
    chosen = [
        omm
        for omm, new in zip(others.omms, reachable & ~before, strict=True)
        if new and omm.catalog_number != number
    ]
    assert not {omm.catalog_number for omm in chosen} & set(SGP4_FAILING)
    result = screen(
        [primary],
        chosen,
        START,
        duration,
        volume,
        adjust_primaries=lambda _: moved,
    )
    kept = {s.catalog_number for s in result.skipped}
    scanned = [omm for omm in chosen if omm.catalog_number not in kept]
    assert_same_minima(result, dense_minima(moved, scanned, duration, volume))


def test_a_hook_short_of_the_search_is_refused_on_any_data() -> None:
    # No minimum near the end of this window: the search alone would not see it.
    primary = elements(FORMATION[:1])
    short = hook_burning(0.05, START, DAY)
    with pytest.raises(ValueError, match=r"2\.0 s past the end of the window"):
        screen(primary, [], START, DAY, adjust_primaries=short)
    enough = hook_burning(0.05, START, DAY + 2 * DERIVATIVE_STEP)
    assert screen(primary, FIXTURE, START, DAY, adjust_primaries=enough).approaches


def test_the_hook_is_not_called_without_primaries() -> None:
    calls: list[Propagator] = []

    def hook(primaries: Propagator) -> Propagator:
        calls.append(primaries)
        return primaries

    stale = screen(elements([STALE]), FIXTURE, START, DAY, adjust_primaries=hook)
    assert calls == []
    assert [s.reason for s in stale.skipped if s.role == "primary"] == [
        SkipReason.STALE
    ]


def test_the_hook_gets_the_primaries_that_are_screened() -> None:
    seen: list[tuple[int, ...]] = []

    def hook(primaries: Propagator) -> Propagator:
        assert isinstance(primaries, Sgp4Propagator)
        seen.append(tuple(omm.catalog_number for omm in primaries.omms))
        return primaries

    primaries = elements([*FORMATION, STALE])
    screen(primaries, [], START, DAY, adjust_primaries=hook)
    assert seen == [tuple(FORMATION)]


def test_the_hook_must_keep_the_primaries() -> None:
    def drop(primaries: Propagator) -> Propagator:
        return primaries.subset(np.array([0]))

    message = "adjust_primaries must keep the 3 primaries, got 1"
    with pytest.raises(ValueError, match=message):
        screen(elements(FORMATION), FIXTURE, START, DAY, adjust_primaries=drop)


def test_the_hook_must_keep_the_frame() -> None:
    def elsewhere(primaries: Propagator) -> Propagator:
        state = np.hstack(frame_to_gcrf_of(primaries))
        force = ForceModel(earth_pole(START))
        return CowellPropagator(START, state, [force], (0.0, DAY), step=DAY)

    message = "adjust_primaries must keep the frame TEME, got GCRF"
    with pytest.raises(ValueError, match=message):
        screen(elements(FORMATION[:1]), [], START, DAY, adjust_primaries=elsewhere)


def test_a_hook_that_cannot_propagate_stops_the_screening() -> None:
    def fails(primaries: Propagator) -> Propagator:
        return ManeuveredPropagator(primaries, START, -3000.0, DAY)

    with pytest.raises(PropagationError, match="fell below the Earth's surface"):
        screen(elements(FORMATION[:1]), FIXTURE, START, DAY, adjust_primaries=fails)


def test_screening_needs_nothing_but_inputs() -> None:
    primary = BY_NUMBER[68377]
    alone = screen([primary], [], START)
    assert (alone.approaches, alone.skipped, alone.catalog_count) == ((), (), 0)
    assert dict(alone.volumes) == {68377: DEFAULT_VOLUME}
    assert alone.duration == DEFAULT_DURATION == 7 * DAY
    nobody = screen([], [*FIXTURE, BY_NUMBER[68377]], START, DAY)
    assert (nobody.approaches, dict(nobody.volumes)) == ((), {})
    assert [(s.catalog_number, s.reason) for s in nobody.skipped] == [
        (STALE, SkipReason.STALE)
    ]
    assert nobody.catalog_count == len(FIXTURE) - 1


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {"start": START + np.array([0.0, 1.0]) * u.s},
            r"start must be a single time, got shape \(2,\)",
        ),
        ({"duration": -1.0}, "duration must be finite and >= 0, got -1.0"),
        ({"max_age": float("nan")}, "max_age must be finite and >= 0, got nan"),
        ({"max_lead": -1.0}, "max_lead must be finite and >= 0, got -1.0"),
        ({"workers": 0}, "workers must be at least 1, got 0"),
        ({"max_states": 0}, "max_states must be at least 1, got 0"),
    ],
)
def test_screening_checks_its_arguments(
    change: dict[str, object], message: str
) -> None:
    arguments: dict[str, object] = {
        "primaries": elements(FORMATION),
        "catalog": FIXTURE,
        "start": START,
        "duration": DAY,
    }
    with pytest.raises(ValueError, match=message):
        screen(**(arguments | change))  # type: ignore[arg-type]
