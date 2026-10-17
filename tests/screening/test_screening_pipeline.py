"""Screening end to end: the search's minima as close approaches in GCRF, the
element sets left out and why, and the same result in any number of
processes."""

import dataclasses
from collections.abc import Callable, Sequence

import astropy.units as u
import numpy as np
import numpy.typing as npt
import pytest
from screening_data import (
    FAST_SPEED,
    FIXTURE_START,
    fixture_by_number,
    load_fixture,
    socrates_approaches,
)

from keepout.errors import GeometryError, UnsupportedElements
from keepout.frames import teme_to_gcrf
from keepout.omm import Omm
from keepout.propagation import Sgp4Propagator, describe_error
from keepout.screening import (
    DEFAULT_VOLUME,
    ApproachSearch,
    RangeMinimum,
    ScreeningResult,
    ScreeningVolume,
    SkipReason,
    apogee_perigee_filter,
    screen,
)
from keepout.screening.pipeline import DEFAULT_DURATION, MAX_AGE, MAX_LEAD
from keepout.screening.tca import PERTURBATION
from keepout.units import DAY

START = FIXTURE_START
FIXTURE = load_fixture()
BY_NUMBER = fixture_by_number()
FORMATION = [68377, 56156, 58199]
"""Formation flyers: many slow approaches."""
SPREAD = [omm.catalog_number for omm in FIXTURE[::9][:5]]
"""LEO, GEO and HEO objects of the fixture."""
FAILING = [45665, 45711]
"""SGP4 fails for them 41 h 36 min and 23 h 12 min after START (error 1);
each is the other's only neighbour."""
STALE = 32360
"""Its elements are 29.3 days old at START."""
REENTERING = 270365
"""An orbit with its perigee 113 km up: SGP4's drag on it exceeds the search's
bound on perturbations from 8 days after its epoch; SGP4 fails at 10.5 days."""


def elements(numbers: Sequence[int]) -> list[Omm]:
    return [BY_NUMBER[number] for number in numbers]


def without(*numbers: int) -> list[Omm]:
    return [omm for omm in FIXTURE if omm.catalog_number not in numbers]


def summary(result: ScreeningResult) -> list[object]:
    """Everything in a result, exact, for comparisons."""
    approaches: list[object] = [
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
    skipped = [(s.elements, s.role, s.reason, s.detail) for s in result.skipped]
    return [
        *approaches,
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
        assert approach.miss_distance == pytest.approx(minimum.distance, abs=1e-8)
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
    assert summary(result)[:-4] == summary(plain)[:-4]
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


def test_default_ages_keep_sgp4_within_the_searchs_assumption() -> None:
    # Acceleration beyond point mass and J2 from MAX_LEAD before each epoch
    # to MAX_AGE after it, every 12 hours, from positions 1 s apart, until
    # SGP4 fails (a window holding the failure skips the object).
    mu, re, j2 = 398600.8e9, 6378.135e3, 0.001082616  # SGP4's WGS 72
    worst: dict[int, float] = {}
    for omm in FIXTURE:
        satrec = omm.satrec()
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
    assert summary(result)[:-4] == summary(rest)[:-4]
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
    assert summary(result)[:-4] == summary(plain)[:-4]
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
    assert summary(result)[:-4] == summary(others)[:-4]


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


@pytest.mark.validation
def test_every_socrates_approach_of_the_fixture_is_found_in_gcrf() -> None:
    # Offline, about 4 s. SOCRATES prints TCA to the millisecond, range and
    # relative speed to the metre (per second); a slow pair's TCA is
    # ill-conditioned.
    socrates = [
        approach
        for approach in socrates_approaches()
        if approach.primary in BY_NUMBER and approach.secondary in BY_NUMBER
    ]
    start = min(approach.tca for approach in socrates) - 1 * u.hour
    duration = (max(approach.tca for approach in socrates) - start).sec + 3600.0
    primaries = sorted({approach.primary for approach in socrates})
    result = screen(elements(primaries), FIXTURE, start, duration)
    found: dict[tuple[int, int], list[npt.NDArray[np.float64]]] = {}
    for a in result.approaches:
        row = [(a.tca - start).to_value(u.s), a.miss_distance, a.relative_speed]
        found.setdefault((a.primary_number, a.secondary_number), []).append(
            np.array(row)
        )
    assert len(socrates) == 1033
    for approach in socrates:
        offset = (approach.tca - start).sec
        match = min(
            found[approach.primary, approach.secondary],
            key=lambda m: abs(m[0] - offset),
        )
        fast = approach.relative_speed >= FAST_SPEED
        assert match[0] == pytest.approx(offset, abs=1e-3 if fast else 0.5)
        assert match[1] == pytest.approx(approach.miss_distance, abs=0.5)
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
        ({"step": 0.0}, "step must be finite and > 0, got 0.0"),
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
