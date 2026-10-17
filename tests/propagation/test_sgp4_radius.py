"""Bounds on the radius SGP4 gives: SGP4's own coefficients, and never inside
the propagated radius."""

import dataclasses
from typing import Any

import astropy.units as u
import numpy as np
import numpy.typing as npt
import pytest
from astropy.time import Time
from hypothesis import given
from hypothesis import strategies as st
from screening_data import DATA, SNAPSHOT, load_fixture
from sgp4 import propagation as python_sgp4
from sgp4.api import WGS72, Satrec
from sgp4.model import Satrec as PythonSatrec

from keepout._types import FloatArray
from keepout.omm import Omm, load, loads
from keepout.propagation import RadiusBounds, Sgp4Propagator, propagate_in_chunks
from keepout.propagation._sgp4_radius import C1L, C1SS, drag_terms
from keepout.units import DAY

OMMS = load_fixture()
"""222 objects of the frozen public catalog, LEO to HEO; three fail SGP4
within the week below."""
PROPAGATOR = Sgp4Propagator(OMMS)
START = Time("2026-10-06T00:00:00", scale="utc")
"""The day the catalog was downloaded."""
WEEK = 7 * DAY

ORBIT_RAISING = """\
OBJECT_NAME,OBJECT_ID,EPOCH,MEAN_MOTION,ECCENTRICITY,INCLINATION,RA_OF_ASC_NODE,\
ARG_OF_PERICENTER,MEAN_ANOMALY,EPHEMERIS_TYPE,CLASSIFICATION_TYPE,NORAD_CAT_ID,\
ELEMENT_SET_NO,REV_AT_EPOCH,BSTAR,MEAN_MOTION_DOT,MEAN_MOTION_DDOT
STARLINK-38144,2026-181L,2026-10-06T06:00:04.000320,15.39947888,0.00018092,97.2844,\
323.0382,109.4209,172.3276,0,U,100256,999,595,-0.10199055,-0.03573334,0.0
"""
"""A Starlink raising its orbit (B* < 0), from the frozen public catalog: its
semi-major axis peaks inside the week, 45 km above its value at both ends."""


def python_satrec(satrec: Satrec) -> PythonSatrec:
    """sgp4's pure-Python record of the same elements, with every coefficient."""
    python = PythonSatrec()
    python.sgp4init(
        WGS72,
        "i",
        satrec.satnum,
        satrec.jdsatepoch - 2433281.5 + satrec.jdsatepochF,
        satrec.bstar,
        satrec.ndot,
        satrec.nddot,
        satrec.ecco,
        satrec.argpo,
        satrec.inclo,
        satrec.mo,
        satrec.no_kozai,
        satrec.nodeo,
    )
    return python


def radii(
    propagator: Sgp4Propagator, start: Time, offsets: npt.ArrayLike
) -> FloatArray:
    """Radii [m] of the states without error (NaN elsewhere), ``(N, T)``."""
    ephemeris = propagator.propagate(start, offsets)
    radius = np.linalg.norm(ephemeris.r, axis=-1)
    return np.where(ephemeris.valid, radius, np.nan)


def assert_within(bounds: RadiusBounds, radius: FloatArray) -> None:
    valid = ~np.isnan(radius)
    lowest = np.broadcast_to(bounds.lowest[:, None], radius.shape)
    highest = np.broadcast_to(bounds.highest[:, None], radius.shape)
    assert np.all(lowest[valid] <= radius[valid])
    assert np.all(radius[valid] <= highest[valid])


def test_drag_terms_are_sgp4inits() -> None:
    near = [satrec for satrec in (omm.satrec() for omm in OMMS) if satrec.method == "n"]
    terms = drag_terms(near)
    python = [python_satrec(satrec) for satrec in near]
    simple = np.array([p.isimp == 1 for p in python])
    np.testing.assert_array_equal(terms.simple, simple)
    for name in ("cc1", "cc4", "cc5"):
        expected = np.array([getattr(p, name) for p in python])
        np.testing.assert_allclose(getattr(terms, name), expected, rtol=1e-12)
    for power, name in ((2, "d2"), (3, "d3"), (4, "d4")):
        expected = np.array([getattr(p, name) if p.isimp != 1 else 0.0 for p in python])
        got = getattr(terms, f"c{power}") * terms.cc1**power
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-300)


@pytest.fixture
def lunisolar_changes(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Every change of the eccentricity by SGP4's lunisolar periodic terms."""
    changes: list[float] = []
    original = python_sgp4._dpper

    def recording(
        satrec: PythonSatrec,
        inclo: float,
        init: str,
        ep: float,
        *rest: *tuple[float, float, float, float, str],
    ) -> tuple[float, float, float, float, float]:
        result = original(satrec, inclo, init, ep, *rest)
        changes.append(result[0] - ep)
        return result

    monkeypatch.setattr(python_sgp4, "_dpper", recording)
    return changes


def test_the_lunisolar_terms_stay_within_their_bound(
    lunisolar_changes: list[float],
) -> None:
    deep = [satrec for satrec in (omm.satrec() for omm in OMMS) if satrec.method == "d"]
    assert len(deep) > 80
    for satrec in deep:
        python = python_satrec(satrec)
        lunisolar_changes.clear()
        for minutes in np.linspace(-1440.0, 10 * 1440.0, 45):
            python.sgp4_tsince(float(minutes))
        bound = 7.5 * satrec.ecco * (C1SS + C1L) / python.no_unkozai
        assert max(abs(change) for change in lunisolar_changes) <= bound


@given(
    start=st.floats(min_value=-20.0, max_value=10.0),
    duration=st.floats(min_value=0.0, max_value=10.0),
)
def test_the_bounds_hold_the_propagated_radius(start: float, duration: float) -> None:
    begin = START + start * u.day
    bounds = PROPAGATOR.radius_bounds(begin, duration * 86400.0)
    assert len(bounds) == len(OMMS)
    offsets = np.linspace(0.0, duration * 86400.0, 97)
    assert_within(bounds, radii(PROPAGATOR, begin, offsets))


def test_the_bounds_are_close_over_the_week() -> None:
    bounds = PROPAGATOR.radius_bounds(START, WEEK)
    radius = radii(PROPAGATOR, START, np.arange(0.0, WEEK + 1.0, 60.0))
    assert_within(bounds, radius)
    near = np.array([omm.satrec().method == "n" for omm in OMMS])
    chosen = near & ~np.any(np.isnan(radius), axis=1)
    below = radius[chosen].min(axis=1) - bounds.lowest[chosen]
    above = bounds.highest[chosen] - radius[chosen].max(axis=1)
    # A few kilometres in LEO: the J2 short-period terms and the drag.
    assert np.median(below) < 10e3
    assert np.median(above) < 10e3


def test_the_bounds_hold_an_orbit_raising_starlink() -> None:
    propagator = Sgp4Propagator(loads(ORBIT_RAISING))
    bounds = propagator.radius_bounds(START, WEEK)
    radius = radii(propagator, START, np.arange(0.0, WEEK + 1.0, 60.0))
    assert_within(bounds, radius)
    # Not merely the radius at both ends: it rises 45 km above both.
    ends = radius[0, [0, -1]].max()
    assert np.nanmax(radius) > ends + 40e3


@pytest.mark.validation
def test_the_bounds_hold_the_public_catalog_over_the_week() -> None:
    # The whole frozen catalog, every minute for a week: about a minute.
    catalog = Sgp4Propagator(load(DATA / SNAPSHOT))
    bounds = catalog.radius_bounds(START, WEEK)
    assert np.all(np.isfinite(bounds.highest))
    for (ephemeris,) in propagate_in_chunks(
        [catalog], START, np.arange(0.0, WEEK + 1.0, 60.0)
    ):
        radius = np.linalg.norm(ephemeris.r, axis=-1)
        assert_within(bounds, np.where(ephemeris.valid, radius, np.nan))


def changed(number: int, **changes: Any) -> Omm:
    omm = next(omm for omm in OMMS if omm.catalog_number == number)
    return dataclasses.replace(omm, **changes)


@pytest.mark.parametrize(
    "omm",
    [
        changed(14129, eccentricity=0.9999),  # AO-10, SGP4 fails (deep space)
        changed(31631, bstar=1.0),  # FENGYUN 1C DEB, decays within the week
    ],
)
def test_an_object_sgp4_cannot_bound_has_no_bounds(omm: Omm) -> None:
    bounds = Sgp4Propagator([omm]).radius_bounds(START, WEEK)
    assert (bounds.lowest[0], bounds.highest[0]) == (0.0, np.inf)


def test_no_objects_no_bounds() -> None:
    assert len(Sgp4Propagator([]).radius_bounds(START, WEEK)) == 0


def test_bounds_leave_the_states_unchanged() -> None:
    offsets = np.linspace(-86400.0, WEEK, 50)
    before = PROPAGATOR.propagate(START, offsets)
    PROPAGATOR.radius_bounds(START - 3 * u.day, WEEK)
    after = PROPAGATOR.propagate(START, offsets)
    np.testing.assert_array_equal(before.r, after.r)
    np.testing.assert_array_equal(before.v, after.v)
    np.testing.assert_array_equal(before.errors, after.errors)


@pytest.mark.parametrize(
    ("start", "duration", "message"),
    [
        (START, -1.0, "duration must be finite and >= 0, got -1.0"),
        (START, np.inf, "duration must be finite and >= 0, got inf"),
        (START + np.array([0.0, 1.0]) * u.s, 1.0, "epoch must be a single time"),
    ],
)
def test_the_span_is_checked(start: Time, duration: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PROPAGATOR.radius_bounds(start, duration)
