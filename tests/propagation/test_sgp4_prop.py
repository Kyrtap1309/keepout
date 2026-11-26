"""SGP4 for many objects: the same states as sgp4's own Satrec, in metres."""

import dataclasses
import itertools
import pickle

import astropy.units as u
import numpy as np
import numpy.typing as npt
import pytest
from astropy.time import Time
from hypothesis import given
from hypothesis import strategies as st
from screening_data import FIXTURE_START, SGP4_FAILING, load_fixture

from keepout.errors import UnsupportedElements
from keepout.omm import Omm

# Whitebox: the dates keepout hands SGP4, for the tests named so.
from keepout.propagation import (
    Ephemeris,
    Propagator,
    Sgp4Propagator,
    describe_error,
    propagate_in_chunks,
    satrec_of,
    sgp4_prop,
)
from keepout.propagation.sgp4_prop import Sgp4Sampler, _utc_dates

OMMS = load_fixture()
"""222 objects of the frozen public catalog (LEO to HEO, 6-digit numbers, and
three that fail SGP4 on the grid below)."""
PROPAGATOR = Sgp4Propagator(OMMS)
"""The day the catalog was downloaded."""
OFFSETS = np.concatenate([np.linspace(-86400.0, 7 * 86400.0, 97), [0.25, 1e-3]])
WHOLE = PROPAGATOR.propagate(FIXTURE_START, OFFSETS)


def satrec_states(
    omm: Omm, times: Time
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """sgp4's own single-object propagation, converted to metres."""
    utc = times.utc
    errors, r, v = satrec_of(omm).sgp4_array(
        np.atleast_1d(utc.jd1), np.atleast_1d(utc.jd2)
    )
    return errors, r * 1e3, v * 1e3


def test_states_are_sgp4s_for_each_object() -> None:
    # The fractions of day differ from astropy's by ~10 ps (keepout keeps
    # whole days apart), which SGP4 turns into at most ~2 um.
    assert WHOLE.frame == PROPAGATOR.frame == "TEME"
    assert WHOLE.r.shape == WHOLE.v.shape == (len(OMMS), len(OFFSETS), 3)
    times = FIXTURE_START + OFFSETS * u.s
    for i, omm in enumerate(OMMS):
        errors, r, v = satrec_states(omm, times)
        np.testing.assert_array_equal(WHOLE.errors[i], errors)
        np.testing.assert_allclose(WHOLE.r[i], r, rtol=0, atol=1e-5)
        np.testing.assert_allclose(WHOLE.v[i], v, rtol=0, atol=1e-8)


def test_whitebox_states_are_bit_identical_to_satrec_at_keepouts_dates() -> None:
    # The same compiled SGP4 as Satrec.sgp4, given the same Julian dates.
    jd, fraction = _utc_dates(FIXTURE_START, OFFSETS)
    for i, omm in enumerate(OMMS):
        errors, r, v = satrec_of(omm).sgp4_array(jd, fraction)
        np.testing.assert_array_equal(WHOLE.errors[i], errors)
        np.testing.assert_array_equal(WHOLE.r[i], r * 1e3)
        np.testing.assert_array_equal(WHOLE.v[i], v * 1e3)


@pytest.mark.parametrize(
    "epoch",
    [
        FIXTURE_START,
        Time("2026-10-06T17:23:11.123456", scale="utc"),
        Time(61000.3, format="mjd"),
    ],
)
def test_whitebox_keepouts_dates_are_astropys_to_100_ps(epoch: Time) -> None:
    jd, fraction = _utc_dates(epoch, OFFSETS)
    utc = (epoch + OFFSETS * u.s).utc
    difference = ((jd - utc.jd1) + (fraction - utc.jd2)) * 86400.0
    assert np.max(np.abs(difference)) < 1e-10
    # Whole days are kept out of the fraction (its resolution is ~10 ps).
    assert np.all(np.abs(fraction) < 0.51)


@pytest.mark.parametrize("scale", ["utc", "tdb"])
def test_the_sampler_gives_propagates_states_bit_for_bit(scale: str) -> None:
    epoch = getattr(FIXTURE_START, scale)
    first, second = PROPAGATOR.subset(np.arange(5)), PROPAGATOR.subset([7, 1])
    chosen = [6, 0, 3, 6]
    errors, r, v = Sgp4Sampler(epoch, (first, second)).states(chosen)(OFFSETS)
    rows = [1, 0, 3, 1]
    np.testing.assert_array_equal(errors, WHOLE.errors[rows])
    assert r.tobytes() == WHOLE.r[rows].tobytes()
    assert v.tobytes() == WHOLE.v[rows].tobytes()


def test_offsets_count_the_leap_second() -> None:
    # 2016-12-31 ended with 23:59:60: 31 s after 23:59:30 is midnight.
    omm = dataclasses.replace(OMMS[0], epoch=Time("2016-12-31T12:00:00", scale="utc"))
    epoch = Time("2016-12-31T23:59:30", scale="utc")
    offsets = [29.0, 30.0, 30.5, 31.0]
    ephemeris = Sgp4Propagator([omm]).propagate(epoch, offsets)
    assert ephemeris.times.utc.isot.tolist() == [
        "2016-12-31T23:59:59.000",
        "2016-12-31T23:59:60.000",
        "2016-12-31T23:59:60.500",
        "2017-01-01T00:00:00.000",
    ]
    errors, r, v = satrec_states(omm, ephemeris.times)
    np.testing.assert_array_equal(ephemeris.errors[0], errors)
    np.testing.assert_allclose(ephemeris.r[0], r, rtol=0, atol=1e-5)
    np.testing.assert_allclose(ephemeris.v[0], v, rtol=0, atol=1e-8)
    _, midnight, _ = satrec_states(omm, Time("2017-01-01T00:00:00", scale="utc"))
    np.testing.assert_allclose(ephemeris.r[0, 3], midnight[0], rtol=0, atol=1e-5)


@pytest.mark.parametrize("scale", ["tt", "tai", "tdb"])
def test_any_time_scale_gives_the_same_states(scale: str) -> None:
    epoch = getattr(FIXTURE_START, scale)
    other = PROPAGATOR.propagate(epoch, OFFSETS)
    assert other.epoch is epoch
    np.testing.assert_array_equal(other.r, WHOLE.r)
    np.testing.assert_array_equal(other.v, WHOLE.v)


def test_metres_and_metres_per_second() -> None:
    # SGP4's velocity is its own formula, not exactly the rate of its
    # position: they differ by 0.04 m/s (median) and up to ~12 m/s on
    # eccentric or decaying orbits. A km/m mix-up would be 1000 times off.
    step = 0.1
    later = PROPAGATOR.propagate(FIXTURE_START, OFFSETS + step)
    earlier = PROPAGATOR.propagate(FIXTURE_START, OFFSETS - step)
    rate = (later.r - earlier.r) / (2.0 * step)
    both = WHOLE.valid & later.valid & earlier.valid
    difference = np.linalg.norm(rate - WHOLE.v, axis=-1)[both]
    assert np.median(difference) < 0.1
    assert np.max(difference) < 20.0
    radius = np.linalg.norm(WHOLE.r, axis=-1)[WHOLE.valid]
    assert 6.4e6 < radius.min() < radius.max() < 2e8


def test_failed_states_have_codes_and_nan() -> None:
    # Two Starlinks of the fixture decay within the week (SGP4 error 1),
    # and so does one analyst object a day before the start.
    failed = np.flatnonzero(~WHOLE.valid.all(axis=1))
    assert [OMMS[i].catalog_number for i in failed] == list(SGP4_FAILING)
    assert {describe_error(code) for code in np.unique(WHOLE.errors)} == {
        "no error",
        "error 1: mean eccentricity outside the range 0 to 1",
    }
    invalid = ~WHOLE.valid
    assert np.all(np.isnan(WHOLE.r[invalid]))
    assert np.all(np.isnan(WHOLE.v[invalid]))
    assert np.all(np.isfinite(WHOLE.r[WHOLE.valid]))
    assert np.all(np.isfinite(WHOLE.v[WHOLE.valid]))


def test_subsets_share_the_records(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(omm: Omm) -> None:
        raise AssertionError("a subset must not initialise SGP4 again")

    monkeypatch.setattr(sgp4_prop, "satrec_of", refuse)
    subset = PROPAGATOR.subset([5, 0, 5])
    assert len(subset) == 3
    assert subset.omms == (OMMS[5], OMMS[0], OMMS[5])
    ephemeris = subset.propagate(FIXTURE_START, OFFSETS)
    np.testing.assert_array_equal(ephemeris.r, WHOLE.r[[5, 0, 5]])
    np.testing.assert_array_equal(ephemeris.v, WHOLE.v[[5, 0, 5]])
    np.testing.assert_array_equal(ephemeris.errors, WHOLE.errors[[5, 0, 5]])
    nobody = PROPAGATOR.subset([])
    assert len(nobody) == 0
    assert nobody.propagate(FIXTURE_START, OFFSETS).r.shape == (0, len(OFFSETS), 3)


@pytest.mark.parametrize(
    ("indices", "error", "message"),
    [
        ([[0, 1]], ValueError, r"one-dimensional, got shape \(1, 2\)"),
        ([0.0, 1.0], ValueError, "must be integers, got float64"),
        ([True, False], ValueError, "must be integers, got bool"),
        ([0, -1], IndexError, r"in \[0, 222\), got -1 .. 0"),
        ([222], IndexError, r"in \[0, 222\), got 222 .. 222"),
    ],
)
def test_subset_rejects_bad_indices(
    indices: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        PROPAGATOR.subset(indices)  # type: ignore[arg-type]


@given(
    cuts=st.lists(st.integers(min_value=0, max_value=len(OFFSETS)), max_size=4),
    objects=st.lists(
        st.integers(min_value=0, max_value=len(OMMS) - 1), min_size=1, max_size=8
    ),
)
def test_chunks_and_subsets_give_the_same_numbers(
    cuts: list[int], objects: list[int]
) -> None:
    # What chunking and worker processes rely on: a state does not depend
    # on the other objects or times of the call.
    subset = PROPAGATOR.subset(objects)
    bounds = [0, *sorted(cuts), len(OFFSETS)]
    for start, stop in itertools.pairwise(bounds):
        part = subset.propagate(FIXTURE_START, OFFSETS[start:stop])
        np.testing.assert_array_equal(part.r, WHOLE.r[objects, start:stop])
        np.testing.assert_array_equal(part.v, WHOLE.v[objects, start:stop])
        np.testing.assert_array_equal(part.errors, WHOLE.errors[objects, start:stop])


def test_one_object_at_one_time() -> None:
    # The call a TCA search makes many times: a pair at one offset.
    pair = PROPAGATOR.subset([3, 7])
    ephemeris = pair.propagate(FIXTURE_START, 1234.5)
    assert ephemeris.r.shape == (2, 1, 3)
    expected = PROPAGATOR.propagate(FIXTURE_START, [1234.5])
    np.testing.assert_array_equal(ephemeris.r, expected.r[[3, 7]])


def test_the_propagator_pickles_and_follows_the_protocol() -> None:
    propagator: Propagator = PROPAGATOR
    assert isinstance(propagator, Propagator)
    assert len(PROPAGATOR) == len(OMMS)
    assert PROPAGATOR.omms == OMMS
    assert repr(PROPAGATOR) == "Sgp4Propagator(222 objects)"
    copy = pickle.loads(pickle.dumps(PROPAGATOR.subset(range(10))))
    assert isinstance(copy, Sgp4Propagator)
    assert copy.omms == OMMS[:10]
    ephemeris = copy.propagate(FIXTURE_START, OFFSETS)
    np.testing.assert_array_equal(ephemeris.r, WHOLE.r[:10])
    np.testing.assert_array_equal(ephemeris.v, WHOLE.v[:10])


def test_chunked_propagation_of_the_fixture() -> None:
    chunks = [
        chunk
        for (chunk,) in propagate_in_chunks(
            [PROPAGATOR], FIXTURE_START, OFFSETS, max_states=2000
        )
    ]
    assert [len(chunk.offsets) for chunk in chunks] == [9] * 11
    np.testing.assert_array_equal(
        np.concatenate([c.r for c in chunks], axis=1), WHOLE.r
    )


def test_empty_propagators_and_grids() -> None:
    nobody = Sgp4Propagator([])
    assert len(nobody) == 0
    assert nobody.propagate(FIXTURE_START, OFFSETS).errors.shape == (0, len(OFFSETS))
    never = PROPAGATOR.propagate(FIXTURE_START, [])
    assert isinstance(never, Ephemeris)
    assert never.r.shape == (len(OMMS), 0, 3)
    assert never.errors.shape == (len(OMMS), 0)


def test_elements_sgp4_cannot_hold_are_named() -> None:
    too_large = dataclasses.replace(OMMS[1], catalog_number=400_000)
    with pytest.raises(
        UnsupportedElements,
        match=r"element set 1 \(catalog number 400000\) cannot initialise SGP4: "
        r"catalog number 400000 is above 339999",
    ):
        Sgp4Propagator([OMMS[0], too_large])


def test_from_omms_sets_aside_the_elements_sgp4_cannot_hold() -> None:
    too_large = dataclasses.replace(OMMS[1], catalog_number=400_000)
    propagator, kept, rejected = Sgp4Propagator.from_omms(
        [OMMS[0], too_large, OMMS[2], too_large]
    )
    assert propagator.omms == (OMMS[0], OMMS[2])
    assert kept.tolist() == [0, 2]
    assert [index for index, _ in rejected] == [1, 3]
    for _, error in rejected:
        assert isinstance(error, UnsupportedElements)
        assert str(error).startswith("catalog number 400000 is above 339999")
    np.testing.assert_array_equal(
        propagator.propagate(FIXTURE_START, OFFSETS).r, WHOLE.r[[0, 2]]
    )
    nobody, none_kept, all_rejected = Sgp4Propagator.from_omms([too_large])
    assert (len(nobody), none_kept.tolist(), len(all_rejected)) == (0, [], 1)


@pytest.mark.parametrize(
    ("epoch", "offsets", "message"),
    [
        (
            FIXTURE_START + np.array([0.0, 1.0]) * u.s,
            [0.0],
            "epoch must be a single time",
        ),
        (FIXTURE_START, [[0.0]], "offsets must be one-dimensional"),
        (FIXTURE_START, [np.nan], "offsets must be finite"),
    ],
)
def test_propagate_rejects_bad_grids(
    epoch: Time, offsets: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PROPAGATOR.propagate(epoch, offsets)  # type: ignore[arg-type]
