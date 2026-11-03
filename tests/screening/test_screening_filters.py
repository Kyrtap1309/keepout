"""The apogee/perigee filter: exact on the radius bounds, and it never drops a
pair the brute-force oracle finds within the primary's volume."""

import numpy as np
import pytest
from astropy.time import Time
from hypothesis import given, settings
from hypothesis import strategies as st
from screening_data import FIXTURE_START, SGP4_FAILING, load_fixture
from screening_oracle import Oracle, brute_force

from keepout.propagation import RadiusBounds, Sgp4Propagator
from keepout.screening import (
    DEFAULT_VOLUME,
    ScreeningVolume,
    VolumeRule,
    apogee_perigee_filter,
    volume_for,
)
from keepout.units import DAY, KM

CATALOG = load_fixture()
"""222 objects of the frozen public catalog, LEO to HEO."""
COLUMN = {omm.catalog_number: index for index, omm in enumerate(CATALOG)}
PRIMARIES = CATALOG[::9]
"""25 primaries spread over the fixture (LEO, MEO, GEO, HEO)."""
WEEK = 7 * DAY
WINDOWS = (
    FIXTURE_START,  # after most element epochs
    Time("2026-09-29T00:00:00", scale="utc"),  # before most of them
)
LARGEST_REACH = 500 * KM


def bands(*kilometres: tuple[float, float]) -> RadiusBounds:
    return RadiusBounds(
        np.array([low for low, _ in kilometres]) * KM,
        np.array([high for _, high in kilometres]) * KM,
    )


def test_pairs_within_the_bounding_radius_are_kept() -> None:
    primaries = bands((7000.0, 7010.0))
    catalog = bands(
        (7015.0, 7020.0),  # 5 km above: kept, the boundary is inside
        (7015.001, 7020.0),  # 1 m farther: dropped
        (6990.0, 6995.0),  # 5 km below: kept
        (6990.0, 6994.999),  # dropped
        (6000.0, 8000.0),  # around: kept
        (7002.0, 7003.0),  # inside: kept
    )
    keep = apogee_perigee_filter(primaries, catalog, DEFAULT_VOLUME)
    np.testing.assert_array_equal(keep, [[True, False, True, False, True, True]])


def test_each_primary_is_padded_with_its_own_volume() -> None:
    primaries = bands((7000.0, 7010.0), (7000.0, 7010.0))
    catalog = bands((7030.0, 7040.0), (7061.5, 7070.0))
    volumes = [ScreeningVolume.sphere(5 * KM), ScreeningVolume(400.0, 44 * KM, 51 * KM)]
    keep = apogee_perigee_filter(primaries, catalog, volumes)
    np.testing.assert_array_equal(keep, [[False, False], [True, False]])
    with pytest.raises(ValueError, match="1 volumes given for 2 primaries"):
        apogee_perigee_filter(primaries, catalog, volumes[:1])


def test_objects_without_bounds_are_always_kept() -> None:
    unbounded = RadiusBounds(np.zeros(1), np.full(1, np.inf))
    leo, geo = bands((7000.0, 7010.0)), bands((42160.0, 42170.0))
    assert apogee_perigee_filter(unbounded, geo, DEFAULT_VOLUME).tolist() == [[True]]
    assert apogee_perigee_filter(leo, unbounded, DEFAULT_VOLUME).tolist() == [[True]]
    assert apogee_perigee_filter(leo, geo, DEFAULT_VOLUME).tolist() == [[False]]


def test_the_result_has_a_row_per_primary() -> None:
    none = bands()
    some = bands((7000.0, 7010.0), (8000.0, 8010.0), (9000.0, 9010.0))
    assert apogee_perigee_filter(none, some, DEFAULT_VOLUME).shape == (0, 3)
    assert apogee_perigee_filter(some, none, DEFAULT_VOLUME).shape == (3, 0)
    assert apogee_perigee_filter(none, some, []).shape == (0, 3)


radii = st.floats(min_value=6000.0, max_value=50000.0)
fractions = st.floats(min_value=0.0, max_value=1.0)


widths = st.floats(min_value=0.0, max_value=5000.0)
reaches = st.floats(min_value=0.1, max_value=100.0)


@given(
    radii,
    widths,
    widths,
    fractions,
    fractions,
    st.floats(min_value=-1.0, max_value=1.0),
    reaches,
)
def test_radii_within_reach_keep_the_pair(
    low_p: float,
    width_p: float,
    width_s: float,
    at_p: float,
    at_s: float,
    separation: float,
    reach: float,
) -> None:
    # Bands holding radii within the reach of each other: kept.
    radius_p = low_p + at_p * width_p
    # 1 mm inside the reach: well above rounding at these radii [m].
    radius_s = radius_p + separation * (reach - 1e-6)
    low_s = radius_s - at_s * width_s
    keep = apogee_perigee_filter(
        bands((low_p, low_p + width_p)),
        bands((low_s, low_s + width_s)),
        ScreeningVolume.sphere(reach * KM),
    )
    assert keep[0, 0]


@given(radii, widths, widths, st.floats(min_value=1.0, max_value=2.0), reaches)
def test_bands_beyond_reach_drop_the_pair(
    low_p: float, width_p: float, width_s: float, gap: float, reach: float
) -> None:
    # The secondary's band starts more than the reach above the primary's.
    low_s = low_p + width_p + gap * reach + 1e-6
    keep = apogee_perigee_filter(
        bands((low_p, low_p + width_p)),
        bands((low_s, low_s + width_s)),
        ScreeningVolume.sphere(reach * KM),
    )
    assert not keep[0, 0]


@pytest.fixture(scope="module")
def oracles() -> tuple[Oracle, ...]:
    """Every sampled minimum within 500 km of the primaries over each window."""
    return tuple(
        brute_force(PRIMARIES, CATALOG, start, WEEK, LARGEST_REACH, step=120.0)
        for start in WINDOWS
    )


@pytest.fixture(scope="module")
def bounds() -> tuple[RadiusBounds, ...]:
    """Radius bounds of the whole fixture over each window."""
    catalog = Sgp4Propagator(CATALOG)
    return tuple(catalog.radius_bounds(start, WEEK) for start in WINDOWS)


def test_the_oracle_finds_pairs_to_screen(oracles: tuple[Oracle, ...]) -> None:
    for found in oracles:
        assert found.failed <= set(SGP4_FAILING)
        assert len(found.pairs) > 500
        assert len(found.within(5 * KM)) > 100


volumes = st.one_of(
    st.builds(
        ScreeningVolume.sphere, st.floats(min_value=100.0, max_value=LARGEST_REACH)
    ),
    st.builds(
        ScreeningVolume,
        st.floats(min_value=100.0, max_value=LARGEST_REACH),
        st.floats(min_value=100.0, max_value=LARGEST_REACH),
        st.floats(min_value=100.0, max_value=LARGEST_REACH),
    ),
    st.just(ScreeningVolume.sfs_handbook),
)


# An example propagates the fixture's objects: no 200 ms deadline.
@settings(deadline=None)
@given(
    window=st.integers(min_value=0, max_value=len(WINDOWS) - 1),
    chosen=st.lists(
        st.integers(min_value=0, max_value=len(PRIMARIES) - 1),
        min_size=1,
        max_size=len(PRIMARIES),
        unique=True,
    ),
    volume=volumes,
)
def test_no_pair_the_oracle_finds_is_dropped(
    oracles: tuple[Oracle, ...],
    bounds: tuple[RadiusBounds, ...],
    window: int,
    chosen: list[int],
    volume: ScreeningVolume | VolumeRule,
) -> None:
    primaries = [PRIMARIES[i] for i in chosen]
    every = bounds[window]
    rows = np.array([COLUMN[omm.catalog_number] for omm in primaries])
    own = RadiusBounds(every.lowest[rows], every.highest[rows])
    resolved = [volume_for(volume, primary) for primary in primaries]
    keep = apogee_perigee_filter(own, every, resolved)
    row = {omm.catalog_number: i for i, omm in enumerate(primaries)}
    reach = {
        omm.catalog_number: v.bounding_radius
        for omm, v in zip(primaries, resolved, strict=True)
    }
    for minimum in oracles[window].minima:
        if minimum.primary in row and minimum.distance <= reach[minimum.primary]:
            assert keep[row[minimum.primary], COLUMN[minimum.secondary]], minimum


def test_the_filter_drops_most_pairs(
    oracles: tuple[Oracle, ...], bounds: tuple[RadiusBounds, ...]
) -> None:
    every = bounds[0]
    keep = apogee_perigee_filter(every, every, DEFAULT_VOLUME)
    assert keep.mean() < 0.4
    # The objects the oracle finds within 5 km are all kept.
    for minimum in oracles[0].within(5 * KM):
        assert keep[COLUMN[minimum.primary], COLUMN[minimum.secondary]]
