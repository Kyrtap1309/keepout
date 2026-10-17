"""Screening volumes: RTN ellipsoids, presets and the handbook's regimes."""

import dataclasses
import math
import pickle

import numpy as np
import pytest
from builders import LEO_R, LEO_V
from hypothesis import given
from hypothesis import strategies as st
from scipy.optimize import brentq
from screening_data import fixture_by_number
from sgp4.api import WGS72
from sgp4.model import Satrec as PythonSatrec

from keepout._types import FloatArray
from keepout.errors import UnsupportedElements
from keepout.frames import relative_position_rtn
from keepout.omm import Omm
from keepout.screening import (
    DEFAULT_VOLUME,
    SFS_HANDBOOK_VOLUMES,
    ScreeningVolume,
    SfsRegime,
    sfs_regime,
    volume_for,
)
from keepout.screening.volume import (
    NASA_GP_DATA_RADIUS,
    SFS_DEEP_SPACE_INCLINATION,
    SFS_DEEP_SPACE_PERIOD,
    SFS_ECCENTRICITY,
    SFS_LEO_PERIGEES,
    SOCRATES_RADIUS,
)

BY_NUMBER = fixture_by_number()
"""The 222 objects of the frozen public catalog's fixture."""

semi_axis = st.floats(min_value=1.0, max_value=1e5)
volumes = st.builds(ScreeningVolume, semi_axis, semi_axis, semi_axis)
spheres = st.builds(ScreeningVolume.sphere, semi_axis)


@st.composite
def directions(draw: st.DrawFn) -> FloatArray:
    """Unit vectors in any direction (never the zero vector)."""
    polar = draw(st.floats(min_value=0.0, max_value=math.pi))
    azimuth = draw(st.floats(min_value=-math.pi, max_value=math.pi))
    return np.array(
        [
            math.sin(polar) * math.cos(azimuth),
            math.sin(polar) * math.sin(azimuth),
            math.cos(polar),
        ]
    )


@st.composite
def volumes_and_offsets(
    draw: st.DrawFn, volumes: st.SearchStrategy[ScreeningVolume] = volumes
) -> tuple[ScreeningVolume, FloatArray]:
    """A volume and a non-zero offset from well inside to well outside it.

    The distance is 0.25 to 1.5 times a reach between the smallest and the
    largest semi-axis, as often below the reach as above it, so that both
    sides of the boundary are drawn for any shape.
    """
    volume = draw(volumes)
    reach = draw(
        st.floats(
            min_value=min(volume.radial, volume.transverse, volume.normal),
            max_value=volume.bounding_radius,
        )
    )
    nearer = draw(st.booleans())
    scale = draw(
        st.floats(min_value=0.25, max_value=1.0)
        if nearer
        else st.floats(min_value=1.0, max_value=1.5)
    )
    return volume, scale * reach * draw(directions())


def test_a_volume_holds_its_semi_axes_as_floats() -> None:
    volume = ScreeningVolume(400, 25e3, np.float64(25e3), label="test")
    assert (volume.radial, volume.transverse, volume.normal) == (400.0, 25e3, 25e3)
    assert all(
        type(value) is float
        for value in (volume.radial, volume.transverse, volume.normal)
    )
    np.testing.assert_array_equal(volume.semi_axes, [400.0, 25e3, 25e3])
    assert not volume.semi_axes.flags.writeable
    assert volume.bounding_radius == 25e3
    assert not volume.is_sphere


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_semi_axes_must_be_finite_and_positive(axis: int, bad: float) -> None:
    radial, transverse, normal = [bad if i == axis else 1.0 for i in range(3)]
    name = ("radial", "transverse", "normal")[axis]
    with pytest.raises(ValueError, match=f"{name} must be finite and > 0"):
        ScreeningVolume(radial, transverse, normal)


def test_the_label_is_not_part_of_equality() -> None:
    assert ScreeningVolume.sphere(5e3, label="a") == ScreeningVolume.sphere(5e3)
    assert ScreeningVolume.sphere(5e3) != ScreeningVolume(5e3, 5e3, 5.1e3)


def test_presets() -> None:
    assert (
        DEFAULT_VOLUME
        == ScreeningVolume.sphere(5e3)
        == ScreeningVolume.sphere(SOCRATES_RADIUS)
    )
    assert DEFAULT_VOLUME.is_sphere
    assert "SOCRATES" in DEFAULT_VOLUME.label
    nasa = ScreeningVolume.nasa_gp_data()
    assert (
        nasa
        == ScreeningVolume.sphere(10e3)
        == ScreeningVolume.sphere(NASA_GP_DATA_RADIUS)
    )
    assert "NASA" in nasa.label
    assert nasa.bounding_radius == 10e3


def test_the_handbook_table() -> None:
    # USSPACECOM Spaceflight Safety Handbook v1.7, Table 3, in km.
    table = {
        SfsRegime.LEO_1: (0.4, 44.0, 51.0),
        SfsRegime.LEO_2: (0.4, 25.0, 25.0),
        SfsRegime.LEO_3: (0.4, 12.0, 12.0),
        SfsRegime.LEO_4: (0.4, 2.0, 2.0),
        SfsRegime.DEEP_SPACE: (10.0, 10.0, 10.0),
        SfsRegime.OUTSIDE: (10.0, 10.0, 10.0),
    }
    assert set(SFS_HANDBOOK_VOLUMES) == set(SfsRegime)
    for regime, kilometres in table.items():
        volume = SFS_HANDBOOK_VOLUMES[regime]
        np.testing.assert_allclose(volume.semi_axes, np.array(kilometres) * 1e3)
        assert volume.label.startswith("SFS Handbook v1.7 Table 3, ")
    assert "outside" in SFS_HANDBOOK_VOLUMES[SfsRegime.OUTSIDE].label
    assert list(SFS_LEO_PERIGEES.values()) == [500e3, 750e3, 1200e3, 2000e3]


def test_the_handbook_table_is_read_only() -> None:
    with pytest.raises(TypeError, match="does not support item assignment"):
        SFS_HANDBOOK_VOLUMES[SfsRegime.LEO_1] = DEFAULT_VOLUME  # type: ignore[index]


def test_points_on_and_off_the_boundary() -> None:
    volume = ScreeningVolume(400.0, 25e3, 30e3)
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [400.0, 0.0, 0.0],
            [0.0, -25e3, 0.0],
            [0.0, 0.0, 30e3],
            [400.0 * (1 + 1e-9), 0.0, 0.0],
            [0.0, 25e3 * (1 + 1e-9), 0.0],
            [0.0, 0.0, -30e3 * (1 + 1e-9)],
            [300.0, 15e3, 0.0],  # 0.5625 + 0.36 < 1
            [300.0, 17e3, 0.0],  # 0.5625 + 0.4624 > 1
            [np.nan, 0.0, 0.0],
        ]
    )
    np.testing.assert_array_equal(
        volume.contains(points),
        [True, True, True, True, False, False, False, True, False, False],
    )
    assert volume.contains(points).shape == (10,)
    assert volume.contains(points.reshape(2, 5, 3)).shape == (2, 5)
    assert volume.contains([0.0, 0.0, 0.0]).shape == ()


@pytest.mark.parametrize("offsets", [[1.0, 2.0], 1.0])
def test_offsets_need_three_components(offsets: list[float] | float) -> None:
    with pytest.raises(ValueError, match="offsets must have 3 components"):
        DEFAULT_VOLUME.contains(offsets)


@given(volumes_and_offsets(spheres))
def test_a_sphere_holds_the_points_within_its_radius(
    sphere_and_offset: tuple[ScreeningVolume, FloatArray],
) -> None:
    sphere, offset = sphere_and_offset
    radius = sphere.radial
    distance = float(np.linalg.norm(offset))
    inside = bool(sphere.contains(offset))
    # Away from the boundary by more than rounding, the test is the distance.
    if abs(distance - radius) > 1e-12 * radius:
        assert inside == (distance <= radius)


@given(volumes_and_offsets())
def test_the_bounding_sphere_holds_the_volume(
    volume_and_offset: tuple[ScreeningVolume, FloatArray],
) -> None:
    volume, offset = volume_and_offset
    if volume.contains(offset):
        assert float(np.linalg.norm(offset)) <= volume.bounding_radius * (1 + 1e-12)
    # A sphere of the smallest semi-axis lies inside the volume.
    smallest = min(volume.radial, volume.transverse, volume.normal)
    if float(np.linalg.norm(offset)) <= smallest * (1 - 1e-12):
        assert volume.contains(offset)


@given(volumes_and_offsets(), st.floats(min_value=0.0, max_value=1.0))
def test_shrinking_an_inside_point_keeps_it_inside(
    volume_and_offset: tuple[ScreeningVolume, FloatArray], factor: float
) -> None:
    volume, offset = volume_and_offset
    if volume.contains(offset):
        assert volume.contains(factor * offset)


def test_contains_position_tests_the_relative_position_in_rtn() -> None:
    # A circular equatorial orbit: R is +x, T is +y and N is +z.
    volume = ScreeningVolume(400.0, 25e3, 25e3)
    others = LEO_R + np.array([[300.0, 0.0, 0.0], [500.0, 0.0, 0.0], [0.0, 2e4, 0.0]])
    np.testing.assert_array_equal(
        volume.contains_position(LEO_R, LEO_V, others), [True, False, True]
    )
    rng = np.random.default_rng(3)
    r = LEO_R + rng.normal(0.0, 1e5, (6, 1, 3))
    v = LEO_V + rng.normal(0.0, 1e2, (6, 1, 3))
    other = LEO_R + rng.normal(0.0, 2e4, (40, 3))
    np.testing.assert_array_equal(
        volume.contains_position(r, v, other),
        volume.contains(relative_position_rtn(r, v, other)),
    )


@pytest.mark.parametrize(
    ("number", "regime"),
    [
        (31631, SfsRegime.LEO_1),  # FENGYUN 1C DEB, perigee 450 km
        (270214, SfsRegime.LEO_1),  # perigee 431 km, apogee 2866 km, e 0.15
        (30045, SfsRegime.LEO_2),  # FENGYUN 1C DEB, perigee 746 km
        (900, SfsRegime.LEO_3),  # CALSPHERE 1, perigee 954 km
        (270071, SfsRegime.LEO_4),  # perigee 1890 km
        (25924, SfsRegime.DEEP_SPACE),  # ABS-6, geostationary
        (2866, SfsRegime.DEEP_SPACE),  # LES-5, period 1316 min
        (1361, SfsRegime.OUTSIDE),  # LCS 1, perigee 2781 km, period 146 min
        (25933, SfsRegime.OUTSIDE),  # NAVSTAR 46 (GPS), MEO
        (14129, SfsRegime.OUTSIDE),  # AO-10, e 0.60
        (25867, SfsRegime.OUTSIDE),  # CXO, e 0.77
        (42965, SfsRegime.OUTSIDE),  # QZS-4, geosynchronous at 40 degrees
        (44204, SfsRegime.OUTSIDE),  # BEIDOU-3 IGSO-1, at 59 degrees
    ],
)
def test_regimes_of_real_orbits(number: int, regime: SfsRegime) -> None:
    primary = BY_NUMBER[number]
    assert sfs_regime(primary) is regime
    assert ScreeningVolume.sfs_handbook(primary) is SFS_HANDBOOK_VOLUMES[regime]


def sgp4_perigee(omm: Omm) -> float:
    satrec = omm.satrec()
    return float(satrec.altp * satrec.radiusearthkm * 1e3)


def sgp4_period(omm: Omm) -> float:
    """Period [s] of SGP4's un-Kozai'd mean motion, from sgp4's pure-Python model.

    ``Satrec.no`` is the element set's Kozai mean motion, not this one.
    """
    satrec = omm.satrec()
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
    return float(2.0 * math.pi / python.no_unkozai * 60.0)


def with_mean_motion(omm: Omm, target: float, of: str) -> Omm:
    """``omm`` with the mean motion that gives SGP4 perigee or period ``target``."""
    measure = sgp4_perigee if of == "perigee" else sgp4_period

    def miss(mean_motion: float) -> float:
        return measure(dataclasses.replace(omm, mean_motion=mean_motion)) - target

    found = brentq(miss, 0.2 * omm.mean_motion, 5.0 * omm.mean_motion, xtol=1e-16)
    return dataclasses.replace(omm, mean_motion=found)


@pytest.mark.parametrize(
    ("highest", "below", "above"),
    [
        (500e3, SfsRegime.LEO_1, SfsRegime.LEO_2),
        (750e3, SfsRegime.LEO_2, SfsRegime.LEO_3),
        (1200e3, SfsRegime.LEO_3, SfsRegime.LEO_4),
        (2000e3, SfsRegime.LEO_4, SfsRegime.OUTSIDE),
    ],
)
def test_leo_regimes_are_bounded_by_the_sgp4_perigee(
    highest: float, below: SfsRegime, above: SfsRegime
) -> None:
    calsphere = BY_NUMBER[900]
    assert sfs_regime(with_mean_motion(calsphere, highest - 1.0, "perigee")) is below
    assert sfs_regime(with_mean_motion(calsphere, highest + 1.0, "perigee")) is above


def test_a_decaying_orbit_is_leo_1() -> None:
    assert sfs_regime(with_mean_motion(BY_NUMBER[31631], -50e3, "perigee")) is (
        SfsRegime.LEO_1
    )


def test_eccentricity_bounds_the_leo_and_deep_space_regimes() -> None:
    low = BY_NUMBER[270214]  # perigee 431 km
    assert sfs_regime(dataclasses.replace(low, eccentricity=0.249)) is SfsRegime.LEO_1
    assert sfs_regime(dataclasses.replace(low, eccentricity=SFS_ECCENTRICITY)) is (
        SfsRegime.OUTSIDE
    )
    geo = BY_NUMBER[25924]
    assert sfs_regime(dataclasses.replace(geo, eccentricity=0.2)) is (
        SfsRegime.DEEP_SPACE
    )
    assert sfs_regime(dataclasses.replace(geo, eccentricity=0.25)) is (
        SfsRegime.OUTSIDE
    )


def test_deep_space_is_bounded_by_period_and_inclination() -> None:
    geo = BY_NUMBER[25924]
    lowest, highest = SFS_DEEP_SPACE_PERIOD
    # The period of the Kozai mean motion is ~3 s shorter here: 1 s off a
    # bound tells the two readings apart.
    for period, regime in [
        (lowest - 1.0, SfsRegime.OUTSIDE),
        (lowest + 1.0, SfsRegime.DEEP_SPACE),
        (highest - 1.0, SfsRegime.DEEP_SPACE),
        (highest + 1.0, SfsRegime.OUTSIDE),
    ]:
        assert sfs_regime(with_mean_motion(geo, period, "period")) is regime
    tilted = math.nextafter(SFS_DEEP_SPACE_INCLINATION, 0.0)
    assert sfs_regime(dataclasses.replace(geo, inclination=tilted)) is (
        SfsRegime.DEEP_SPACE
    )
    assert sfs_regime(
        dataclasses.replace(geo, inclination=SFS_DEEP_SPACE_INCLINATION)
    ) is (SfsRegime.OUTSIDE)


def test_regimes_need_an_sgp4_element_set() -> None:
    beyond = dataclasses.replace(BY_NUMBER[900], catalog_number=340_000)
    with pytest.raises(UnsupportedElements, match="339999"):
        sfs_regime(beyond)


def test_volume_for_a_primary() -> None:
    leo, geo = BY_NUMBER[31631], BY_NUMBER[25924]
    assert volume_for(DEFAULT_VOLUME, leo) is DEFAULT_VOLUME
    rule = ScreeningVolume.sfs_handbook
    assert volume_for(rule, leo) is SFS_HANDBOOK_VOLUMES[SfsRegime.LEO_1]
    assert volume_for(rule, geo) is SFS_HANDBOOK_VOLUMES[SfsRegime.DEEP_SPACE]
    # The rule pickles (for worker processes).
    assert (
        pickle.loads(pickle.dumps(rule))(geo)
        is SFS_HANDBOOK_VOLUMES[SfsRegime.DEEP_SPACE]
    )


def test_a_volume_rule_must_return_a_volume() -> None:
    with pytest.raises(TypeError, match="must return a ScreeningVolume, got float"):
        volume_for(lambda omm: 5e3, BY_NUMBER[31631])  # type: ignore[arg-type,return-value]


def test_volumes_pickle() -> None:
    volume = SFS_HANDBOOK_VOLUMES[SfsRegime.LEO_2]
    copy = pickle.loads(pickle.dumps(volume))
    assert copy == volume
    assert copy.label == volume.label
