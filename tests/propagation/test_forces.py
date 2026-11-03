import itertools
import math
import pickle
from typing import Any, Final

import numpy as np
import pytest
from astropy.time import Time
from scipy.spatial.transform import Rotation
from scipy.special import eval_legendre

from keepout._types import FloatArray
from keepout.orbits import EARTH_MU
from keepout.propagation._atmosphere import density
from keepout.propagation.forces import EARTH_RADIUS, ForceModel, earth_pole
from keepout.units import KM

EARTH_ROTATION: Final = 7.292115e-5
"""Earth rotation [rad/s], as the brief of the force model gives it."""

NORMALISED_ZONALS: Final = {
    2: -0.484165143790815e-3,
    3: 0.957161207093473e-6,
    4: 0.539965866638991e-6,
    5: 0.686702913736681e-7,
    6: -0.149953927978527e-6,
}
"""EGM2008 tide-free C̄n0, copied from the coefficient file."""

ATMOSPHERE: Final = [
    (0, 1.225, 7.249),
    (25, 3.899e-2, 6.349),
    (30, 1.774e-2, 6.682),
    (40, 3.972e-3, 7.554),
    (50, 1.057e-3, 8.382),
    (60, 3.206e-4, 7.714),
    (70, 8.770e-5, 6.549),
    (80, 1.905e-5, 5.799),
    (90, 3.396e-6, 5.382),
    (100, 5.297e-7, 5.877),
    (110, 9.661e-8, 7.263),
    (120, 2.438e-8, 9.473),
    (130, 8.484e-9, 12.636),
    (140, 3.845e-9, 16.149),
    (150, 2.070e-9, 22.523),
    (180, 5.464e-10, 29.740),
    (200, 2.789e-10, 37.105),
    (250, 7.248e-11, 45.546),
    (300, 2.418e-11, 53.628),
    (350, 9.518e-12, 53.298),
    (400, 3.725e-12, 58.515),
    (450, 1.585e-12, 60.828),
    (500, 6.967e-13, 63.822),
    (600, 1.454e-13, 71.835),
    (700, 3.614e-14, 88.667),
    (800, 1.170e-14, 124.64),
    (900, 5.245e-15, 181.05),
    (1000, 3.019e-15, 268.00),
]
"""Vallado, table 8-4: base altitude [km], density [kg/m³], scale height [km]."""

Z: Final = np.array([0.0, 0.0, 1.0])
"""A pole along the z axis."""


def zonal_coefficient(n: int) -> float:
    """Jn of a degree, from its normalised coefficient."""
    return -NORMALISED_ZONALS[n] * math.sqrt(2 * n + 1)


def positions() -> FloatArray:
    """Five positions at low to high altitude, at different latitudes [m]."""
    return np.array(
        [
            [6_778_000.0, 0.0, 0.0],
            [4_000_000.0, 3_000_000.0, 4_500_000.0],
            [-1_000_000.0, 5_000_000.0, -5_000_000.0],
            [100_000.0, -200_000.0, 7_200_000.0],
            [30_000_000.0, 28_000_000.0, 1_000_000.0],
        ]
    )


def velocities() -> FloatArray:
    """Five velocities of the same size as orbital ones [m/s]."""
    return np.array(
        [
            [0.0, 7_600.0, 0.0],
            [-4_000.0, 6_000.0, 500.0],
            [7_000.0, 1_000.0, -2_000.0],
            [7_400.0, 100.0, -50.0],
            [-2_000.0, 2_100.0, 300.0],
        ]
    )


def two_body(r: FloatArray) -> FloatArray:
    """Point-mass acceleration of positions ``(N, 3)``."""
    radius = np.linalg.norm(r, axis=1, keepdims=True)
    acceleration: FloatArray = -EARTH_MU * r / radius**3
    return acceleration


def zonal_potential(r: FloatArray, pole: FloatArray, degree: int) -> float:
    """The zonal part of the potential at one position [m²/s²]."""
    radius = float(np.linalg.norm(r))
    s = float(r @ pole) / radius
    return -sum(
        EARTH_MU
        / radius
        * zonal_coefficient(n)
        * (EARTH_RADIUS / radius) ** n
        * float(eval_legendre(n, s))
        for n in range(2, degree + 1)
    )


def zonal_gradient(r: FloatArray, pole: FloatArray, degree: int) -> FloatArray:
    """Central differences of :func:`zonal_potential` at one position."""
    step = 10.0
    offsets = step * np.eye(3)
    return np.array(
        [
            (
                zonal_potential(r + offset, pole, degree)
                - zonal_potential(r - offset, pole, degree)
            )
            / (2.0 * step)
            for offset in offsets
        ]
    )


def test_degree_zero_is_two_body() -> None:
    r = positions()
    acceleration = ForceModel(Z, degree=0).acceleration(r, velocities())
    assert acceleration == pytest.approx(two_body(r), rel=1e-14, abs=0.0)


def test_j2_matches_closed_form() -> None:
    r = positions()
    j2 = zonal_coefficient(2)
    radius = np.linalg.norm(r, axis=1)
    x, y, z = r.T
    ratio = 1.5 * j2 * (EARTH_RADIUS / radius) ** 2
    sine2 = (z / radius) ** 2
    scale = -EARTH_MU / radius**3
    expected = np.column_stack(
        (
            scale * x * (1.0 - ratio * (5.0 * sine2 - 1.0)),
            scale * y * (1.0 - ratio * (5.0 * sine2 - 1.0)),
            scale * z * (1.0 - ratio * (5.0 * sine2 - 3.0)),
        )
    )
    acceleration = ForceModel(Z, degree=2).acceleration(r, velocities())
    assert acceleration == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("degree", [2, 3, 4, 5, 6])
def test_acceleration_is_gradient_of_potential(degree: int) -> None:
    # Against the zonal terms alone: J6 is 1e-3 of J2, and would hide in the
    # central force.
    pole = np.array([0.6, 0.0, 0.8])
    r = positions()
    zonal = ForceModel(pole, degree=degree).acceleration(r, velocities()) - two_body(r)
    for row, acceleration in zip(r, zonal, strict=True):
        gradient = zonal_gradient(row, pole, degree)
        assert acceleration == pytest.approx(gradient, rel=1e-7, abs=1e-14)


def test_the_default_degree_is_six() -> None:
    r = positions()
    default = ForceModel(Z).acceleration(r, velocities())
    assert np.array_equal(
        default, ForceModel(Z, degree=6).acceleration(r, velocities())
    )


def test_batch_equals_single() -> None:
    r, v = positions(), velocities()
    force = ForceModel(Z, degree=6, drag_term=0.01)
    batch = force.acceleration(r, v)
    single = [
        force.acceleration(r[index : index + 1], v[index : index + 1])[0]
        for index in range(len(r))
    ]
    assert np.array_equal(batch, single)


def test_zonal_terms_follow_the_pole() -> None:
    rotation = Rotation.from_euler("zyx", [0.3, -0.7, 1.1]).as_matrix()
    r, v = positions(), velocities()
    fixed = ForceModel(Z).acceleration(r, v)
    rotated = ForceModel(rotation @ Z).acceleration(r @ rotation.T, v @ rotation.T)
    assert rotated == pytest.approx(fixed @ rotation.T, rel=1e-13, abs=1e-12)


def drag_of(force: ForceModel, r: FloatArray, v: FloatArray) -> FloatArray:
    """The acceleration of a model less that of the model without drag."""
    plain = ForceModel(force.pole, degree=force.degree)
    difference: FloatArray = force.acceleration(r, v) - plain.acceleration(r, v)
    return difference


def test_drag_opposes_air_relative_velocity() -> None:
    r, v = positions()[:4], velocities()[:4]
    pole = np.array([0.0, 0.6, 0.8])
    relative = v - EARTH_ROTATION * np.cross(pole, r)
    drag = drag_of(ForceModel(pole, drag_term=0.01), r, v)
    assert np.all(np.sum(drag * relative, axis=1) < 0.0)


def test_drag_has_the_size_of_the_formula() -> None:
    r = np.array([[EARTH_RADIUS + 300_000.0, 0.0, 0.0]])
    v = np.array([[0.0, 7_700.0, 0.0]])
    drag_term = 0.02
    # The air at the equator moves along +y, with the Earth.
    air_speed = 7_700.0 - EARTH_ROTATION * r[0, 0]
    expected = -0.5 * 2.418e-11 * drag_term * air_speed**2
    drag = drag_of(ForceModel(Z, drag_term=drag_term), r, v)
    assert drag == pytest.approx(np.array([[0.0, expected, 0.0]]), rel=1e-6, abs=1e-12)


def test_the_air_turns_about_the_pole() -> None:
    # Air at 300 km: over the orbit speed, the Earth's turn is a 6 % effect.
    rotation = Rotation.from_euler("zyx", [0.3, -0.7, 1.1]).as_matrix()
    r, v = positions()[:2], velocities()[:2]
    fixed = drag_of(ForceModel(Z, drag_term=0.01), r, v)
    tilted = ForceModel(rotation @ Z, drag_term=0.01)
    drag = drag_of(tilted, r @ rotation.T, v @ rotation.T)
    assert drag == pytest.approx(fixed @ rotation.T, rel=1e-7, abs=0.0)


def test_drag_scales_with_the_drag_term() -> None:
    r, v = positions()[:1], velocities()[:1]
    light = drag_of(ForceModel(Z, drag_term=0.01), r, v)
    heavy = drag_of(ForceModel(Z, drag_term=0.03), r, v)
    assert heavy == pytest.approx(3.0 * light, rel=1e-6)


def test_no_drag_term_means_no_drag() -> None:
    r = positions()
    still = ForceModel(Z).acceleration(r, np.zeros_like(r))
    moving = ForceModel(Z).acceleration(r, velocities())
    assert np.array_equal(still, moving)


def test_derivative_is_velocity_then_acceleration() -> None:
    r, v = positions(), velocities()
    force = ForceModel(Z, drag_term=0.01)
    state = np.hstack((r, v)).ravel()
    expected = np.hstack((v, force.acceleration(r, v))).ravel()
    assert np.array_equal(force.derivative(state), expected)


def test_pole_near_z_in_the_2020s() -> None:
    pole = earth_pole(Time("2026-01-01T00:00:00", scale="utc"))
    assert np.linalg.norm(pole) == pytest.approx(1.0, abs=1e-14)
    assert math.degrees(math.acos(pole[2])) < 0.5
    # Precession tips the pole towards +x, about 2004 arcsec per century since
    # J2000: a sign the transposed matrix would get wrong.
    assert pole[0] == pytest.approx(math.radians(2004.19 * 0.26 / 3600.0), abs=1e-4)


def test_pole_is_of_one_time() -> None:
    with pytest.raises(ValueError, match="epoch must be a single time"):
        earth_pole(Time(["2026-01-01", "2026-01-02"], scale="utc"))


def test_density_follows_the_table() -> None:
    table = np.array(ATMOSPHERE)
    assert np.array_equal(density(table[:, 0] * KM), table[:, 1])


def test_density_decays_exponentially_within_a_band() -> None:
    altitude, base, scale = ATMOSPHERE[20]
    above = np.array([(altitude + 30.0) * KM])
    expected = base * math.exp(-30.0 / scale)
    # abs=0: approx's default absolute tolerance would swallow such small values.
    assert density(above) == pytest.approx(np.array([expected]), rel=1e-12, abs=0.0)


@pytest.mark.parametrize(("row", "above"), list(itertools.pairwise(ATMOSPHERE)))
def test_density_decreases_with_altitude(
    row: tuple[int, float, float], above: tuple[int, float, float]
) -> None:
    # Within a band and from base to base. Not across the edge of a band: the
    # table is not continuous, and the first band ends 0.14 % under the second.
    altitudes = np.linspace(row[0], above[0], 21)[:-1] * KM
    assert np.all(np.diff(density(altitudes)) < 0.0)
    assert above[1] < row[1]


def test_density_above_the_table_extends_the_last_row() -> None:
    expected = 3.019e-15 * math.exp(-200.0 / 268.00)
    assert density(np.array([1_200.0 * KM])) == pytest.approx(
        np.array([expected]), rel=1e-12, abs=0.0
    )


def test_density_below_the_surface_stays_at_sea_level() -> None:
    assert np.array_equal(density(np.array([-6_000.0 * KM])), [1.225])


def test_force_model_pickles() -> None:
    force = ForceModel(np.array([0.0, 0.6, 0.8]), degree=4, drag_term=0.01)
    copy = pickle.loads(pickle.dumps(force))
    r, v = positions(), velocities()
    assert np.array_equal(copy.acceleration(r, v), force.acceleration(r, v))
    assert not copy.pole.flags.writeable


def test_pole_is_a_read_only_copy() -> None:
    pole = Z.copy()
    force = ForceModel(pole)
    pole[0] = 1.0
    assert np.array_equal(force.pole, Z)
    assert not force.pole.flags.writeable


@pytest.mark.parametrize(
    ("pole", "match"),
    [
        (np.array([0.0, 1.0]), "pole must have shape"),
        (np.array([0.0, 0.0, 2.0]), "pole must be a unit vector"),
        (np.array([0.0, 0.0, np.nan]), "pole must be finite"),
    ],
)
def test_a_bad_pole_is_refused(pole: FloatArray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ForceModel(pole)


def test_degree_and_drag_term_are_keyword_only() -> None:
    constructor: Any = ForceModel  # the type checkers refuse the call too
    with pytest.raises(TypeError, match="positional"):
        constructor(Z, 2)


@pytest.mark.parametrize("degree", [-1, 7])
def test_a_degree_outside_the_model_is_refused(degree: int) -> None:
    with pytest.raises(ValueError, match="degree must be"):
        ForceModel(Z, degree=degree)


def test_a_negative_drag_term_is_refused() -> None:
    with pytest.raises(ValueError, match="drag_term must be finite"):
        ForceModel(Z, drag_term=-0.01)
