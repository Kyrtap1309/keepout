"""Builders, constants and Hypothesis strategies shared by the test packages."""

import math
import sqlite3
from pathlib import Path
from typing import TypedDict, Unpack

import astropy.units as u
import numpy as np
import numpy.typing as npt
import pytest
from astropy.time import Time
from hypothesis import assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.spatial.transform import Rotation

from keepout._types import FloatArray, override
from keepout.cdm import CovarianceMethod, Maneuverable, ObjectMetadata, make_cdm
from keepout.conjunction import Conjunction
from keepout.covariance import Covariance
from keepout.events import ConjunctionEvent
from keepout.maneuver import BurnGrid
from keepout.objects import ObjectState
from keepout.orbits import EARTH_MU
from keepout.pc.result import PcResult
from keepout.propagation import ManeuveredPropagator
from keepout.propagation.forces import EARTH_RADIUS
from keepout.risk import PcKind, PcObservation
from keepout.state import StateVector
from keepout.units import DAY, HOUR, KM
from keepout.validation.cases import Suite, ValidationCase
from keepout.validation.criteria import TWO_D, BoundCriterion

CDM_DATA = Path(__file__).parent / "cdm" / "data"
"""Reference CDM files shared by the CDM and validation tests."""

B1_KVN = CDM_DATA / "b1_example.kvn"
"""CCSDS 508.0-B-1 annex example, KVN."""

KELVINS_EXCERPT = Path(__file__).parent / "sources" / "data" / "kelvins_excerpt.csv"
"""Every row of six events of ESA's Kelvins raw table."""

EPOCH = Time("2010-03-13T22:37:52.618", scale="utc")

LEO_R = np.array([7000e3, 0.0, 0.0])
"""A LEO position [m], GCRF."""

LEO_V = np.array([0.0, 7546.0, 0.0])
"""Circular speed at :data:`LEO_R`, prograde equatorial [m/s]."""

EARTH_ROTATION_RATE = 7.292115146706979e-5
"""Nominal mean angular velocity of the Earth [rad/s] (IERS Conventions 2010)."""

DIAGONAL_COVARIANCE = np.diag([100.0, 400.0, 25.0, 1e-2, 4e-2, 1e-2])
"""A valid 6x6 covariance [m, m/s]: sigmas 10, 20, 5 m and 0.1, 0.2, 0.1 m/s."""


TINY = 1e-300
"""Below this a relative comparison means nothing (near the float64 floor)."""


def approx(expected: object, rel: float) -> object:
    """``pytest.approx`` that is relative only.

    ``pytest.approx(x, rel=r)`` also accepts anything within its default
    ``abs=1e-12``, which makes every comparison with a tiny ``x`` vacuous; this
    keeps an absolute slack of :data:`TINY` only.
    """
    return pytest.approx(expected, rel=rel, abs=TINY)


def well_posed(r: FloatArray, v: FloatArray) -> bool:
    """Whether a state is far enough from degenerate for an RTN or orbital
    frame: off the origin, moving, and not along its own radius."""
    r_norm = float(np.linalg.norm(r))
    v_norm = float(np.linalg.norm(v))
    h_norm = float(np.linalg.norm(np.cross(r, v)))
    return r_norm > 1.0 and v_norm > 1e-3 and h_norm > 1e-3 * r_norm * v_norm


def known(covariance: Covariance | None) -> Covariance:
    """The covariance, which the test knows is given."""
    assert covariance is not None
    return covariance


def object_state(
    r: npt.ArrayLike,
    v: npt.ArrayLike,
    position_covariance: npt.ArrayLike | None = None,
    *,
    velocity_sigma: float = 0.0,
    epoch: Time = EPOCH,
    object_id: str = "TEST",
) -> ObjectState:
    """A GCRF object state with the given position covariance [m^2].

    ``velocity_sigma`` [m/s] adds an isotropic, uncorrelated velocity block.
    """
    matrix = np.zeros((6, 6))
    matrix[:3, :3] = np.eye(3) if position_covariance is None else position_covariance
    matrix[3:, 3:] = velocity_sigma**2 * np.eye(3)
    return ObjectState(
        object_id,
        StateVector(epoch, r, v, "GCRF"),
        Covariance(matrix, "GCRF", epoch),
    )


def without_covariance(item: ObjectState) -> ObjectState:
    """The object with its covariance unknown."""
    return ObjectState(item.object_id, item.state, None, item.hard_body_radius)


def drifting(velocity_sigma: float) -> Conjunction:
    """A slow crossing whose 3-D Pc differs from its 2-D Pc.

    The secondary drifts past 15 m away at 0.3 m/s, with 10 m position
    sigmas on both objects; over the minutes of the encounter the orbits
    curve and, with ``velocity_sigma`` [m/s] of the order of the relative
    speed, the relative velocity is itself uncertain. Pass
    ``window=DRIFT_WINDOW`` to the 3-D methods.
    """
    covariance = 100.0 * np.eye(3)
    primary = object_state(
        LEO_R, LEO_V, covariance, velocity_sigma=velocity_sigma, object_id="PRI"
    )
    secondary = object_state(
        LEO_R + np.array([15.0, 0.0, 0.0]),
        LEO_V + np.array([0.0, 0.0, 0.3]),
        covariance,
        velocity_sigma=velocity_sigma,
        object_id="SEC",
    )
    return Conjunction(primary, secondary, 5.0)


DRIFT_WINDOW = (-200.0, 200.0)
"""Window [s from TCA] that holds the whole :func:`drifting` encounter."""


def planar(
    miss: npt.ArrayLike, plane_covariance: npt.ArrayLike, radius: float
) -> Conjunction:
    """LEO head-on encounter with a chosen encounter-plane miss and covariance.

    The relative velocity is along GCRF -y, so the encounter plane is spanned
    by GCRF x and z: ``miss`` = (x, z) offset of the secondary [m] and
    ``plane_covariance`` = 2x2 (x, z) covariance [m^2], all on the primary.
    """
    dx, dz = np.asarray(miss, dtype=np.float64)
    block = np.asarray(plane_covariance, dtype=np.float64)
    position = np.eye(3)
    position[np.ix_([0, 2], [0, 2])] = block
    primary = object_state(LEO_R, LEO_V, position, object_id="PRI")
    secondary = object_state(
        LEO_R + np.array([dx, 0.0, dz]), -LEO_V, np.zeros((3, 3)), object_id="SEC"
    )
    return Conjunction(primary, secondary, radius)


def head_on(miss: float, sigma_x: float, sigma_z: float, radius: float) -> Conjunction:
    """:func:`planar` with the miss along x and covariance diag(sx^2, sz^2)."""
    return planar([miss, 0.0], np.diag([sigma_x**2, sigma_z**2]), radius)


TYPICAL_HEAD_ON = head_on(30.0, 10.0, 20.0, 5.0)
"""The workhorse encounter: a 15 km/s LEO head-on crossing, 30 m miss along
x, encounter-plane sigmas 10 m (x) and 20 m (z), 5 m hard-body radius; 2-D
Pc ~ 8.7e-4.

Sharing it is safe: a :class:`~keepout.conjunction.Conjunction` is frozen,
its states and covariances are read-only, and its cached plane depends on
nothing a test patches.
"""

UNBOUND_V = np.array([0.0, 15000.0, 0.0])
"""15 km/s at :data:`LEO_R`: above escape speed, so there is no orbit period."""


def unbound_crossing(z_offset: float = 0.0) -> Conjunction:
    """An unbound primary (:data:`UNBOUND_V`, 1e4 m sigmas) crossed at 1 m/s
    along z by a secondary (1 m sigmas) 30 m away along x and ``z_offset`` m
    along the relative velocity; 5 m hard-body radius."""
    primary = object_state(LEO_R, UNBOUND_V, 1e8 * np.eye(3))
    secondary = object_state(
        LEO_R + np.array([30.0, 0.0, z_offset]),
        UNBOUND_V + np.array([0.0, 0.0, 1.0]),
        np.eye(3),
    )
    return Conjunction(primary, secondary, 5.0)


def case_with_reference(
    reference: float,
    *,
    case_id: str = "case",
    absolute_tolerance: float | None = None,
) -> ValidationCase:
    """A 2-D validation case whose own Pc underflows to 0 (miss 100 sigma).

    ``absolute_tolerance`` replaces the default 2-D criterion by an absolute
    bound.
    """
    criterion = (
        TWO_D
        if absolute_tolerance is None
        else BoundCriterion(absolute=absolute_tolerance)
    )
    return ValidationCase(
        Suite.UNIT_TESTS,
        case_id,
        head_on(1000.0, 10.0, 10.0, 5.0),
        reference,
        criterion=criterion,
    )


def rotate(obj: ObjectState, rotation: FloatArray) -> ObjectState:
    """The same object with state and covariance rotated by ``rotation``."""
    block = np.zeros((6, 6))
    block[:3, :3] = block[3:, 3:] = rotation
    state = obj.state
    return ObjectState(
        obj.object_id,
        StateVector(state.epoch, rotation @ state.r, rotation @ state.v, "GCRF"),
        Covariance(block @ known(obj.covariance).matrix @ block.T, "GCRF", state.epoch),
        obj.hard_body_radius,
    )


@st.composite
def rotations(draw: st.DrawFn) -> FloatArray:
    """Uniform-ish proper rotation matrices from random quaternions."""
    quaternion = draw(arrays(np.float64, (4,), elements=st.floats(-1.0, 1.0)))
    assume(float(np.linalg.norm(quaternion)) > 0.1)
    matrix: FloatArray = Rotation.from_quat(quaternion, scalar_first=True).as_matrix()
    return matrix


@st.composite
def position_covariances(draw: st.DrawFn, scale: float = 100.0) -> FloatArray:
    """Random symmetric positive-definite 3x3 matrices [m^2].

    Root elements lie in ``[-scale, scale]``: sigmas roughly 1 m to ``3 scale``.
    """
    root: FloatArray = draw(
        arrays(np.float64, (3, 3), elements=st.floats(-scale, scale))
    )
    return root @ root.T + np.eye(3)


def shifted_back(conjunction: Conjunction, dt: float) -> Conjunction:
    """The conjunction with both states moved back by ``v dt`` along a line.

    Its straight-line closest approach is then ``dt`` seconds after the
    states' epoch; the covariances are unchanged.
    """

    def move(item: ObjectState) -> ObjectState:
        state = item.state
        moved = StateVector(state.epoch, state.r - state.v * dt, state.v, state.frame)
        return ObjectState(
            item.object_id, moved, item.covariance, item.hard_body_radius
        )

    return Conjunction(
        move(conjunction.primary),
        move(conjunction.secondary),
        conjunction.combined_hard_body_radius,
    )


class ObservationChanges(TypedDict, total=False):
    """Optional fields of a :class:`PcObservation`."""

    kind: PcKind
    ci_high: float | None
    error: str | None
    message_id: str | None
    reported_pc: float | None
    diagnostics: tuple[str, ...]
    result: PcResult | None


def at(
    days: float, pc: float | None, **changes: Unpack[ObservationChanges]
) -> PcObservation:
    """An observation ``days`` before TCA; ``pc=None`` makes it a failed one."""
    if pc is None:
        changes.setdefault("error", "GeometryError: no plane")
    return PcObservation(time_to_tca=days * DAY, source="test", pc=pc, **changes)


def series(*pairs: tuple[float, float | None]) -> list[PcObservation]:
    """Observations from ``(days before TCA, Pc)`` pairs."""
    return [at(days, pc) for days, pc in pairs]


class DifferenceWidened(ManeuveredPropagator):
    """A burn whose radius bounds are its base's widened by the whole
    difference ``|d|``, as where the base fails: far wider than the radius
    after the burn, as if the burn moved it by ``|d|``."""

    __slots__ = ()

    @override
    def _burned_radius(
        self, first: float, last: float, lowest: float
    ) -> tuple[float, float] | None:
        return None


def base_revision_store(path: Path) -> None:
    """Make ``path`` a store at Alembic's base revision, as a keepout before
    the first migration left it: Alembic's table, empty, and nothing else."""
    connection = sqlite3.connect(path)
    with connection:
        connection.execute(
            "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL,"
            " CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num))"
        )
    connection.close()


CROSSING_TCA = Time("2026-10-06T00:00:00", scale="utc")
"""The TCA of :func:`leo_crossing`."""

CROSSING_RADIUS = EARTH_RADIUS + 630 * KM
"""The orbital radius of :func:`leo_crossing`'s objects [m]."""

SMALL_GRID = BurnGrid(steps_per_orbit=1, delta_v_per_direction=6)
"""A burn grid of a few dozen cells (about 70 at 25 h), planned in a
fraction of a second."""


def _crossing_object(object_id: str, r: npt.ArrayLike, angle: float) -> ObjectState:
    """An object of :func:`leo_crossing`, at ``r`` on a circular orbit
    inclined by ``angle``."""
    speed = math.sqrt(EARTH_MU / CROSSING_RADIUS)
    v = speed * np.array([0.0, math.cos(angle), math.sin(angle)])
    covariance = np.diag([100.0**2] * 3 + [0.01**2] * 3)
    return ObjectState(
        object_id,
        StateVector(CROSSING_TCA, r, v, "GCRF"),
        Covariance(covariance, "GCRF", CROSSING_TCA),
        10.0,
    )


def leo_crossing() -> ConjunctionEvent:
    """A fast crossing in LEO, 150 m apart radially, 25 h before TCA, as the
    planner's tests make it: Pc about 5e-3."""
    conjunction = Conjunction(
        _crossing_object("1001", [CROSSING_RADIUS, 0.0, 0.0], 0.9),
        _crossing_object("2002", [CROSSING_RADIUS + 150.0, 0.0, 0.0], -0.9),
        20.0,
    )
    method = CovarianceMethod.CALCULATED
    cdm = make_cdm(
        conjunction,
        ObjectMetadata(
            "PRI", "2020-001A", maneuverable=Maneuverable.YES, covariance_method=method
        ),
        ObjectMetadata("SEC", "2019-001A", covariance_method=method),
        created=CROSSING_TCA - 25 * HOUR * u.s,
    )
    return ConjunctionEvent.from_cdms([cdm], event_id="EVENT-7")
