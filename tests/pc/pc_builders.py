"""Hypothesis strategies for Pc tests."""

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
from builders import LEO_R, LEO_V, head_on, object_state, planar, position_covariances
from hypothesis import assume
from hypothesis import strategies as st

from keepout._types import FloatArray
from keepout.conjunction import Conjunction
from keepout.encounter import EncounterPlane

_BASE_PLANE = head_on(1.0, 1.0, 1.0, 1.0).plane

SIDE_BY_SIDE_OFFSET = np.array([0.0, 0.0, 10.0])
"""Secondary minus primary position of :func:`side_by_side` [m]."""

SIDE_BY_SIDE_VARIANCE = 25.0
"""Position variance of each object of :func:`side_by_side`, on every axis [m^2]."""


def side_by_side(velocity: FloatArray = LEO_V) -> Conjunction:
    """Two objects 10 m apart across track with the same ``velocity``: zero
    relative velocity, so there is no encounter plane.

    Both have 5 m position sigmas and an exact velocity (no velocity block);
    the hard-body radius is 5 m. With the default circular LEO velocity Hall's
    3D-Nc is defined (see :data:`reference_pc.SIDE_BY_SIDE_NC`).
    """
    covariance = SIDE_BY_SIDE_VARIANCE * np.eye(3)
    primary = object_state(LEO_R, velocity, covariance, object_id="PRI")
    secondary = object_state(
        LEO_R + SIDE_BY_SIDE_OFFSET, velocity, covariance, object_id="SEC"
    )
    return Conjunction(primary, secondary, 5.0)


@st.composite
def unit_vectors(draw: st.DrawFn) -> FloatArray:
    """Unit vectors uniform on the sphere (no rejection)."""
    azimuth = draw(st.floats(0.0, 2.0 * math.pi))
    height = draw(st.floats(-1.0, 1.0))
    ring = math.sqrt(1.0 - height * height)
    return np.array([ring * math.cos(azimuth), ring * math.sin(azimuth), height])


@st.composite
def conjunctions(draw: st.DrawFn) -> Conjunction:
    """Random LEO conjunctions at TCA in any orientation: relative speed 50 m/s
    to 15 km/s, full 3x3 covariances, and a miss of up to 6 Mahalanobis units
    on the encounter plane (so Pc never underflows to 0)."""
    unit = draw(unit_vectors())
    relative_velocity = draw(st.floats(50.0, 15000.0)) * unit
    first = draw(position_covariances(300.0))
    second = draw(position_covariances(300.0))
    # Put the secondary exactly at closest approach: no offset along v.
    across = draw(unit_vectors())
    across = across - float(across @ unit) * unit
    assume(float(np.linalg.norm(across)) > 1e-3)
    across = across / np.linalg.norm(across)
    # Miss of k plane Mahalanobis units: d^2 (P^-1)_00 = k^2 in the plane
    # axes (across, v x across), P the combined covariance projected there.
    basis = np.stack([across, np.cross(unit, across)])
    plane = basis @ (first + second) @ basis.T
    assume(float(np.linalg.det(plane)) > 1e-12 * float(np.trace(plane)) ** 2)
    precision = np.linalg.inv(plane)
    distance = draw(st.floats(0.0, 6.0)) / math.sqrt(float(precision[0, 0]))
    offset = distance * across
    primary = object_state(LEO_R, LEO_V, first, object_id="PRI")
    secondary = object_state(
        LEO_R + offset, LEO_V + relative_velocity, second, object_id="SEC"
    )
    radius = draw(st.floats(1.0, 50.0))
    return Conjunction(primary, secondary, radius)


@dataclass(frozen=True)
class PlaneCase:
    """An encounter-plane Gaussian given by its principal axes.

    The covariance is ``diag(sigma_u^2, sigma_v^2)`` rotated by ``angle``; the miss is
    ``(miss_u, miss_v)`` in those axes. Knowing the principal axes lets tests use
    :func:`reference_pc.principal_axes_pc` as an independent oracle.
    """

    miss_u: float
    miss_v: float
    sigma_u: float
    sigma_v: float
    angle: float
    radius: float

    @property
    def rotation(self) -> FloatArray:
        c, s = math.cos(self.angle), math.sin(self.angle)
        return np.array([[c, -s], [s, c]])

    @property
    def miss(self) -> FloatArray:
        miss: FloatArray = self.rotation @ np.array([self.miss_u, self.miss_v])
        return miss

    @property
    def covariance(self) -> FloatArray:
        return (
            self.rotation
            @ np.diag([self.sigma_u**2, self.sigma_v**2])
            @ self.rotation.T
        )

    def plane(self, scale: float = 1.0) -> EncounterPlane:
        """The exact plane, every length multiplied by ``scale``.

        Built directly rather than from states: differencing two ~7000 km
        positions would blur sub-micrometre misses by ~1e-9 m.
        """
        return dataclasses.replace(
            _BASE_PLANE,
            miss_vector=scale * self.miss,
            covariance=scale**2 * self.covariance,
        )

    def conjunction(self) -> Conjunction:
        """The case as a LEO conjunction (plane rebuilt from the states)."""
        return planar(self.miss, self.covariance, self.radius)


ANISOTROPIC = (
    PlaneCase(3.0, 4.0, 20.0, 5.0, 0.3, 10.0),
    PlaneCase(-40.0, 2.0, 30.0, 1.0, 1.1, 5.0),
    PlaneCase(0.0, 0.0, 100.0, 0.1, 2.0, 20.0),
    PlaneCase(250.0, -30.0, 500.0, 50.0, 0.7, 15.0),
    PlaneCase(5.0, 0.5, 1.0, 0.2, 2.9, 3.0),  # covariance smaller than the body
    PlaneCase(0.0, 25.0, 1000.0, 10.0, 0.0, 10.0),  # Pc ~ 1e-6, aspect 100:1
)
"""Correlated, off-axis plane Gaussians for checks against the oracle."""


@st.composite
def plane_cases(draw: st.DrawFn) -> PlaneCase:
    """Plane Gaussians from round to line-like (aspect up to 1e4), any
    orientation, miss within 6 sigma, hard body 0.1x to 10x the wide sigma.

    Sigmas are drawn on a log scale and the miss in sigma units, so most
    draws have a Pc well above zero.
    """
    sigma_u = 10.0 ** draw(st.floats(-1.0, 3.0))
    sigma_v = sigma_u * 10.0 ** draw(st.floats(-4.0, 0.0))
    return PlaneCase(
        miss_u=sigma_u * draw(st.floats(-6.0, 6.0)),
        miss_v=sigma_v * draw(st.floats(-6.0, 6.0)),
        sigma_u=sigma_u,
        sigma_v=sigma_v,
        angle=draw(st.floats(0.0, math.pi)),
        radius=sigma_u * 10.0 ** draw(st.floats(-1.0, 1.0)),
    )
