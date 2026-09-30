"""Encounter-plane geometry for a short-duration encounter.

With relative position ``r = r_s - r_p`` and relative velocity
``v = v_s - v_p`` (secondary minus primary, GCRF) the axes are
(Foster & Estes 1992; Chan 2008):

* ``y = v/|v|`` — normal to the encounter plane;
* ``x`` — unit component of ``r`` perpendicular to ``v``;
* ``z = x x y``.

The plane is spanned by ``(x, z)``; the miss vector is ``(|r_perp|, 0)``. If
``r`` is parallel to ``v`` (a direct hit in the linear model), ``x`` is any
unit vector perpendicular to ``v`` — Pc does not depend on that choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from astropy.time import Time

from keepout._types import FloatArray, symmetric_part
from keepout.errors import GeometryError
from keepout.objects import ObjectState, require_same_epoch

Role: TypeAlias = Literal["primary", "secondary"]
"""Which object of a conjunction."""


@dataclass(frozen=True, eq=False)
class EncounterPlane:
    """A conjunction projected onto the encounter plane.

    Parameters
    ----------
    tca
        Time of closest approach (the primary's epoch).
    miss_vector
        Relative position in the plane axes ``(x, z)`` [m].
    covariance
        Combined position covariance projected onto the plane [m^2].
    relative_position
        Secondary minus primary position, GCRF [m].
    relative_velocity
        Secondary minus primary velocity, GCRF [m/s].
    relative_speed
        Length of ``relative_velocity`` [m/s].
    combined_covariance
        Sum of the known 3x3 position covariances, GCRF [m^2].
    basis
        2x3 matrix whose rows are the plane axes ``x`` and ``z`` in GCRF.
    unknown_covariance
        Objects whose covariance is unknown; they contribute nothing to
        ``covariance`` and ``combined_covariance``.
    """

    tca: Time
    miss_vector: FloatArray
    covariance: FloatArray
    relative_position: FloatArray
    relative_velocity: FloatArray
    relative_speed: float
    combined_covariance: FloatArray
    basis: FloatArray
    unknown_covariance: tuple[Role, ...] = ()

    @property
    def miss_distance(self) -> float:
        """Length of the miss vector [m]."""
        return float(np.linalg.norm(self.miss_vector))

    @property
    def direction(self) -> FloatArray:
        """Unit relative velocity: the normal of the plane."""
        return self.relative_velocity / self.relative_speed

    @property
    def closest_approach_offset(self) -> float:
        """Relative-position component along the relative velocity [m].

        Zero when the states are at the straight-line closest approach.
        """
        return float(self.relative_position @ self.direction)


def _perpendicular(unit: FloatArray) -> FloatArray:
    axis = np.zeros(3)
    axis[int(np.argmin(np.abs(unit)))] = 1.0
    other = np.cross(unit, axis)
    return other / np.linalg.norm(other)


def encounter_plane(primary: ObjectState, secondary: ObjectState) -> EncounterPlane:
    """Project a conjunction onto the encounter plane.

    Parameters
    ----------
    primary
        The protected object.
    secondary
        The other object, at the same epoch.

    Raises
    ------
    GeometryError
        If the epochs differ or the relative velocity is zero.
    """
    require_same_epoch(
        secondary.state.epoch, primary.state.epoch, "primary and secondary"
    )
    r = secondary.state.r - primary.state.r
    v = secondary.state.v - primary.state.v
    speed = float(np.linalg.norm(v))
    if speed == 0.0:
        raise GeometryError(
            "relative velocity is zero: the encounter plane is undefined"
        )
    y = v / speed
    r_perp = r - float(r @ y) * y
    r_perp_norm = float(np.linalg.norm(r_perp))
    if r_perp_norm > 1e-12 * max(float(np.linalg.norm(r)), 1.0):
        x = r_perp / r_perp_norm
    else:
        x = _perpendicular(y)
    basis = np.stack([x, np.cross(x, y)])
    combined = np.zeros((3, 3))
    unknown: list[Role] = []
    objects: tuple[tuple[Role, ObjectState], ...] = (
        ("primary", primary),
        ("secondary", secondary),
    )
    for role, item in objects:
        if item.covariance is None:
            unknown.append(role)
        else:
            combined = combined + item.covariance.position
    projected = basis @ combined @ basis.T
    return EncounterPlane(
        tca=primary.state.epoch,
        miss_vector=basis @ r,
        covariance=symmetric_part(projected),
        relative_position=r,
        relative_velocity=v,
        relative_speed=speed,
        combined_covariance=combined,
        basis=basis,
        unknown_covariance=tuple(unknown),
    )
