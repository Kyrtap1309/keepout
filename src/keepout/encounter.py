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
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
from astropy.time import Time

from keepout._linalg import symmetric_part
from keepout._types import FloatArray, read_only
from keepout.errors import GeometryError
from keepout.frames import PARALLEL_RTOL

if TYPE_CHECKING:
    from keepout.conjunction import Conjunction

__all__ = ["EncounterPlane", "Role", "encounter_plane", "sigma_along"]

Role: TypeAlias = Literal["primary", "secondary"]
"""Which object of a conjunction."""


@dataclass(frozen=True, eq=False, slots=True)
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
    combined_covariance
        Sum of the known 3x3 position covariances, GCRF [m^2].
    basis
        2x3 matrix whose rows are the plane axes ``x`` and ``z`` in GCRF.
    unknown_covariance
        Objects whose covariance is unknown; they contribute nothing to
        ``covariance`` and ``combined_covariance``.

    Notes
    -----
    The arrays are stored as read-only float64 copies: the plane is cached on
    :attr:`Conjunction.plane <keepout.conjunction.Conjunction.plane>`.
    """

    tca: Time
    miss_vector: FloatArray
    covariance: FloatArray
    relative_position: FloatArray
    relative_velocity: FloatArray
    combined_covariance: FloatArray
    basis: FloatArray
    unknown_covariance: tuple[Role, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "miss_vector",
            "covariance",
            "relative_position",
            "relative_velocity",
            "combined_covariance",
            "basis",
        ):
            array = np.array(getattr(self, name), dtype=np.float64)
            read_only(array)
            object.__setattr__(self, name, array)

    @property
    def miss_distance(self) -> float:
        """Length of the miss vector [m]."""
        return float(np.linalg.norm(self.miss_vector))

    @property
    def relative_speed(self) -> float:
        """Length of the relative velocity [m/s]."""
        return float(np.linalg.norm(self.relative_velocity))

    @property
    def direction(self) -> FloatArray:
        """Unit relative velocity: the normal of the plane."""
        return self.relative_velocity / self.relative_speed

    @property
    def distance_along_velocity(self) -> float:
        """Relative-position component along the relative velocity [m].

        Zero when the states are at the straight-line closest approach.
        """
        return float(self.relative_position @ self.direction)


def sigma_along(direction: FloatArray, covariance: FloatArray) -> float:
    """Standard deviation of a position covariance along a unit vector [m]."""
    return float(np.sqrt(max(float(direction @ covariance @ direction), 0.0)))


def _perpendicular(unit: FloatArray) -> FloatArray:
    axis = np.zeros(3)
    axis[int(np.argmin(np.abs(unit)))] = 1.0
    other = np.cross(unit, axis)
    return other / np.linalg.norm(other)


def encounter_plane(conjunction: Conjunction) -> EncounterPlane:
    """A conjunction projected onto the encounter plane.

    Use :attr:`Conjunction.plane <keepout.conjunction.Conjunction.plane>`,
    which caches the result.

    Parameters
    ----------
    conjunction
        Both objects' states at TCA, and their covariances where known.

    Raises
    ------
    GeometryError
        If the relative velocity is zero.
    """
    r = conjunction.relative_position
    v = conjunction.relative_velocity
    speed = float(np.linalg.norm(v))
    if speed == 0.0:
        raise GeometryError(
            "relative velocity is zero: the encounter plane is undefined"
        )
    y = v / speed
    r_perp = r - float(r @ y) * y
    r_perp_norm = float(np.linalg.norm(r_perp))
    if r_perp_norm > PARALLEL_RTOL * max(float(np.linalg.norm(r)), 1.0):
        x = r_perp / r_perp_norm
    else:
        x = _perpendicular(y)
    basis = np.stack([x, np.cross(x, y)])
    combined = np.zeros((3, 3))
    for _, item in conjunction.objects:
        if item.covariance is not None:
            combined = combined + item.covariance.position
    return EncounterPlane(
        tca=conjunction.tca,
        miss_vector=basis @ r,
        covariance=symmetric_part(basis @ combined @ basis.T),
        relative_position=r,
        relative_velocity=v,
        combined_covariance=combined,
        basis=basis,
        unknown_covariance=conjunction.unknown_covariance,
    )
