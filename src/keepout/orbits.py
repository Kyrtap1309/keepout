"""Two-body orbit quantities of an Earth-orbiting state."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from keepout._types import InexactArray
    from keepout.state import StateVector

__all__ = [
    "EARTH_MU",
    "eccentricity_vector",
    "orbital_period",
    "perigee_radius",
]

EARTH_MU: Final = 3.986004418e14
"""Earth's gravitational parameter [m^3/s^2] (WGS 84)."""


def _semi_major_axis(state: StateVector) -> float | None:
    """Two-body semi-major axis [m] of an inertial state; ``None`` if unbound."""
    energy = float(state.v @ state.v) / 2.0 - EARTH_MU / float(np.linalg.norm(state.r))
    if energy >= 0.0:
        return None
    return -EARTH_MU / (2.0 * energy)


def eccentricity_vector(
    r: InexactArray, v: InexactArray, mu: float = EARTH_MU
) -> InexactArray:
    """Eccentricity vectors ``((v^2 - mu/|r|) r - (r . v) v) / mu`` of states.

    This form keeps its precision near a circle. Complex input carries a
    complex-step derivative.

    Parameters
    ----------
    r
        Positions [m], shape ``(..., 3)``.
    v
        Velocities [m/s], same shape.
    mu
        Gravitational parameter [m^3/s^2].

    Returns
    -------
    numpy.ndarray
        The eccentricity vectors, shape ``(..., 3)``.
    """
    radius = np.sqrt(np.sum(r * r, -1))
    speed2 = np.sum(v * v, -1)
    vector: InexactArray = (
        (speed2 - mu / radius)[..., None] * r - np.sum(r * v, -1)[..., None] * v
    ) / mu
    return vector


def orbital_period(state: StateVector) -> float | None:
    """Two-body orbital period [s] of an inertial state; ``None`` if unbound.

    Parameters
    ----------
    state
        Position and velocity in an inertial frame (e.g. GCRF).

    Returns
    -------
    float or None
        The period [s], or ``None`` for an unbound orbit.
    """
    semi_major_axis = _semi_major_axis(state)
    if semi_major_axis is None:
        return None
    return 2.0 * math.pi * math.sqrt(semi_major_axis**3 / EARTH_MU)


def perigee_radius(state: StateVector) -> float | None:
    """Two-body perigee radius [m] of an inertial state; ``None`` if unbound.

    Parameters
    ----------
    state
        Position and velocity in an inertial frame (e.g. GCRF).

    Returns
    -------
    float or None
        The perigee radius [m], or ``None`` for an unbound orbit.
    """
    semi_major_axis = _semi_major_axis(state)
    if semi_major_axis is None:
        return None
    eccentricity = float(np.linalg.norm(eccentricity_vector(state.r, state.v)))
    return semi_major_axis * (1.0 - eccentricity)
