"""Two-body orbit quantities of an Earth-orbiting state."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from keepout.state import StateVector

__all__ = ["EARTH_MU", "orbital_period", "perigee_radius"]

EARTH_MU: Final = 3.986004418e14
"""Earth's gravitational parameter [m^3/s^2] (WGS 84)."""


def _semi_major_axis(state: StateVector) -> float | None:
    """Two-body semi-major axis [m] of an inertial state; ``None`` if unbound."""
    energy = float(state.v @ state.v) / 2.0 - EARTH_MU / float(np.linalg.norm(state.r))
    if energy >= 0.0:
        return None
    return -EARTH_MU / (2.0 * energy)


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
    r, v = state.r, state.v
    # The eccentricity vector, which keeps its precision near a circle.
    radius = float(np.linalg.norm(r))
    eccentricity_vector = ((v @ v - EARTH_MU / radius) * r - (r @ v) * v) / EARTH_MU
    return semi_major_axis * (1.0 - float(np.linalg.norm(eccentricity_vector)))
