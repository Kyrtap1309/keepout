"""Two-body orbit quantities of an Earth-orbiting state."""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from keepout.state import StateVector

__all__ = ["EARTH_MU", "orbital_period"]

EARTH_MU: Final = 3.986004418e14
"""Earth's gravitational parameter [m^3/s^2] (WGS 84)."""


def orbital_period(state: StateVector) -> float | None:
    """Two-body orbital period [s] of an inertial state; ``None`` if unbound."""
    r = float(np.linalg.norm(state.r))
    v = float(np.linalg.norm(state.v))
    energy = 0.5 * v * v - EARTH_MU / r
    if energy >= 0.0:
        return None
    semi_major_axis = -EARTH_MU / (2.0 * energy)
    return 2.0 * math.pi * math.sqrt(semi_major_axis**3 / EARTH_MU)
