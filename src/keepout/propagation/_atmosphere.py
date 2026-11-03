"""Exponential atmosphere: a density from an altitude alone.

The table of Vallado, *Fundamentals of Astrodynamics and Applications*, table
8-4: in each altitude band ``rho = rho0 * exp(-(h - h0) / H)``. It ignores
solar activity and the day-night bulge, and it is not quite continuous: a band
may end up to 0.14 % away from the next band's base density (at 25 km, and
0.01 % or less above that).
"""

from __future__ import annotations

from typing import Final

import numpy as np

from keepout._types import FloatArray, read_only
from keepout.units import KM

__all__ = ["density"]

_TABLE: Final[FloatArray] = np.array(
    [
        (0.0, 1.225, 7.249),
        (25.0, 3.899e-2, 6.349),
        (30.0, 1.774e-2, 6.682),
        (40.0, 3.972e-3, 7.554),
        (50.0, 1.057e-3, 8.382),
        (60.0, 3.206e-4, 7.714),
        (70.0, 8.770e-5, 6.549),
        (80.0, 1.905e-5, 5.799),
        (90.0, 3.396e-6, 5.382),
        (100.0, 5.297e-7, 5.877),
        (110.0, 9.661e-8, 7.263),
        (120.0, 2.438e-8, 9.473),
        (130.0, 8.484e-9, 12.636),
        (140.0, 3.845e-9, 16.149),
        (150.0, 2.070e-9, 22.523),
        (180.0, 5.464e-10, 29.740),
        (200.0, 2.789e-10, 37.105),
        (250.0, 7.248e-11, 45.546),
        (300.0, 2.418e-11, 53.628),
        (350.0, 9.518e-12, 53.298),
        (400.0, 3.725e-12, 58.515),
        (450.0, 1.585e-12, 60.828),
        (500.0, 6.967e-13, 63.822),
        (600.0, 1.454e-13, 71.835),
        (700.0, 3.614e-14, 88.667),
        (800.0, 1.170e-14, 124.64),
        (900.0, 5.245e-15, 181.05),
        (1000.0, 3.019e-15, 268.00),
    ],
    dtype=np.float64,
)
"""Base altitude [km], base density [kg/m³] and scale height [km] per band."""

_BASE_ALTITUDE: Final[FloatArray] = _TABLE[:, 0] * KM
_BASE_DENSITY: Final[FloatArray] = _TABLE[:, 1]
_SCALE_HEIGHT: Final[FloatArray] = _TABLE[:, 2] * KM
read_only(_BASE_ALTITUDE, _BASE_DENSITY, _SCALE_HEIGHT)


def density(altitude: FloatArray) -> FloatArray:
    """Density of the air at altitudes above the Earth.

    Parameters
    ----------
    altitude
        Altitudes [m], of any shape.

    Returns
    -------
    FloatArray
        Density [kg/m³], in the shape of ``altitude``. Above 1000 km the last
        row of the table goes on; at or below 0 m it is the sea-level density,
        so a state inside the Earth stays finite.
    """
    height = np.maximum(altitude, 0.0)
    band = np.searchsorted(_BASE_ALTITUDE, height, side="right") - 1
    decay = (height - _BASE_ALTITUDE[band]) / _SCALE_HEIGHT[band]
    result: FloatArray = _BASE_DENSITY[band] * np.exp(-decay)
    return result
