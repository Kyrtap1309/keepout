"""The 2-D Gaussian on the encounter plane, in its principal axes.

Both 2-D methods integrate the Gaussian over the hard-body circle. In the
principal axes the density separates: along the wide axis ``u`` it is a 1-D
Gaussian, and across the circle's chord at ``u`` the narrow-axis mass is a
difference of error functions (Chan 2008, ch. 4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import special

from keepout._types import FloatArray, require_positive
from keepout.errors import CovarianceError, GeometryError

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)

SIGMA_FLOOR = 1e-4
"""Smallest standard deviation kept on the encounter plane, as a fraction of
the hard-body radius. Smaller ones are raised to it, with a diagnostic, so a
degenerate (line-like) covariance still gives a finite, well-defined Pc."""


@dataclass(frozen=True)
class PlaneGaussian:
    """Miss vector and covariance of the encounter plane in principal axes.

    Parameters
    ----------
    mu, mv
        Miss components along the wide axis ``u`` and the narrow axis ``v`` [m].
    su, sv
        Standard deviations along ``u`` and ``v`` [m]; ``su >= sv > 0``.
    radius
        Hard-body radius [m].
    diagnostics
        Covariance floor messages, if any.
    """

    mu: float
    mv: float
    su: float
    sv: float
    radius: float
    diagnostics: tuple[str, ...] = ()

    def density_u(self, u: FloatArray) -> FloatArray:
        """1-D Gaussian density along ``u`` [1/m]."""
        z = (u - self.mu) / self.su
        density: FloatArray = np.exp(-0.5 * z * z) / (_SQRT_2PI * self.su)
        return density

    def density_u_at(self, u: float) -> float:
        """:meth:`density_u` at one point (no NumPy overhead, for quadrature)."""
        z = (u - self.mu) / self.su
        return math.exp(-0.5 * z * z) / (_SQRT_2PI * self.su)

    # P(-h < v < h) for v ~ N(mv, sv^2). The mass is symmetric in the sign of
    # mv, so with mv >= 0 the lower limit is always <= 0; when the upper one
    # is too, erfc of the far tail keeps tiny masses precise.

    def _limits(self, half_chord: FloatArray | float) -> tuple[Any, Any]:
        scale = _SQRT_2 * self.sv
        offset = abs(self.mv)
        return (half_chord - offset) / scale, (-half_chord - offset) / scale

    def chord_mass(self, half_chord: FloatArray) -> FloatArray:
        """Probability that ``v`` lies within ``+-half_chord`` of the centre."""
        upper, lower = self._limits(half_chord)
        tail = upper <= 0.0
        mass: FloatArray = np.empty_like(upper)
        mass[tail] = 0.5 * (special.erfc(-upper[tail]) - special.erfc(-lower[tail]))
        core = ~tail
        mass[core] = 0.5 * (special.erf(upper[core]) - special.erf(lower[core]))
        return mass

    def chord_mass_at(self, half_chord: float) -> float:
        """:meth:`chord_mass` at one point (no NumPy overhead, for quadrature)."""
        upper, lower = self._limits(half_chord)
        if upper <= 0.0:
            return 0.5 * (math.erfc(-upper) - math.erfc(-lower))
        return 0.5 * (math.erf(upper) - math.erf(lower))


def plane_gaussian(
    miss_vector: FloatArray, covariance: FloatArray, radius: float
) -> PlaneGaussian:
    """Rotate an encounter-plane miss vector and covariance to principal axes.

    Parameters
    ----------
    miss_vector
        Miss vector in the plane axes [m].
    covariance
        Combined 2x2 position covariance in the plane axes [m^2].
    radius
        Hard-body radius [m].

    Raises
    ------
    ValueError
        If ``radius`` is not finite and positive.
    GeometryError
        If the miss vector is not finite.
    CovarianceError
        If the covariance is not finite or has no positive eigenvalue.
    """
    require_positive(radius, "hard_body_radius")
    if not np.all(np.isfinite(miss_vector)):
        raise GeometryError("encounter-plane miss vector must be finite")
    if not np.all(np.isfinite(covariance)):
        raise CovarianceError("encounter-plane covariance must be finite")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)  # ascending
    if not eigenvalues[1] > 0.0:
        raise CovarianceError(
            "encounter-plane covariance has no positive eigenvalue; Pc is undefined"
        )
    floor = (SIGMA_FLOOR * radius) ** 2
    diagnostics: list[str] = []
    for index, name in ((0, "narrow"), (1, "wide")):
        if eigenvalues[index] < floor:
            diagnostics.append(
                f"encounter-plane covariance: {name}-axis variance "
                f"{eigenvalues[index]:.3g} m^2 raised to {floor:.3g} m^2"
            )
    eigenvalues = np.maximum(eigenvalues, floor)
    miss = eigenvectors.T @ miss_vector
    return PlaneGaussian(
        mu=float(miss[1]),
        mv=float(miss[0]),
        su=math.sqrt(float(eigenvalues[1])),
        sv=math.sqrt(float(eigenvalues[0])),
        radius=radius,
        diagnostics=tuple(diagnostics),
    )
