"""The 2-D Gaussian on the encounter plane, in its principal axes.

Both 2-D methods integrate the Gaussian over the hard-body circle. In the
principal axes the density separates: along the wide axis ``u`` it is a 1-D
Gaussian, and across the circle's chord at ``u`` the narrow-axis mass is a
difference of error functions (Chan 2008, ch. 4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import special

from keepout._types import FloatArray
from keepout.errors import CovarianceError

SIGMA_FLOOR = 1e-4
"""Smallest standard deviation kept on the encounter plane, as a fraction of
the hard-body radius. Smaller ones are raised to it, with a diagnostic, so a
degenerate (line-like) covariance still gives a finite, well-defined Pc."""

TAIL_SIGMAS = 12.0
"""Half-width, in standard deviations, of the band along the wide axis that
is integrated; outside it the density is below exp(-72) of its peak."""


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

    def band(self) -> tuple[float, float] | None:
        """Part of ``[-radius, radius]`` along ``u`` where the density matters.

        Returns ``None`` if the band misses the circle (Pc underflows to 0).
        """
        low = max(self.mu - TAIL_SIGMAS * self.su, -self.radius)
        high = min(self.mu + TAIL_SIGMAS * self.su, self.radius)
        return (low, high) if low < high else None

    def density_u(self, u: FloatArray) -> FloatArray:
        """1-D Gaussian density along ``u`` [1/m]."""
        z = (u - self.mu) / self.su
        density: FloatArray = np.exp(-0.5 * z * z) / (
            math.sqrt(2.0 * math.pi) * self.su
        )
        return density

    def chord_mass(self, half_chord: FloatArray) -> FloatArray:
        """Probability that ``v`` lies within ``+-half_chord`` of the centre.

        ``P(-h < v < h)`` for ``v ~ N(mv, sv^2)``, computed with ``erfc`` on
        the far side of the mean so that tiny masses keep their precision.
        """
        scale = math.sqrt(2.0) * self.sv
        upper = (half_chord - self.mv) / scale
        lower = (-half_chord - self.mv) / scale
        # erf(upper) - erf(lower), without cancellation in either tail.
        mass: FloatArray = np.where(
            lower >= 0.0,
            0.5 * (special.erfc(lower) - special.erfc(upper)),
            np.where(
                upper <= 0.0,
                0.5 * (special.erfc(-upper) - special.erfc(-lower)),
                0.5 * (special.erf(upper) - special.erf(lower)),
            ),
        )
        return mass


def require_positive(value: float, name: str) -> None:
    """Raise ``ValueError`` unless ``value`` is finite and positive."""
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be finite and > 0, got {value}")


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
    CovarianceError
        If the covariance has no positive eigenvalue.
    """
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
