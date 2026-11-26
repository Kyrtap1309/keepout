"""The 2-D Gaussian on the encounter plane, in its principal axes.

Both 2-D methods integrate the Gaussian over the hard-body circle. In the
principal axes the density separates: along the wide axis ``u`` it is a 1-D
Gaussian, and across the circle's chord at ``u`` the narrow-axis mass is a
difference of error functions (Chan 2008, ch. 4). With ``R`` the hard-body
radius and ``h = sqrt(R^2 - u^2)`` the half chord, that chord mass is

    M(h) = P(-h < v < h),   v ~ N(miss_v, sigma_v^2)

(:meth:`PlaneGaussian.chord_mass`), and ``Pc`` is the integral of
``N(u; miss_u, sigma_u^2) M(h)`` over ``-R < u < R``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, overload

import numpy as np
from scipy import special

from keepout._validation import require_positive
from keepout.errors import CovarianceError, GeometryError
from keepout.pc._errors import require_known_covariance

if TYPE_CHECKING:
    from keepout._types import FloatArray
    from keepout.encounter import EncounterPlane
    from keepout.pc.result import Method

SQRT_2: Final = math.sqrt(2.0)
SQRT_2PI: Final = math.sqrt(2.0 * math.pi)

SIGMA_FLOOR: Final = 1e-4
"""Smallest standard deviation kept on the encounter plane, as a fraction of
the hard-body radius. Smaller ones are raised to it, with a diagnostic, so a
degenerate (line-like) covariance still gives a finite, well-defined Pc."""


@dataclass(frozen=True, slots=True)
class PlaneGaussian:
    """Miss vector and covariance of the encounter plane in principal axes.

    Parameters
    ----------
    miss_u, miss_v
        Miss components along the wide axis ``u`` and the narrow axis ``v`` [m].
    sigma_u, sigma_v
        Standard deviations along ``u`` and ``v`` [m]; ``sigma_u >= sigma_v > 0``.
    hard_body_radius
        Hard-body radius [m].
    diagnostics
        Covariance floor messages, if any.
    """

    miss_u: float
    miss_v: float
    sigma_u: float
    sigma_v: float
    hard_body_radius: float
    diagnostics: tuple[str, ...] = ()

    @property
    def mahalanobis2(self) -> float:
        """Squared Mahalanobis distance of the miss vector, ``K^2``."""
        return (self.miss_u / self.sigma_u) ** 2 + (self.miss_v / self.sigma_v) ** 2

    def density_u(self, u: FloatArray) -> FloatArray:
        """1-D Gaussian density along ``u`` [1/m]."""
        z = (u - self.miss_u) / self.sigma_u
        density: FloatArray = np.exp(-0.5 * z * z) / (SQRT_2PI * self.sigma_u)
        return density

    def density_u_at(self, u: float) -> float:
        """:meth:`density_u` at one point (no NumPy overhead, for quadrature)."""
        z = (u - self.miss_u) / self.sigma_u
        return math.exp(-0.5 * z * z) / (SQRT_2PI * self.sigma_u)

    # P(-h < v < h) for v ~ N(miss_v, sigma_v^2). The mass is symmetric in the sign of
    # miss_v, so with miss_v >= 0 the lower limit is always <= 0; when the upper one
    # is too, erfc of the far tail keeps tiny masses precise.

    @overload
    def _limits(self, half_chord: float) -> tuple[float, float]: ...
    @overload
    def _limits(self, half_chord: FloatArray) -> tuple[FloatArray, FloatArray]: ...
    def _limits(
        self, half_chord: FloatArray | float
    ) -> tuple[FloatArray | float, FloatArray | float]:
        """Standardised upper and lower limits of ``v`` over ``+-half_chord``."""
        scale = SQRT_2 * self.sigma_v
        offset = abs(self.miss_v)
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
    miss_vector: FloatArray, covariance: FloatArray, hard_body_radius: float
) -> PlaneGaussian:
    """An encounter-plane miss vector and covariance in principal axes.

    Parameters
    ----------
    miss_vector
        Miss vector in the plane axes [m].
    covariance
        Combined 2x2 position covariance in the plane axes [m^2].
    hard_body_radius
        Hard-body radius [m].

    Raises
    ------
    ValueError
        If ``hard_body_radius`` is not finite and positive.
    GeometryError
        If the miss vector is not finite.
    CovarianceError
        If the covariance is not finite or has no positive eigenvalue.
    """
    require_positive(hard_body_radius, "hard_body_radius")
    if not np.all(np.isfinite(miss_vector)):
        raise GeometryError("encounter-plane miss vector must be finite")
    if not np.all(np.isfinite(covariance)):
        raise CovarianceError("encounter-plane covariance must be finite")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)  # ascending
    if not eigenvalues[1] > 0.0:
        raise CovarianceError(
            "encounter-plane covariance has no positive eigenvalue; Pc is undefined"
        )
    floor = (SIGMA_FLOOR * hard_body_radius) ** 2
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
        miss_u=float(miss[1]),
        miss_v=float(miss[0]),
        sigma_u=math.sqrt(float(eigenvalues[1])),
        sigma_v=math.sqrt(float(eigenvalues[0])),
        hard_body_radius=hard_body_radius,
        diagnostics=tuple(diagnostics),
    )


def plane_gaussian_of(
    plane: EncounterPlane, hard_body_radius: float, method: Method | None
) -> PlaneGaussian:
    """The :func:`plane_gaussian` of an encounter plane, for a 2-D method.

    Parameters
    ----------
    plane
        The encounter plane.
    hard_body_radius
        Hard-body radius [m].
    method
        The method that needs both covariances, checked to be known first
        (:func:`require_known_covariance`); ``None`` for a method that
        checks the unknown covariances itself.

    Raises
    ------
    MissingCovariance
        If ``method`` is given and a covariance is unknown.
    ValueError
        If ``hard_body_radius`` is not finite and positive.
    GeometryError
        If the miss vector is not finite.
    CovarianceError
        If the covariance is not finite or has no positive eigenvalue.
    """
    if method is not None:
        require_known_covariance(plane.unknown_covariance, method)
    return plane_gaussian(plane.miss_vector, plane.covariance, hard_body_radius)
