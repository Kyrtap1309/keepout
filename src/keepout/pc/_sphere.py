"""Quadrature over the hard-body sphere for a Gaussian density on it.

The collision rate integrates ``N(R x; r, A)`` times a smooth flux over the
unit sphere ``|x| = 1``. When the smallest standard deviation of ``A`` is much
smaller than ``R`` the density is concentrated in a thin band or small blobs,
which a fixed rule (e.g. Lebedev) under-resolves. The rule here puts the pole
on the narrowest axis ``e1``: the density then varies fast only in ``theta``,
near ``R cos(theta) = r . e1``, so Gauss-Legendre panels in ``theta`` cluster
there; in ``phi`` the density varies on the scale of the second standard
deviation, which sets the spacing of the periodic trapezoid rule (spectrally
accurate for smooth periodic integrands).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Final, NamedTuple

import numpy as np

from keepout._types import FloatArray, read_only

PANEL_ORDER: Final = 16
"""Gauss-Legendre nodes per theta panel."""

BAND_WIDTHS: Final = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
"""Panel breakpoints about the band, in smallest standard deviations."""

PHI_RESOLUTION: Final = 0.25
"""Trapezoid spacing in phi, as a fraction of (second sigma) / R."""

MIN_PHI, MAX_PHI = 64, 16384
"""Bounds on the number of phi nodes."""

_MIN_SIGMA: Final = 1e-12
"""Standard deviation [m] below which the phi spacing no longer shrinks (a
degenerate axis would otherwise ask for infinitely many nodes)."""


@lru_cache(maxsize=8)
def gauss_legendre(order: int) -> tuple[FloatArray, FloatArray]:
    """Read-only Gauss-Legendre nodes and weights on ``[-1, 1]``."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    read_only(nodes, weights)
    return nodes, weights


class SphereGrid(NamedTuple):
    """The product rule of :func:`sphere_grids` for one Gaussian, in the
    principal axes ``(e1, e2, e3)`` of its covariance.

    The nodes are ``cos(theta) e1 + sin(theta) (cos(phi) e2 + sin(phi) e3)``
    for every pair of a polar node ``theta`` and an azimuth node ``phi``, with
    weight ``theta_weight * phi_weight``.

    Parameters
    ----------
    cos_theta, sin_theta
        The polar nodes ``(t,)``, from ``e1``.
    theta_weight
        Their weights ``(t,)``, including the ``sin(theta)`` of the solid angle.
    cos_phi, sin_phi
        The azimuth nodes ``(p,)``, from ``e2`` towards ``e3`` (read-only).
    phi_weight
        The weight of every azimuth node, ``2 pi / p``.
    """

    cos_theta: FloatArray
    sin_theta: FloatArray
    theta_weight: FloatArray
    cos_phi: FloatArray
    sin_phi: FloatArray
    phi_weight: float

    def cartesian(self, axes: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Unit vectors ``(k, 3)`` and weights ``(k,)`` of the nodes, for the
        principal axes ``axes`` (columns)."""
        count = self.cos_phi.size
        ring_cos = np.multiply.outer(self.sin_theta, self.cos_phi)
        ring_sin = np.multiply.outer(self.sin_theta, self.sin_phi)
        nodes = np.empty((self.cos_theta.size, count, 3))
        # Component by component, in the order of the sum
        # cos(theta) e1 + sin(theta) cos(phi) e2 + sin(theta) sin(phi) e3.
        for j in range(3):
            component = nodes[..., j]
            np.multiply(self.cos_theta[:, None], axes[j, 0], out=component)
            component += ring_cos * axes[j, 1]
            component += ring_sin * axes[j, 2]
        weights = np.repeat(self.theta_weight, count) * self.phi_weight
        return nodes.reshape(-1, 3), weights


@lru_cache(maxsize=64)
def _azimuths(count: int) -> tuple[FloatArray, FloatArray]:
    """Read-only cosines and sines of ``count`` periodic trapezoid nodes."""
    phi = (np.arange(count) + 0.5) * (2.0 * math.pi / count)
    cosine, sine = np.cos(phi), np.sin(phi)
    read_only(cosine, sine)
    return cosine, sine


def sphere_grids(
    values: FloatArray,
    along: FloatArray,
    radius: float,
    *,
    order: int = PANEL_ORDER,
    phi_resolution: float = PHI_RESOLUTION,
) -> list[SphereGrid]:
    """Product rules on the unit sphere for ``m`` Gaussians ``N(radius * x;
    mean, A)``, one per Gaussian.

    Batched over the Gaussians; only the panels, whose number differs
    between them, are split at the end.

    Parameters
    ----------
    values
        Eigenvalues ``(m, 3)`` of the 3x3 position covariances [m^2],
        ascending.
    along
        Components ``(m,)`` of the means along their narrowest axes [m].
    radius
        Sphere radius [m].
    order
        Gauss-Legendre nodes per theta panel.
    phi_resolution
        Trapezoid spacing in phi, as a fraction of (second sigma) / radius.
    """
    sigmas = np.sqrt(np.clip(values, 0.0, None))
    offsets = np.array([sign * width for width in BAND_WIDTHS for sign in (-1.0, 1.0)])
    cosine = (along[:, None] + offsets * sigmas[:, :1]) / radius
    poles = np.zeros((along.size, 1)), np.full((along.size, 1), math.pi)
    breaks = np.sort(
        np.concatenate([*poles, np.arccos(np.clip(cosine, -1.0, 1.0))], axis=1),
        axis=1,
    )
    lower, upper = breaks[:, :-1], breaks[:, 1:]
    # Repeated breakpoints (clipped to a pole) give empty panels, left out.
    keep = upper > lower
    x, w = gauss_legendre(order)
    half = 0.5 * (upper - lower)[..., None]
    theta = half * x + 0.5 * (lower + upper)[..., None]
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    theta_weight = half * w * sin_theta
    spacing = phi_resolution * np.maximum(sigmas[:, 1], _MIN_SIGMA) / radius
    counts = np.nan_to_num(
        np.clip(np.ceil(2.0 * math.pi / spacing), MIN_PHI, MAX_PHI), nan=MIN_PHI
    ).astype(np.intp)
    grids: list[SphereGrid] = []
    for k, count in enumerate(counts.tolist()):
        panels = keep[k]
        cos_phi, sin_phi = _azimuths(count)
        grids.append(
            SphereGrid(
                cos_theta[k, panels].reshape(-1),
                sin_theta[k, panels].reshape(-1),
                theta_weight[k, panels].reshape(-1),
                cos_phi,
                sin_phi,
                2.0 * math.pi / count,
            )
        )
    return grids
