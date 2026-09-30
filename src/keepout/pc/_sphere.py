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
from itertools import pairwise

import numpy as np

from keepout._types import FloatArray

PANEL_ORDER = 16
"""Gauss-Legendre nodes per theta panel."""

BAND_WIDTHS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
"""Panel breakpoints about the band, in smallest standard deviations."""

PHI_RESOLUTION = 0.25
"""Trapezoid spacing in phi, as a fraction of (second sigma) / R."""

MIN_PHI, MAX_PHI = 64, 16384
"""Bounds on the number of phi nodes."""


@lru_cache(maxsize=4)
def _gauss_legendre(order: int) -> tuple[FloatArray, FloatArray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes.flags.writeable = weights.flags.writeable = False
    return nodes, weights


def sphere_rule(
    covariance: FloatArray,
    mean: FloatArray,
    radius: float,
    *,
    order: int = PANEL_ORDER,
    phi_resolution: float = PHI_RESOLUTION,
) -> tuple[FloatArray, FloatArray]:
    """Nodes and weights on the unit sphere for ``N(radius * x; mean, covariance)``.

    Parameters
    ----------
    covariance
        3x3 position covariance [m^2], positive definite.
    mean
        Mean relative position [m].
    radius
        Sphere radius [m].
    order
        Gauss-Legendre nodes per theta panel.
    phi_resolution
        Trapezoid spacing in phi, as a fraction of (second sigma) / radius.

    Returns
    -------
    tuple
        Unit vectors ``(k, 3)`` and weights ``(k,)`` (solid angle, summing to
        ``4 pi``).
    """
    values, axes = np.linalg.eigh(covariance)
    sigmas = np.sqrt(np.clip(values, 0.0, None))
    along = float(axes[:, 0] @ mean)
    breaks = {0.0, math.pi}
    for width in BAND_WIDTHS:
        for sign in (-1.0, 1.0):
            cosine = (along + sign * width * sigmas[0]) / radius
            breaks.add(math.acos(min(1.0, max(-1.0, cosine))))
    panels = list(pairwise(sorted(breaks)))
    x, w = _gauss_legendre(order)
    theta = np.concatenate([0.5 * (b - a) * x + 0.5 * (a + b) for a, b in panels])
    theta_weight = np.concatenate([0.5 * (b - a) * w for a, b in panels]) * np.sin(
        theta
    )
    spacing = phi_resolution * max(float(sigmas[1]), 1e-12) / radius
    count = int(min(MAX_PHI, max(MIN_PHI, math.ceil(2.0 * math.pi / spacing))))
    phi = (np.arange(count) + 0.5) * (2.0 * math.pi / count)
    sin_theta = np.sin(theta)[:, None]
    nodes = (
        np.cos(theta)[:, None, None] * axes[:, 0]
        + (sin_theta * np.cos(phi))[..., None] * axes[:, 1]
        + (sin_theta * np.sin(phi))[..., None] * axes[:, 2]
    ).reshape(-1, 3)
    weights = np.repeat(theta_weight, count) * (2.0 * math.pi / count)
    return nodes, weights
