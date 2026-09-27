"""2-D Pc by Gauss-Chebyshev quadrature of the error-function integral.

The same 1-D integral as :mod:`keepout.pc.foster`, taken over the whole chord
range ``u = R t``, ``t`` in ``[-1, 1]``, and written as

    Pc = R^2 * integral of [N(Rt) * M(h) / h] * sqrt(1 - t^2) dt,   h = R sqrt(1 - t^2)

which Gauss-Chebyshev quadrature of the second kind integrates with fixed
nodes (Elrod 2019, as used by NASA CARA). ``M(h) / h`` stays finite at the
edge because ``M`` vanishes linearly there.

By default the order adapts: it starts high enough to resolve the Gaussian
along ``u`` and doubles until two orders agree to :data:`ELROD_RTOL`.
"""

from __future__ import annotations

import math

import numpy as np

from keepout._types import FloatArray
from keepout.encounter import EncounterPlane
from keepout.pc._plane import PlaneGaussian, plane_gaussian, require_positive
from keepout.pc.result import PcResult

START_ORDER = 64
"""Smallest order tried in adaptive mode (the CARA default)."""

MAX_ORDER = 65536
"""Largest order tried in adaptive mode."""

ELROD_RTOL = 1e-10
"""Relative agreement of two successive orders that ends adaptive mode."""


def _nodes(order: int) -> tuple[FloatArray, FloatArray]:
    """Gauss-Chebyshev (second kind) nodes and weights on ``[-1, 1]``."""
    angles = np.arange(1, order + 1) * (math.pi / (order + 1))
    return np.cos(angles), (math.pi / (order + 1)) * np.sin(angles) ** 2


def _sum(gaussian: PlaneGaussian, order: int) -> float:
    t, weights = _nodes(order)
    radius = gaussian.radius
    half_chord = radius * np.sqrt(1.0 - t * t)
    terms = (
        gaussian.density_u(radius * t) * gaussian.chord_mass(half_chord) / half_chord
    )
    return radius * radius * float(np.sum(weights * terms))


def _resolving_order(gaussian: PlaneGaussian) -> int:
    """Order whose node spacing near the centre (~ pi R / n) is below su / 2."""
    needed = 2.0 * math.pi * gaussian.radius / gaussian.su
    if needed <= START_ORDER:
        return START_ORDER
    return 1 << math.ceil(math.log2(needed))


def elrod(
    plane: EncounterPlane, hard_body_radius: float, *, order: int | None = None
) -> PcResult:
    """2-D Pc of a short-duration encounter by Gauss-Chebyshev quadrature.

    Parameters
    ----------
    plane
        The encounter plane (miss vector and combined covariance).
    hard_body_radius
        Combined hard-body radius [m].
    order
        Number of quadrature nodes. ``None`` (default) adapts the order;
        an integer fixes it (CARA uses 64).

    Returns
    -------
    PcResult
        ``details["order"]`` is the order of the returned value.

    Raises
    ------
    ValueError
        If ``hard_body_radius`` is not finite and positive, or ``order`` < 1.
    CovarianceError
        If the plane covariance has no positive eigenvalue.
    """
    require_positive(hard_body_radius, "hard_body_radius")
    if order is not None and order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    gaussian = plane_gaussian(plane.miss_vector, plane.covariance, hard_body_radius)
    diagnostics = list(gaussian.diagnostics)
    resolving = _resolving_order(gaussian)
    if order is not None:
        if order < resolving:
            diagnostics.append(
                f"Elrod order {order} may be too low: about {resolving} nodes "
                "are needed to resolve the covariance; use order=None"
            )
        value = _sum(gaussian, order)
    else:
        order = min(resolving, MAX_ORDER)
        value = _sum(gaussian, order)
        converged = False
        while order < MAX_ORDER:
            order *= 2
            previous, value = value, _sum(gaussian, order)
            if abs(value - previous) <= ELROD_RTOL * abs(value):
                converged = True
                break
        if not converged:
            diagnostics.append(
                f"Elrod quadrature did not converge by order {order}; "
                "use method='foster'"
            )
    return PcResult(
        min(max(value, 0.0), 1.0), "elrod", tuple(diagnostics), {"order": float(order)}
    )
