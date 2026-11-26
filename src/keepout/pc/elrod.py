"""2-D Pc by Gauss-Chebyshev quadrature of the error-function integral.

The same 1-D integral as :mod:`keepout.pc.foster`, taken over the whole chord
range ``u = R t`` (``R`` the hard-body radius), ``t`` in ``[-1, 1]``, and
written as

    Pc = R^2 * integral of [N(Rt) * M(h) / h] * sqrt(1 - t^2) dt,   h = R sqrt(1 - t^2)

which Gauss-Chebyshev quadrature of the second kind integrates with fixed
nodes (Elrod 2019, as used by NASA CARA). ``M(h) / h`` stays finite at the
edge because ``M`` vanishes linearly there.

The ``n`` nodes are ``t_k = cos(theta_k)``, ``theta_k = k pi / (n + 1)``, with
weights ``pi / (n + 1) sin^2(theta_k)``. With ``h_k = R sin(theta_k)`` the rule
reads

    Pc ~ R pi / (n + 1) * sum of N(R cos theta_k) M(R sin theta_k) sin theta_k

so no division by ``h`` is needed. The nodes of order ``n`` are the even ones of
order ``2n + 1``: by default the order adapts along that nested sequence,
starting high enough to resolve the Gaussian along ``u``, and each step only
evaluates the new nodes, until two orders agree to :data:`ELROD_RTOL`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import numpy as np

from keepout._validation import require_count
from keepout.pc._plane import PlaneGaussian, plane_gaussian_of
from keepout.pc.result import PcResult

if TYPE_CHECKING:
    from keepout._types import FloatArray
    from keepout.encounter import EncounterPlane

__all__ = ["ELROD_RTOL", "elrod"]

_START_ORDER: Final = 63
"""Smallest order tried in adaptive mode: ``2^6 - 1``, so the nested orders
``n -> 2n + 1`` stay of the form ``2^j - 1``."""

_MAX_ORDER: Final = 65535
"""Largest order tried in adaptive mode (``2^16 - 1``)."""

ELROD_RTOL: Final = 1e-10
"""Relative agreement of two successive orders that ends adaptive mode."""


def _terms(gaussian: PlaneGaussian, k: FloatArray, order: int) -> FloatArray:
    """``N(R cos theta) M(R sin theta) sin theta`` at ``theta = k pi / (order + 1)``."""
    theta = k * (math.pi / (order + 1))
    radius = gaussian.hard_body_radius
    sin_theta = np.sin(theta)
    terms: FloatArray = (
        gaussian.density_u(radius * np.cos(theta))
        * gaussian.chord_mass(radius * sin_theta)
        * sin_theta
    )
    return terms


def _all_terms(gaussian: PlaneGaussian, order: int) -> FloatArray:
    """Terms at every node of the rule of order ``order``."""
    return _terms(gaussian, np.arange(1.0, order + 1), order)


def _rule_value(gaussian: PlaneGaussian, terms: FloatArray) -> float:
    """The rule of order ``terms.size`` from its terms (see the module docstring)."""
    return gaussian.hard_body_radius * math.pi / (terms.size + 1) * float(np.sum(terms))


def _refined(gaussian: PlaneGaussian, terms: FloatArray) -> FloatArray:
    """Terms of order ``2n + 1`` from those of order ``n``: only the odd nodes
    are new."""
    order = 2 * terms.size + 1
    refined = np.empty(order)
    refined[1::2] = terms
    refined[0::2] = _terms(gaussian, np.arange(1.0, order + 1, 2.0), order)
    return refined


def _resolving_order(gaussian: PlaneGaussian) -> int:
    """Order whose node spacing near the centre (~ pi R / (n + 1)) is below
    sigma_u / 2, of the form ``2^j - 1``."""
    needed = 2.0 * math.pi * gaussian.hard_body_radius / gaussian.sigma_u
    if needed <= _START_ORDER + 1:
        return _START_ORDER
    return (1 << math.ceil(math.log2(needed))) - 1


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
        Number of quadrature nodes. ``None`` (default) adapts the order
        along ``n -> 2n + 1`` from 63 (or the order that resolves the
        covariance) up to 65535; an integer fixes it (CARA uses 64).

    Returns
    -------
    PcResult
        ``details["order"]`` is the order of the returned value.

    Raises
    ------
    ValueError
        If ``hard_body_radius`` is not finite and positive, or ``order`` is
        not an integer of at least 1.
    GeometryError
        If the miss vector is not finite.
    CovarianceError
        If a covariance is unknown, or the plane covariance is not finite or
        has no positive eigenvalue.
    """
    if order is not None:
        require_count(order, 1, "order")
    gaussian = plane_gaussian_of(plane, hard_body_radius, "elrod")
    diagnostics = list(gaussian.diagnostics)
    resolving = _resolving_order(gaussian)
    if order is not None:
        if order < resolving:
            diagnostics.append(
                f"Elrod order {order} may be too low: about {resolving} nodes "
                "are needed to resolve the covariance; use order=None"
            )
        value = _rule_value(gaussian, _all_terms(gaussian, order))
    else:
        # Start below _MAX_ORDER so that at least two orders are compared.
        order = min(resolving, _MAX_ORDER // 2)
        terms = _all_terms(gaussian, order)
        value = _rule_value(gaussian, terms)
        while order < _MAX_ORDER:
            terms = _refined(gaussian, terms)
            order = terms.size
            previous, value = value, _rule_value(gaussian, terms)
            if abs(value - previous) <= ELROD_RTOL * abs(value):
                break
        else:
            diagnostics.append(
                f"Elrod quadrature did not converge by order {order}; "
                "use method='foster'"
            )
    return PcResult.clamped(value, "elrod", tuple(diagnostics), {"order": float(order)})
