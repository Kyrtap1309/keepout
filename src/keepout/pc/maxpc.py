"""Maximum Pc: upper bounds when a covariance is missing or not trusted.

* :func:`frisbee` — one covariance unknown (Frisbee 2015): the maximum over
  a variance ``s >= 0`` added along the miss direction ``u = m / |m|`` to the
  known covariance ``C_k``. For a small hard body the Gaussian density at the
  miss vector under ``C_k + s u u^T`` is largest at
  ``s = |m|^2 (K^2 - 1) / K^2`` with ``K^2 = m^T C_k^-1 m``
  (Sherman-Morrison); that optimum seeds a bounded search, since it is exact
  only when the hard body is small against the miss. ``s = 0`` is kept
  unless the analytic ``s`` beats its Pc by more than
  :data:`SEARCH_GAIN_RTOL`, and that point is kept unless the search's
  maximum beats it by the same margin.
* :func:`dilution` — both covariances known but possibly too large: the
  maximum over a scale factor ``k`` of the combined covariance (Alfano 2005;
  Hejduk et al. 2019). The small-body optimum ``k = K^2 / 2`` seeds a bounded search.
* :func:`alfano` — no covariance at all: the maximum over every 2-D
  covariance for the miss distance ``d`` alone, reached by a line-like
  covariance along the miss vector with ``sigma^2 = 2 d R / ln((d+R)/(d-R))``
  (Alfano 2005).

All three are upper bounds, not estimates: their results have
``kind="upper_bound"`` and say so in their diagnostics.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Final

import numpy as np
from scipy import optimize

from keepout._types import FloatArray
from keepout._validation import require_positive
from keepout.encounter import EncounterPlane
from keepout.pc._plane import (
    plane_gaussian,
    plane_gaussian_of,
)
from keepout.pc.foster import gaussian_pc
from keepout.pc.result import PcResult

__all__ = [
    "DILUTION_TOLERANCE",
    "SEARCH_DECADES",
    "SEARCH_GAIN_RTOL",
    "SEARCH_XATOL",
    "SEED_SCALE_FLOOR",
    "alfano",
    "dilution",
    "frisbee",
]

SEARCH_DECADES: Final = 4.0
"""Half-width, in decades, of the bounded searches about the small-body
optima (the added variance of :func:`frisbee`, the scale factor of
:func:`dilution`)."""

SEARCH_GAIN_RTOL: Final = 1e-8
"""Relative Pc gain a searched point needs to replace the seed point. The
optima are flat and Foster's quadrature is accurate to about
:data:`~keepout.pc.foster.FOSTER_RTOL` (1e-10), so a smaller gain is noise;
keeping the seed then keeps its exact added variance or scale factor."""

SEARCH_XATOL: Final = 1e-9
"""Convergence of the bounded searches, in decades of the searched variable."""

SEED_SCALE_FLOOR: Final = 1e-300
"""Smallest dilution seed scale: keeps its logarithm finite for a zero miss."""

DILUTION_TOLERANCE: Final = 1e-3
"""Relative excess of the maximum over the nominal Pc that marks a
conjunction as diluted (as in CARA's ``PcDilution``)."""

_INSIDE: Final = (
    "miss distance {miss:.3g} m is inside the hard body ({radius:.3g} m): "
    "Pc tends to 1 as the covariance shrinks"
)

_EDGE: Final = (
    "miss distance {miss:.3g} m is on the hard-body edge: "
    "Pc tends to 1/2 as the covariance shrinks"
)


def _pc(miss_vector: FloatArray, covariance: FloatArray, radius: float) -> float:
    """Foster's Pc of a miss vector under a trial covariance."""
    return gaussian_pc(plane_gaussian(miss_vector, covariance, radius)).value


def _search_log_max(pc: Callable[[float], float], centre: float) -> tuple[float, float]:
    """The maximum of ``pc(10**x)`` over ``x`` in ``centre +- SEARCH_DECADES``.

    Returns the maximising argument ``10**x`` and the maximum.
    """
    search = optimize.minimize_scalar(
        lambda x: -pc(10.0**x),
        bounds=(centre - SEARCH_DECADES, centre + SEARCH_DECADES),
        method="bounded",
        options={"xatol": SEARCH_XATOL},
    )
    return 10.0 ** float(search.x), -float(search.fun)


def _prefer(
    seed: tuple[float, float], other: tuple[float, float]
) -> tuple[float, float]:
    """``seed`` (argument, Pc), or ``other`` if it beats that Pc by more than
    :data:`SEARCH_GAIN_RTOL`."""
    return other if other[1] > seed[1] * (1.0 + SEARCH_GAIN_RTOL) else seed


def frisbee(plane: EncounterPlane, hard_body_radius: float) -> PcResult:
    """Maximum Pc over the unknown covariance of one object (Frisbee 2015).

    The unknown covariance is taken as a variance ``s`` along the miss
    direction. Frisbee's analytic ``s`` maximises the density at the miss
    vector, which is the Pc optimum only for a small hard body; it seeds a
    bounded search over ``log s``. The seed is chosen in two steps with the
    same margin :data:`SEARCH_GAIN_RTOL`: ``s = 0`` is kept unless the
    analytic ``s`` beats its Pc by more than that margin, and the point kept
    is replaced by the search's maximum only if that beats it by the same
    margin. A small-body result therefore reports Frisbee's ``s`` exactly.

    Parameters
    ----------
    plane
        Encounter plane whose ``covariance`` is the known object's projection
        and ``unknown_covariance`` names exactly one object.
    hard_body_radius
        Combined hard-body radius [m].

    Returns
    -------
    PcResult
        ``details``: ``"added_variance_m2"`` (the maximising ``s``),
        ``"mahalanobis"`` (``K``) and ``"pc_analytic"`` (the Pc at Frisbee's
        analytic ``s``).

    Raises
    ------
    ValueError
        If ``hard_body_radius`` is not finite and positive, or the plane does
        not have exactly one unknown covariance.
    CovarianceError
        If the known covariance has no positive eigenvalue on the plane.
    """
    if len(plane.unknown_covariance) != 1:
        raise ValueError(
            "frisbee needs exactly one unknown covariance; unknown: "
            f"{plane.unknown_covariance or 'none'}"
        )
    gaussian = plane_gaussian_of(plane, hard_body_radius, None)
    k2 = gaussian.mahalanobis2
    miss = plane.miss_distance
    diagnostics = [
        (
            f"maximum Pc over the unknown {plane.unknown_covariance[0]} covariance "
            "(Frisbee 2015): an upper bound, not an estimate"
        ),
        *gaussian.diagnostics,
    ]
    analytic = miss * miss * (k2 - 1.0) / k2 if k2 > 1.0 else 0.0
    pc_zero = gaussian_pc(gaussian).value
    # A zero miss centres the hard body on the Gaussian: any added variance
    # lowers Pc (Anderson's theorem) and there is no miss direction.
    best = (0.0, pc_zero)
    pc_analytic = pc_zero
    if miss > 0.0:
        unit = plane.miss_vector / miss
        outer = np.outer(unit, unit)

        def pc(added: float) -> float:
            return _pc(
                plane.miss_vector, plane.covariance + added * outer, hard_body_radius
            )

        seed = best
        if analytic > 0.0:
            pc_analytic = pc(analytic)
            seed = _prefer(best, (analytic, pc_analytic))
        centre = math.log10(analytic if analytic > 0.0 else miss * miss)
        best = _prefer(seed, _search_log_max(pc, centre))
    added, value = best
    if added == 0.0:
        diagnostics.append(
            "the Pc is largest with zero added covariance: the maximum is the "
            "known covariance's Pc"
        )
    details = {
        "added_variance_m2": added,
        "mahalanobis": math.sqrt(k2),
        "pc_analytic": pc_analytic,
    }
    return PcResult.clamped(
        value, "frisbee", tuple(diagnostics), details, kind="upper_bound"
    )


def dilution(plane: EncounterPlane, hard_body_radius: float) -> PcResult:
    """Maximum Pc over a scale factor of the combined covariance.

    Parameters
    ----------
    plane
        Encounter plane with both covariances known.
    hard_body_radius
        Combined hard-body radius [m].

    Returns
    -------
    PcResult
        ``details``: ``"pc_nominal"`` (scale 1), ``"scale_factor"`` (the
        variance scale of the maximum; 0 when the miss is inside the hard
        body, 1 when no scale beats the nominal Pc by more than
        :data:`SEARCH_GAIN_RTOL`). ``diluted`` is true if the maximum needs
        a smaller covariance and exceeds the nominal Pc by more than
        :data:`DILUTION_TOLERANCE`.

    Raises
    ------
    ValueError
        If ``hard_body_radius`` is not finite and positive.
    CovarianceError
        If a covariance is unknown or the plane covariance is degenerate.
    """
    gaussian = plane_gaussian_of(plane, hard_body_radius, "dilution")
    nominal = gaussian_pc(gaussian).value
    miss = plane.miss_distance
    note = "maximum Pc over covariance scaling (dilution): an upper bound"
    if miss < hard_body_radius:
        details = {"pc_nominal": nominal, "scale_factor": 0.0}
        inside = _INSIDE.format(miss=miss, radius=hard_body_radius)
        return PcResult(
            1.0,
            "dilution",
            (note, inside),
            details,
            kind="upper_bound",
            diluted=True,
        )
    seed_scale = max(gaussian.mahalanobis2 / 2.0, SEED_SCALE_FLOOR)

    def pc(scale: float) -> float:
        return _pc(plane.miss_vector, scale * plane.covariance, hard_body_radius)

    seed = _prefer((1.0, nominal), (seed_scale, pc(seed_scale)))
    scale, value = _prefer(seed, _search_log_max(pc, math.log10(seed_scale)))
    diluted = scale < 1.0 and value > nominal * (1.0 + DILUTION_TOLERANCE)
    details = {"pc_nominal": nominal, "scale_factor": scale}
    return PcResult.clamped(
        value, "dilution", (note,), details, kind="upper_bound", diluted=diluted
    )


def alfano(plane: EncounterPlane, hard_body_radius: float) -> PcResult:
    """Maximum Pc for a miss distance alone, over every covariance (Alfano 2005).

    Parameters
    ----------
    plane
        Encounter plane; only its miss distance is used.
    hard_body_radius
        Combined hard-body radius [m].

    Returns
    -------
    PcResult
        ``details["sigma_m"]``: the standard deviation of the worst-case
        line-like covariance along the miss vector (0 inside or on the edge
        of the hard body, where the supremum 1 or 1/2 is reached only in the
        limit of a vanishing covariance).

    Raises
    ------
    ValueError
        If ``hard_body_radius`` is not finite and positive.
    """
    require_positive(hard_body_radius, "hard_body_radius")
    d, radius = plane.miss_distance, hard_body_radius
    note = (
        "maximum Pc over every covariance for this miss distance (Alfano 2005): "
        "an upper bound that uses no covariance information"
    )
    if d < radius:
        inside = _INSIDE.format(miss=d, radius=radius)
        return PcResult(
            1.0, "alfano", (note, inside), {"sigma_m": 0.0}, kind="upper_bound"
        )
    if d == radius:  # ln((d+R)/(d-R)) diverges: sigma -> 0, Pc -> 1/2
        edge = _EDGE.format(miss=d)
        return PcResult(
            0.5, "alfano", (note, edge), {"sigma_m": 0.0}, kind="upper_bound"
        )
    sigma = math.sqrt(2.0 * d * radius / math.log1p(2.0 * radius / (d - radius)))
    scale = math.sqrt(2.0) * sigma
    value = 0.5 * (math.erfc((d - radius) / scale) - math.erfc((d + radius) / scale))
    return PcResult.clamped(
        value, "alfano", (note,), {"sigma_m": sigma}, kind="upper_bound"
    )
