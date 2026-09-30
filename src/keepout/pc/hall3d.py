"""3-D Pc by Hall's 3D-Nc method (Hall 2021, J. Spacecraft Rockets 58(3)).

The expected number of collisions ``Nc`` is the time integral of the rate at
which the secondary's centre enters the hard-body sphere around the primary
(eqs. 51e, 52):

    Ncdot(t) = R^2 * integral over the unit sphere of
               N(R r; r_rel, A) * nu(r),
    nu(r) = sigma / sqrt(2 pi) * [exp(-x^2) - sqrt(pi) x erfc(x)],

with the relative position mean ``r_rel`` and covariance ``A`` and the
conditional relative-velocity statistics (``sigma``, ``x``) taken from a
first-order expansion of two-body motion in equinoctial elements. The
expansion is centred not on the mean orbits but on the states at the peak
overlap of the two position densities, found by iteration at each time
(sec. VII.B).

``Nc`` counts every entry into the hard-body sphere, so it is an upper bound
on Pc, the probability of at least one entry. For a single, short encounter
(a straight-line pass, each sample entering at most once) the two agree. For
several encounters in the window, or a long or zero-relative-velocity
encounter in which a sample can leave and re-enter, ``Nc`` exceeds Pc; a
diagnostic says so.
"""

from __future__ import annotations

import math

import numpy as np

from keepout._types import require_positive
from keepout.conjunction import Conjunction
from keepout.errors import PcConvergenceError
from keepout.pc._duration import DURATION_LIMIT, encounter_duration
from keepout.pc._rate import (
    CollisionRate,
    check_fallbacks,
    collision_rate,
    encounter_scan,
    reach,
    straight_line_closest_time,
)
from keepout.pc._timeline import QUAD_RTOL, RATE_FLOOR, RateIntegral, integrate_segments
from keepout.pc.result import PcResult

QUIET_AGREEMENT = 1e-4
"""Time-integral agreement below which a missed ``rtol`` is not reported. It
is per piece (the last two Gauss-Legendre orders of one piece of an
encounter segment, relative to the total Nc) and 10 times inside Hall's
tightest validation criterion (1e-3 relative to CARA), so it cannot affect
a result at that accuracy."""


def _diagnostics(
    integral: RateIntegral, window_given: bool, rtol: float = QUAD_RTOL
) -> list[str]:
    notes: list[str] = []
    nc = integral.nc
    if integral.peak <= 0.0:
        start, end = integral.window
        notes.append(
            f"collision rate negligible over the whole window [{start:.6g}, "
            f"{end:.6g}] s"
        )
    if nc > 1.0:
        notes.append(
            f"expected number of collisions Nc = {nc:.4g} exceeds 1; Pc set to 1"
        )
    if len(integral.segments) > 1:
        largest = max(s.nc for s in integral.segments)
        notes.append(
            f"{len(integral.segments)} encounters in the window: Pc lies between "
            f"the largest one's Nc ({largest:.3g}) and the total Nc ({nc:.3g}); "
            "the value is Nc (Hall 2021)"
        )
    if not window_given and integral.edge >= RATE_FLOOR:
        notes.append(
            f"the collision rate is {integral.edge:.1e} of its peak at the window "
            "edge (the window spans an orbital period): the encounter may be "
            "longer; pass window="
        )
    if integral.agreement > max(rtol, QUIET_AGREEMENT):
        notes.append(
            f"the time integral did not reach rtol: its worst piece agrees to "
            f"{integral.agreement:.1e} of Nc (rtol {rtol:.1e})"
        )
    return notes


def hall3d(
    conjunction: Conjunction,
    *,
    window: tuple[float, float] | None = None,
    rtol: float = QUAD_RTOL,
) -> PcResult:
    """3-D Pc of a conjunction by Hall's 3D-Nc method.

    Parameters
    ----------
    conjunction
        The conjunction; both covariances must be known (6x6, including
        velocity) and both orbits elliptic.
    window
        Integration window ``(start, end)`` [s from TCA]. ``None`` (default)
        grows a window about TCA and the straight-line closest approach until
        the rate at its edges is negligible, up to one shorter orbital period
        wide.
    rtol
        Relative tolerance of the time integral.

    Returns
    -------
    PcResult
        ``value = min(Nc, 1)``. ``details``: ``"nc"``, ``"window_start_s"``,
        ``"window_end_s"``, ``"encounters"`` (segments where the rate is not
        negligible), ``"largest_encounter_nc"`` (a lower bound on Pc),
        ``"evaluations"`` (rate evaluations) and ``"fallbacks"`` (the number
        of times the peak-overlap iteration failed in the far tail, outside
        every encounter).

    Raises
    ------
    CovarianceError
        If a covariance is unknown.
    GeometryError
        If an orbit is not elliptic.
    PcConvergenceError
        If the peak-overlap iteration fails where the rate matters, the rate
        cannot be computed, or the encounters in ``window`` need more scan
        points than allowed.
    ValueError
        If ``window`` is not finite with ``start < end``, or ``rtol`` is not
        finite and positive.
    """
    require_positive(rtol, "rtol")
    rate = collision_rate(conjunction, "hall3d")
    found = encounter_scan(conjunction, rate, window)
    integral = integrate_segments(rate, found, rtol=rtol)
    check_fallbacks(rate.fallbacks, found)
    if not math.isfinite(integral.nc):
        raise PcConvergenceError(f"Hall 3D-Nc: Nc is not finite ({integral.nc})")
    details = {
        "nc": integral.nc,
        "window_start_s": integral.window[0],
        "window_end_s": integral.window[1],
        "encounters": float(len(integral.segments)),
        "largest_encounter_nc": max((s.nc for s in integral.segments), default=0.0),
        "evaluations": float(rate.evaluations),
        "fallbacks": float(len(rate.fallbacks)),
    }
    notes = [
        *_floor_notes(rate),
        *_diagnostics(integral, window is not None, rtol),
        *_entry_notes(conjunction, rate, integral, window is not None),
    ]
    return PcResult.clamped(min(integral.nc, 1.0), "hall3d", tuple(notes), details)


def _floor_notes(rate: CollisionRate) -> list[str]:
    """Where :data:`VARIANCE_FLOOR` raised the combined covariance.

    As in the 2-D methods, an object whose position covariance is zero is
    perfectly known (only an all-zero 6x6 CDM covariance means unknown), and
    only the floor on the combined covariance is reported.
    """
    if rate.floored < rate.floor:
        return [
            f"combined position covariance: smallest variance {rate.floored:.3g} "
            f"m^2 raised to {rate.floor:.3g} m^2"
        ]
    return []


def _entry_notes(
    conjunction: Conjunction,
    rate: CollisionRate,
    integral: RateIntegral,
    window_given: bool,
) -> list[str]:
    """Notes on what ``Nc`` counts and on an encounter the scan did not find."""
    notes: list[str] = []
    repeated = (
        "Nc counts repeated entries into the hard-body sphere, so it is an "
        "upper bound on Pc"
    )
    if conjunction.relative_speed == 0.0:
        notes.append(f"relative velocity is zero: {repeated}")
    else:
        duration = encounter_duration(conjunction)
        period = rate.primary.period
        if duration > DURATION_LIMIT * period:
            notes.append(
                f"long encounter (about {duration:.3g} s, "
                f"{100.0 * duration / period:.2g}% of the primary's orbital "
                f"period): {repeated}"
            )
        if integral.peak <= 0.0 and not window_given:
            covariance = (
                rate.primary.position_covariance + rate.secondary.position_covariance
            )
            extent = reach(
                float(np.linalg.eigvalsh(covariance)[-1]), conjunction.hard_body_radius
            )
            miss = conjunction.plane.miss_distance
            if miss <= extent:
                notes.append(
                    f"the straight-line closest approach ({miss:.3g} m, "
                    f"t = {straight_line_closest_time(conjunction):.6g} s) is "
                    f"within {extent:.3g} m, but two-body motion keeps the "
                    "objects apart over the whole window: check that the "
                    "states are near TCA"
                )
    return notes
