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
from dataclasses import dataclass
from typing import Final

import numpy as np

from keepout._types import require_positive
from keepout.conjunction import Conjunction
from keepout.errors import PcConvergenceError
from keepout.pc._duration import encounter_duration, long_encounter
from keepout.pc._rate import (
    CollisionRate,
    check_fallbacks,
    collision_rate,
    encounter_scan,
    reach,
    straight_line_closest_time,
)
from keepout.pc._timeline import (
    RATE_FLOOR,
    RateIntegral,
    RateScan,
    integrate_segments,
    negligible_note,
)
from keepout.pc.result import Method, PcResult

__all__ = [
    "HALL_RTOL",
    "QUIET_AGREEMENT",
    "Encounters",
    "hall3d",
    "integrate",
    "is_long",
]

HALL_RTOL: Final = 1e-6
"""Default relative tolerance of the time integral: the agreement of two
nested Clenshaw-Curtis rules."""

QUIET_AGREEMENT: Final = 1e-4
"""Time-integral agreement below which a missed ``rtol`` is not reported. It
is per piece (the last two Clenshaw-Curtis rules of one piece of an
encounter segment, relative to the total Nc) and 10 times inside Hall's
tightest validation criterion (1e-3 relative to CARA), so it cannot affect
a result at that accuracy."""


def _integral_notes(
    integral: RateIntegral, *, window_given: bool, rtol: float
) -> list[str]:
    """Diagnostics on the time integral: negligible rate, Nc above 1, several
    encounters, a non-negligible window edge and a missed ``rtol``."""
    notes: list[str] = []
    nc = integral.nc
    scan = integral.scan
    if scan.peak <= 0.0:
        notes.append(negligible_note(scan))
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
    if not window_given and scan.edge >= RATE_FLOOR:
        notes.append(
            f"the collision rate is {scan.edge:.1e} of its peak at the window "
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
    rtol: float = HALL_RTOL,
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
    encounters = Encounters.scan(conjunction, "hall3d", window)
    return integrate(conjunction, encounters, rtol=rtol, long=is_long(conjunction))


@dataclass(frozen=True, eq=False)
class Encounters:
    """A conjunction's 3D-Nc collision rate and the scan locating its encounters.

    :func:`hall3d` integrates the rate over the scan's segments and
    :func:`~keepout.pc.montecarlo.montecarlo` samples trials only inside them,
    so one scan serves both (``compute_pc`` falls back from one to the other).

    Parameters
    ----------
    rate
        The collision rate.
    found
        Its scan over the window.
    window_given
        Whether the window was passed rather than sized automatically.
    scan_fallbacks
        Times [s] and rates where the peak-overlap iteration failed during
        the scan; ``rate.fallbacks`` also collects later failures (at Hall's
        quadrature nodes).
    """

    rate: CollisionRate
    found: RateScan
    window_given: bool
    scan_fallbacks: tuple[tuple[float, float], ...]

    @classmethod
    def scan(
        cls,
        conjunction: Conjunction,
        method: Method,
        window: tuple[float, float] | None,
    ) -> Encounters:
        """Scan the rate over ``window``, or an automatic one if ``None``.

        Raises
        ------
        CovarianceError
            If a covariance is unknown; the message names ``method``.
        GeometryError
            If an orbit is not elliptic.
        ValueError
            If ``window`` is not finite with ``start < end``.
        PcConvergenceError
            If the encounters cannot be resolved (see
            :func:`~keepout.pc._rate.encounter_scan`).
        """
        rate = collision_rate(conjunction, method)
        found = encounter_scan(conjunction, rate, window)
        return cls(rate, found, window is not None, tuple(rate.fallbacks))


def is_long(conjunction: Conjunction) -> bool:
    """Whether the encounter is long (see
    :func:`~keepout.pc._duration.long_encounter`), so that a sample can leave
    the hard-body sphere and enter it again."""
    if conjunction.relative_speed == 0.0:
        return False
    return long_encounter(conjunction, encounter_duration(conjunction)) is not None


def integrate(
    conjunction: Conjunction,
    encounters: Encounters,
    *,
    long: bool,
    rtol: float = HALL_RTOL,
) -> PcResult:
    """Hall's 3D-Nc over an encounter scan.

    The core of :func:`hall3d`, without its argument checks: for
    :func:`~keepout.pc.compute_pc`, which has already scanned the rate or
    tested the encounter's duration. ``long`` is :func:`is_long`; a long
    encounter only adds the note that Nc counts repeated entries (the
    duration itself is in ``compute_pc``'s applicability warning).
    """
    rate, found = encounters.rate, encounters.found
    integral = integrate_segments(rate, found, rtol=rtol)
    # Checked after integrating (Monte Carlo checks before sampling), so a
    # failure of the iteration at a quadrature node is caught too.
    check_fallbacks(rate.fallbacks, found)
    if not math.isfinite(integral.nc):
        raise PcConvergenceError(f"Hall 3D-Nc: Nc is not finite ({integral.nc})")
    details = {
        "nc": integral.nc,
        "window_start_s": found.window[0],
        "window_end_s": found.window[1],
        "encounters": float(len(integral.segments)),
        "largest_encounter_nc": max((s.nc for s in integral.segments), default=0.0),
        "evaluations": float(rate.evaluations),
        "fallbacks": float(len(rate.fallbacks)),
    }
    notes = [
        *_floor_notes(rate),
        *_integral_notes(integral, window_given=encounters.window_given, rtol=rtol),
        *_entry_notes(
            conjunction,
            rate,
            integral,
            window_given=encounters.window_given,
            long=long,
        ),
    ]
    return PcResult.clamped(min(integral.nc, 1.0), "hall3d", tuple(notes), details)


def _floor_notes(rate: CollisionRate) -> list[str]:
    """Where :data:`~keepout.pc._plane.SIGMA_FLOOR` raised the combined covariance.

    As in the 2-D methods, an object whose position covariance is zero is
    perfectly known (only an all-zero 6x6 CDM covariance means unknown), and
    only the floor on the combined covariance is reported.
    """
    if rate.smallest_floored_variance < rate.floor_variance:
        return [
            "combined position covariance: smallest variance "
            f"{rate.smallest_floored_variance:.3g} m^2 raised to "
            f"{rate.floor_variance:.3g} m^2"
        ]
    return []


def _entry_notes(
    conjunction: Conjunction,
    rate: CollisionRate,
    integral: RateIntegral,
    *,
    window_given: bool,
    long: bool,
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
        if long:
            notes.append(f"long encounter: {repeated}")
        if integral.scan.peak <= 0.0 and not window_given:
            extent = reach(
                float(np.linalg.eigvalsh(rate.position_covariance)[-1]),
                conjunction.hard_body_radius,
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
