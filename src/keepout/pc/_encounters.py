"""The encounters of a conjunction: its 3D-Nc collision rate and the scan that
locates where the rate is not negligible, shared by
:func:`~keepout.pc.hall3d.hall3d` and :func:`~keepout.pc.montecarlo.montecarlo`."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Final

from keepout.errors import GeometryError, PcConvergenceError
from keepout.pc._duration import straight_line_half_window
from keepout.pc._errors import MissingCovariance, UnusableOrbit
from keepout.pc._orbit import OrbitUncertainty
from keepout.pc._rate import CollisionRate
from keepout.pc._screen import screen
from keepout.pc._timeline import RateScan, locate, locate_auto

if TYPE_CHECKING:
    from collections.abc import Sequence

    from keepout.conjunction import Conjunction
    from keepout.covariance import Covariance
    from keepout.encounter import Role
    from keepout.state import StateVector

__all__ = [
    "FALLBACK_TOLERANCE",
    "Encounters",
    "check_fallbacks",
    "collision_rate",
    "encounter_scan",
    "straight_line_closest_time",
]

FALLBACK_TOLERANCE: Final = 1e-10
"""Largest rate, relative to the peak, allowed where the peak-overlap
iteration failed and the mean-centred expansion was used instead."""


@dataclass(frozen=True, eq=False, slots=True)
class Encounters:
    """A conjunction's 3D-Nc collision rate and the scan locating its encounters.

    :func:`~keepout.pc.hall3d.hall3d` integrates the rate over the scan's
    segments and :func:`~keepout.pc.montecarlo.montecarlo` samples only
    inside them, so one scan serves both (``compute_pc`` falls back from
    one to the other).

    Parameters
    ----------
    rate
        The collision rate.
    scan
        Its scan over the window.
    window_given
        Whether the window was passed rather than sized automatically.
    scan_fallbacks
        Times [s] and rates where the peak-overlap iteration failed during
        the scan (``rate.fallbacks`` also collects later failures, at Hall's
        quadrature nodes).
    """

    rate: CollisionRate
    scan: RateScan
    window_given: bool
    scan_fallbacks: tuple[tuple[float, float], ...]

    @classmethod
    def locate(
        cls,
        conjunction: Conjunction,
        window: tuple[float, float] | None,
    ) -> Encounters:
        """The encounters found by scanning ``window``, or an automatic one if ``None``.

        Raises
        ------
        MissingCovariance
            If a covariance is unknown (a :class:`CovarianceError` naming
            no method: the callers do).
        UnusableOrbit
            If an orbit has no equinoctial elements (a
            :class:`GeometryError`, with the object).
        ValueError
            If ``window`` is not finite with ``start < end``.
        PcConvergenceError
            If the encounters cannot be resolved (they need more detection
            points than allowed, the rate is significant where the screen
            ruled it negligible, or the rate cannot be computed on the scan
            grid).
        """
        rate = collision_rate(conjunction)
        scan = encounter_scan(conjunction, rate, window)
        return cls(rate, scan, window is not None, tuple(rate.fallbacks))


def collision_rate(conjunction: Conjunction) -> CollisionRate:
    """The 3D-Nc collision rate of a conjunction with both covariances known.

    It names no method: :func:`~keepout.pc.hall3d.hall3d` and
    :func:`~keepout.pc.montecarlo.montecarlo` check the covariances
    themselves, and :func:`~keepout.pc.compute_pc` reads these errors.

    Raises
    ------
    MissingCovariance
        If a covariance is unknown.
    UnusableOrbit
        If an orbit has no equinoctial elements.
    """
    primary, secondary = conjunction.primary, conjunction.secondary
    if primary.covariance is None or secondary.covariance is None:
        raise MissingCovariance(conjunction.unknown_covariance)
    return CollisionRate(
        _orbit(primary.state, primary.covariance, "primary"),
        _orbit(secondary.state, secondary.covariance, "secondary"),
        conjunction.combined_hard_body_radius,
    )


def _orbit(state: StateVector, covariance: Covariance, role: Role) -> OrbitUncertainty:
    """One object's orbit uncertainty.

    Raises
    ------
    UnusableOrbit
        If the orbit has no equinoctial elements.
    """
    try:
        return OrbitUncertainty.of(state, covariance)
    except GeometryError as error:
        raise UnusableOrbit(role, str(error)) from error


def encounter_scan(
    conjunction: Conjunction, rate: CollisionRate, window: tuple[float, float] | None
) -> RateScan:
    """The rate's scan over ``window`` or an automatically sized one.

    The automatic window covers TCA (t = 0) and the straight-line closest
    approach of the states, ``t* = -(r . v) / |v|^2``, widened on both sides
    by the straight-line half-window of :func:`straight_line_half_window`, doubled
    until the rate at the edges is negligible (or, while the rate is 0
    everywhere scanned, until an encounter appears). ``t*`` is clipped to
    half the shorter orbital period ``P`` and the window is never wider than
    ``P``, so no encounter of the next revolution is counted twice.

    Raises
    ------
    ValueError
        If ``window`` is not finite with ``start < end``.
    PcConvergenceError
        If the encounters in the window cannot be resolved.
    """
    screened = partial(screen, rate)
    if window is not None:
        start, end = (float(v) for v in window)
        if not (math.isfinite(start) and math.isfinite(end) and start < end):
            raise ValueError(f"window must be finite with start < end, got {window}")
        return locate(rate, (start, end), screened)
    half = straight_line_half_window(
        conjunction.relative_velocity,
        rate.combined_position_covariance,
        conjunction.combined_hard_body_radius,
    )
    period = min(rate.primary.period, rate.secondary.period)
    closest = min(
        max(straight_line_closest_time(conjunction), -0.5 * period), 0.5 * period
    )
    centre = (min(0.0, closest), max(0.0, closest))
    limit = 0.5 * (period - (centre[1] - centre[0]))
    return locate_auto(rate, half, limit, screened, centre=centre)


def straight_line_closest_time(conjunction: Conjunction) -> float:
    """Time [s from TCA] of the straight-line closest approach of the states.

    ``-(r . v) / |v|^2`` with the relative position ``r`` and velocity ``v``;
    0 at zero relative velocity.
    """
    velocity = conjunction.relative_velocity
    speed2 = float(velocity @ velocity)
    if speed2 == 0.0:
        return 0.0
    return -float(conjunction.relative_position @ velocity) / speed2


def check_fallbacks(fallbacks: Sequence[tuple[float, float]], scan: RateScan) -> None:
    """A check that the peak-overlap iteration did not fail where the rate may matter.

    A failure inside an encounter segment, or within one grid step of it, is
    fatal whatever the mean-centred rate there (that rate is what the
    iteration replaces, and with a large along-track uncertainty it can be
    ~0 where the true rate is near its peak); points where the mean-centred
    clearance is negligible never reach the iteration. Elsewhere, in the far tail,
    a failure is fatal only if the mean-centred rate exceeds
    :data:`FALLBACK_TOLERANCE` of the peak.

    Raises
    ------
    PcConvergenceError
        If a failure in ``fallbacks`` is fatal by these rules.
    """
    times, peak = scan.times, scan.peak
    zones = [
        (
            float(times[first] - (times[first + 1] - times[first])),
            float(times[last] + (times[last] - times[last - 1])),
        )
        for first, last in scan.index_ranges
        if last > first
    ]
    for t, value in fallbacks:
        inside = any(a <= t <= b for a, b in zones)
        if inside or value > FALLBACK_TOLERANCE * peak:
            where = (
                "inside an encounter segment"
                if inside
                else f"where the collision rate is {value / peak:.1e} of its peak"
            )
            raise PcConvergenceError(
                f"Hall 3D-Nc: the peak-overlap iteration did not converge at "
                f"t = {t:.6g} s from TCA, {where}"
            )
