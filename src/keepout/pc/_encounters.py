"""The encounters of a conjunction: its 3D-Nc collision rate and the scan that
locates where the rate is not negligible, shared by
:func:`~keepout.pc.hall3d.hall3d` and :func:`~keepout.pc.montecarlo.montecarlo`."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Final, Self

from keepout._types import override
from keepout.conjunction import Conjunction
from keepout.encounter import Role
from keepout.errors import GeometryError, PcConvergenceError
from keepout.pc._duration import straight_line_half_window
from keepout.pc._orbit import OrbitUncertainty
from keepout.pc._plane import require_known_covariance
from keepout.pc._rate import CollisionRate
from keepout.pc._screen import screen
from keepout.pc._timeline import RateScan, locate, locate_auto

__all__ = [
    "FALLBACK_TOLERANCE",
    "Encounters",
    "UnusableOrbit",
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
    segments and :func:`~keepout.pc.montecarlo.montecarlo` samples trials
    only inside them, so one scan serves both (``compute_pc`` falls back from
    one to the other).

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
        found = encounter_scan(conjunction, rate, window)
        return cls(rate, found, window is not None, tuple(rate.fallbacks))


class UnusableOrbit(GeometryError):
    """An object's orbit has no equinoctial elements (not elliptic,
    rectilinear, or on the elements' singularity), so the 3D-Nc rate is
    undefined.

    Parameters
    ----------
    role
        The object.
    reason
        Why, as :func:`~keepout.equinoctial.to_equinoctial` says; it is the
        message.
    """

    role: Role

    def __init__(self, role: Role, reason: str) -> None:
        super().__init__(reason)
        self.role = role

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[object, ...]]:
        # Unpickled (from a worker process) by its arguments, not the message.
        return type(self), (self.role, str(self))


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
    require_known_covariance(conjunction.unknown_covariance)
    return CollisionRate(
        _orbit(conjunction, "primary"),
        _orbit(conjunction, "secondary"),
        conjunction.hard_body_radius,
    )


def _orbit(conjunction: Conjunction, role: Role) -> OrbitUncertainty:
    """The orbit uncertainty of one object of a conjunction.

    Raises
    ------
    UnusableOrbit
        If the orbit has no equinoctial elements.
    """
    item = conjunction.primary if role == "primary" else conjunction.secondary
    try:
        return OrbitUncertainty.of(item)
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
        rate.position_covariance,
        conjunction.hard_body_radius,
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


def check_fallbacks(fallbacks: Sequence[tuple[float, float]], found: RateScan) -> None:
    """A check that the peak-overlap iteration did not fail where the rate may matter.

    A failure inside an encounter segment, or within one grid step of it, is
    fatal whatever the mean-centred rate there (that rate is what the
    iteration replaces, and with a large along-track uncertainty it can be
    ~0 where the true rate is near its peak); points where the mean-centred
    gap is negligible never reach the iteration. Elsewhere, in the far tail,
    a failure is fatal only if the mean-centred rate exceeds
    :data:`FALLBACK_TOLERANCE` of the peak.

    Raises
    ------
    PcConvergenceError
        If a failure in ``fallbacks`` is fatal by these rules.
    """
    times, peak = found.times, found.peak
    zones = [
        (
            float(times[first] - (times[first + 1] - times[first])),
            float(times[last] + (times[last] - times[last - 1])),
        )
        for first, last in found.index_ranges
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
