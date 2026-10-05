"""How long an encounter lasts, and when that is too long for 2-D Pc.

Used by :mod:`keepout.pc.api` (the applicability warning) and by
:mod:`keepout.pc.hall3d` (the note that Nc counts repeated entries).
"""

from __future__ import annotations

from typing import Final

from keepout.conjunction import Conjunction
from keepout.orbits import orbital_period
from keepout.pc._timeline import straight_line_half_window

SIGMA_EXTENT: Final = 5.0
"""Standard deviations of the combined position covariance along the
relative velocity that the encounter is taken to span when estimating its
duration."""

DURATION_LIMIT: Final = 0.01
"""Largest encounter duration, as a fraction of the primary's orbital
period, for which the 2-D (straight-line) model is considered sound."""


def encounter_duration(conjunction: Conjunction) -> float:
    """Time [s] the secondary needs to cross the combined uncertainty region.

    ``2 (SIGMA_EXTENT * sigma_v + HBR) / |v_rel|``, with ``sigma_v`` the
    standard deviation of the combined 3x3 position covariance along the
    relative velocity: only that extent is crossed during the encounter.
    """
    plane = conjunction.plane
    return 2.0 * straight_line_half_window(
        plane.relative_velocity,
        plane.combined_covariance,
        conjunction.hard_body_radius,
        sigmas=SIGMA_EXTENT,
    )


def long_encounter(conjunction: Conjunction, duration: float) -> float | None:
    """``duration`` as a fraction of the primary's orbital period, if too long.

    ``None`` if it is within :data:`DURATION_LIMIT`, or if the primary is
    unbound (no period). The period is the two-body one of the primary's
    state, which needs no covariance, so the check holds with a covariance
    unknown too.
    """
    period = orbital_period(conjunction.primary.state)
    if period is None or duration <= DURATION_LIMIT * period:
        return None
    return duration / period
