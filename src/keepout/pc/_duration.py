"""How long an encounter lasts, and when that is too long for 2-D Pc.

Used by :mod:`keepout.pc.api` (the applicability warning) and by
:mod:`keepout.pc.hall3d` (the note that Nc counts repeated entries).
"""

from __future__ import annotations

from keepout.conjunction import Conjunction
from keepout.encounter import sigma_along

SIGMA_EXTENT = 5.0
"""Standard deviations of the combined position covariance along the
relative velocity that the encounter is taken to span when estimating its
duration."""

DURATION_LIMIT = 0.01
"""Largest encounter duration, as a fraction of the primary's orbital
period, for which the 2-D (straight-line) model is considered sound."""


def encounter_duration(conjunction: Conjunction) -> float:
    """Time [s] the secondary needs to cross the combined uncertainty region.

    ``2 (SIGMA_EXTENT * sigma_v + HBR) / |v_rel|``, with ``sigma_v`` the
    standard deviation of the combined 3x3 position covariance along the
    relative velocity: only that extent is crossed during the encounter.
    """
    plane = conjunction.plane
    sigma = sigma_along(plane.direction, plane.combined_covariance)
    extent = SIGMA_EXTENT * sigma + conjunction.hard_body_radius
    return 2.0 * extent / plane.relative_speed
