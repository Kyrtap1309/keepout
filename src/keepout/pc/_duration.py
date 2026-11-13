"""How long an encounter lasts, and when that is too long for 2-D Pc.

Used by :mod:`keepout.pc.api` (the applicability warning), by
:mod:`keepout.pc.hall3d` (the note that Nc counts repeated entries) and,
for the straight-line half-window, by the automatic 3D-Nc window.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import numpy as np

from keepout.encounter import sigma_along
from keepout.orbits import orbital_period
from keepout.pc._timeline import SIGMA_REACH

if TYPE_CHECKING:
    from keepout._types import FloatArray
    from keepout.conjunction import Conjunction

SIGMA_EXTENT: Final = 5.0
"""Standard deviations of the combined position covariance along the
relative velocity that the encounter is taken to span when estimating its
duration."""

DURATION_LIMIT: Final = 0.01
"""Largest encounter duration, as a fraction of the primary's orbital
period, for which the 2-D (straight-line) model is considered sound."""


def straight_line_half_window(
    relative_velocity: FloatArray,
    combined_position_covariance: FloatArray,
    hard_body_radius: float,
    n_sigma: float = SIGMA_REACH,
) -> float:
    """Half-window of a straight-line encounter [s].

    With ``sigma_along`` the combined position standard deviation along the
    relative velocity and ``R`` the hard-body radius, the rate is negligible
    beyond ``(n_sigma * sigma_along + R) / |v|`` (``n_sigma`` defaults to
    :data:`~keepout.pc._timeline.SIGMA_REACH`); infinite for zero relative velocity.
    """
    speed = float(np.linalg.norm(relative_velocity))
    if speed == 0.0:
        return math.inf
    sigma = sigma_along(relative_velocity / speed, combined_position_covariance)
    return (n_sigma * sigma + hard_body_radius) / speed


def encounter_duration(conjunction: Conjunction) -> float:
    """Time [s] the secondary needs to cross the combined uncertainty region.

    ``2 (SIGMA_EXTENT * sigma_along + HBR) / |v_rel|``, with ``sigma_along`` the
    standard deviation of the combined 3x3 position covariance along the
    relative velocity: only that extent is crossed during the encounter.

    Parameters
    ----------
    conjunction
        The conjunction at TCA; both covariances must be known.

    Returns
    -------
    float
        The encounter duration [s].
    """
    plane = conjunction.plane
    return 2.0 * straight_line_half_window(
        plane.relative_velocity,
        plane.combined_covariance,
        conjunction.combined_hard_body_radius,
        n_sigma=SIGMA_EXTENT,
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
