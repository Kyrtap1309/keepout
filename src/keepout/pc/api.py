"""One entry point for every Pc method, with applicability checks."""

from __future__ import annotations

import math
from typing import Literal, TypeAlias, get_args

import numpy as np

from keepout.conjunction import Conjunction
from keepout.objects import ObjectState
from keepout.pc.elrod import elrod
from keepout.pc.foster import foster
from keepout.pc.result import PcResult

MethodChoice: TypeAlias = Literal["auto", "foster", "elrod"]
"""Methods :func:`compute_pc` accepts; ``"auto"`` picks one."""

EARTH_MU = 3.986004418e14
"""Earth's gravitational parameter [m^3/s^2] (WGS 84)."""

SIGMA_EXTENT = 5.0
"""Standard deviations of the combined position covariance along the
relative velocity that the encounter is taken to span when estimating its
duration."""

DURATION_LIMIT = 0.01
"""Largest encounter duration, as a fraction of the primary's orbital
period, for which the 2-D (straight-line) model is considered sound."""

TCA_OFFSET_LIMIT = 0.01
"""Largest relative-position component along the relative velocity, as a
fraction of the relative distance, before the states are reported as not
being at the time of closest approach."""


def encounter_duration(conjunction: Conjunction) -> float:
    """Time [s] the secondary needs to cross the combined uncertainty region.

    ``2 (SIGMA_EXTENT * sigma_v + HBR) / |v_rel|``, with ``sigma_v`` the
    standard deviation of the combined 3x3 position covariance along the
    relative velocity: only that extent is crossed during the encounter.
    """
    combined = (
        conjunction.primary.covariance.position
        + conjunction.secondary.covariance.position
    )
    direction = conjunction.plane.relative_velocity / conjunction.relative_speed
    sigma_along = math.sqrt(max(float(direction @ combined @ direction), 0.0))
    extent = SIGMA_EXTENT * sigma_along + conjunction.hard_body_radius
    return 2.0 * extent / conjunction.relative_speed


def orbital_period(obj: ObjectState) -> float | None:
    """Two-body orbital period [s] from a GCRF state; ``None`` if unbound."""
    r = float(np.linalg.norm(obj.state.r))
    v = float(np.linalg.norm(obj.state.v))
    energy = 0.5 * v * v - EARTH_MU / r
    if energy >= 0.0:
        return None
    semi_major_axis = -EARTH_MU / (2.0 * energy)
    return 2.0 * math.pi * math.sqrt(semi_major_axis**3 / EARTH_MU)


def applicability(conjunction: Conjunction) -> tuple[str, ...]:
    """Warnings about the assumptions behind 2-D Pc for this conjunction."""
    warnings: list[str] = []
    duration = encounter_duration(conjunction)
    period = orbital_period(conjunction.primary)
    if period is not None and duration > DURATION_LIMIT * period:
        warnings.append(
            f"2-D Pc assumptions questionable: the encounter lasts about "
            f"{duration:.3g} s, {100.0 * duration / period:.2g}% of the primary's "
            f"orbital period (limit {100.0 * DURATION_LIMIT:g}%); a 3-D Pc may "
            "differ"
        )
    plane = conjunction.plane
    distance = float(np.linalg.norm(plane.relative_position))
    along = (
        abs(float(plane.relative_position @ plane.relative_velocity))
        / conjunction.relative_speed
    )
    if along > TCA_OFFSET_LIMIT * distance:
        warnings.append(
            f"states are {along:.3g} m from closest approach along the relative "
            "velocity; Pc uses the straight-line closest approach"
        )
    return tuple(warnings)


def compute_pc(conjunction: Conjunction, method: MethodChoice = "auto") -> PcResult:
    """Probability of collision of a conjunction.

    Parameters
    ----------
    conjunction
        The conjunction at TCA.
    method
        ``"foster"``, ``"elrod"`` or ``"auto"`` (currently Foster, the most
        accurate 2-D method).

    Returns
    -------
    PcResult
        ``diagnostics`` start with the conjunction's own (covariance repairs),
        then applicability warnings, then the method's. ``details`` include
        ``"encounter_duration_s"``.

    Raises
    ------
    ValueError
        If ``method`` is unknown.
    """
    if method not in get_args(MethodChoice):
        raise ValueError(
            f"unknown Pc method {method!r}; expected one of {get_args(MethodChoice)}"
        )
    plane, radius = conjunction.plane, conjunction.hard_body_radius
    result = elrod(plane, radius) if method == "elrod" else foster(plane, radius)
    return PcResult(
        result.value,
        result.method,
        conjunction.diagnostics + applicability(conjunction) + result.diagnostics,
        {**result.details, "encounter_duration_s": encounter_duration(conjunction)},
    )
