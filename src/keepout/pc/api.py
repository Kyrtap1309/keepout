"""One entry point for every Pc method, with applicability checks."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import numpy as np

from keepout.conjunction import Conjunction
from keepout.orbits import orbital_period
from keepout.pc.elrod import elrod
from keepout.pc.foster import foster
from keepout.pc.result import Method, PcResult

MethodChoice: TypeAlias = Literal["auto"] | Method
"""Methods :func:`compute_pc` accepts; ``"auto"`` picks one."""

_METHODS: dict[str, Callable[..., PcResult]] = {
    "auto": foster,
    "foster": foster,
    "elrod": elrod,
}

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
    plane = conjunction.plane
    direction = plane.direction
    variance = float(direction @ plane.combined_covariance @ direction)
    extent = SIGMA_EXTENT * math.sqrt(max(variance, 0.0)) + conjunction.hard_body_radius
    return 2.0 * extent / plane.relative_speed


def applicability(conjunction: Conjunction, duration: float) -> tuple[str, ...]:
    """Warnings about the assumptions behind 2-D Pc for this conjunction.

    Parameters
    ----------
    conjunction
        The conjunction at TCA.
    duration
        Its :func:`encounter_duration` [s].
    """
    warnings: list[str] = []
    period = orbital_period(conjunction.primary.state)
    if period is not None and duration > DURATION_LIMIT * period:
        warnings.append(
            f"2-D Pc assumptions questionable: the encounter lasts about "
            f"{duration:.3g} s, {100.0 * duration / period:.2g}% of the primary's "
            f"orbital period (limit {100.0 * DURATION_LIMIT:g}%); a 3-D Pc may "
            "differ"
        )
    plane = conjunction.plane
    along = abs(plane.closest_approach_offset)
    if along > TCA_OFFSET_LIMIT * float(np.linalg.norm(plane.relative_position)):
        warnings.append(
            f"states are {along:.3g} m from closest approach along the relative "
            "velocity; Pc uses the straight-line closest approach"
        )
    return tuple(warnings)


def compute_pc(
    conjunction: Conjunction, method: MethodChoice = "auto", **options: Any
) -> PcResult:
    """Probability of collision of a conjunction.

    Parameters
    ----------
    conjunction
        The conjunction at TCA.
    method
        ``"foster"``, ``"elrod"`` or ``"auto"`` (currently Foster, the most
        accurate 2-D method).
    **options
        Passed to the method: ``rtol`` for :func:`~keepout.pc.foster.foster`,
        ``order`` for :func:`~keepout.pc.elrod.elrod`.

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
    TypeError
        If an option is not accepted by the method.
    """
    if method not in _METHODS:
        raise ValueError(
            f"unknown Pc method {method!r}; expected one of {tuple(_METHODS)}"
        )
    result = _METHODS[method](
        conjunction.plane, conjunction.hard_body_radius, **options
    )
    duration = encounter_duration(conjunction)
    return dataclasses.replace(
        result,
        diagnostics=conjunction.diagnostics
        + applicability(conjunction, duration)
        + result.diagnostics,
        details={**result.details, "encounter_duration_s": duration},
    )
