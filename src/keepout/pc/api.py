"""One entry point for every Pc method, with applicability checks."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import numpy as np

from keepout.conjunction import Conjunction
from keepout.errors import GeometryError, PcConvergenceError
from keepout.orbits import orbital_period
from keepout.pc.elrod import elrod
from keepout.pc.foster import foster
from keepout.pc.hall3d import hall3d
from keepout.pc.maxpc import alfano, dilution, frisbee
from keepout.pc.montecarlo import montecarlo
from keepout.pc.result import Method, PcResult

MethodChoice: TypeAlias = Literal["auto"] | Method
"""Methods :func:`compute_pc` accepts; ``"auto"`` picks one."""

_METHODS: dict[str, Callable[..., PcResult]] = {
    "foster": lambda c, **o: foster(c.plane, c.hard_body_radius, **o),
    "elrod": lambda c, **o: elrod(c.plane, c.hard_body_radius, **o),
    "frisbee": lambda c, **o: frisbee(c.plane, c.hard_body_radius, **o),
    "dilution": lambda c, **o: dilution(c.plane, c.hard_body_radius, **o),
    "alfano": lambda c, **o: alfano(c.plane, c.hard_body_radius, **o),
    "hall": hall3d,
    "montecarlo": montecarlo,
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

AUTO_MC_SAMPLES = 1_000_000
"""Trial budget of ``auto``'s Monte Carlo fallback (seconds to a minute)."""

AUTO_MC_SEED = 0
"""Seed of ``auto``'s Monte Carlo fallback, so ``auto`` is reproducible."""


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

    Both hold whether or not the covariances are known: the closest-approach
    offset does not depend on them, and with a covariance unknown the
    duration from the known one alone is a lower bound on the true duration,
    so a duration warning is still valid.

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
        lasts = "at least about" if conjunction.unknown_covariance else "about"
        warnings.append(
            f"2-D Pc assumptions questionable: the encounter lasts {lasts} "
            f"{duration:.3g} s, {100.0 * duration / period:.2g}% of the primary's "
            f"orbital period (limit {100.0 * DURATION_LIMIT:g}%); a 3-D Pc may "
            "differ"
        )
    plane = conjunction.plane
    along = abs(plane.closest_approach_offset)
    if along > TCA_OFFSET_LIMIT * float(np.linalg.norm(plane.relative_position)):
        warnings.append(
            f"states are {along:.3g} m from closest approach along the relative "
            "velocity; 2-D Pc uses the straight-line closest approach"
        )
    return tuple(warnings)


_Choice: TypeAlias = tuple[PcResult, tuple[str, ...], dict[str, float]]
"""A method's result, applicability notes and extra details."""


def _bound(conjunction: Conjunction) -> bool:
    return all(
        orbital_period(item.state) is not None
        for item in (conjunction.primary, conjunction.secondary)
    )


def _notes(conjunction: Conjunction) -> tuple[tuple[str, ...], dict[str, float]]:
    """Applicability notes and encounter duration, when they are meaningful.

    Only with an encounter plane (non-zero relative velocity). The notes are
    kept with a covariance unknown (see :func:`applicability`); the duration
    detail is not, since it is then only a lower bound.
    """
    if conjunction.relative_speed == 0.0:
        return (), {}
    duration = encounter_duration(conjunction)
    notes = applicability(conjunction, duration)
    if conjunction.unknown_covariance:
        return notes, {}
    return notes, {"encounter_duration_s": duration}


def _auto(conjunction: Conjunction) -> _Choice:
    """Pick and run a method; return its result, applicability notes and details."""
    if conjunction.relative_speed == 0.0:
        _require_three_d(conjunction)
        notes: tuple[str, ...] = ("relative velocity is zero: 2-D Pc is undefined",)
        return _three_d(conjunction, notes, {})
    unknown = conjunction.unknown_covariance
    notes, details = _notes(conjunction)
    if len(unknown) == 2:
        return alfano(conjunction.plane, conjunction.hard_body_radius), notes, details
    if unknown:
        return frisbee(conjunction.plane, conjunction.hard_body_radius), notes, details
    if not notes or not _bound(conjunction):
        return foster(conjunction.plane, conjunction.hard_body_radius), notes, details
    two_d = foster(conjunction.plane, conjunction.hard_body_radius).value
    return _three_d(conjunction, notes, {**details, "pc_2d": two_d})


def _require_three_d(conjunction: Conjunction) -> None:
    """Raise unless a 3-D method applies at zero relative velocity.

    Without an encounter plane only the 3-D methods remain, and they need
    both covariances and elliptic orbits.
    """
    reason = "no Pc method applies: the relative velocity is zero (no encounter plane)"
    items = {"primary": conjunction.primary, "secondary": conjunction.secondary}
    unknown = [
        f"{role} ({items[role].object_id})" for role in conjunction.unknown_covariance
    ]
    if len(unknown) == 2:
        raise GeometryError(
            f"{reason} and the covariances of {' and '.join(unknown)} are unknown"
        )
    if unknown:
        raise GeometryError(f"{reason} and the covariance of {unknown[0]} is unknown")
    if not _bound(conjunction):
        raise GeometryError(f"{reason} and an orbit is not elliptic")


def _three_d(
    conjunction: Conjunction, notes: tuple[str, ...], details: dict[str, float]
) -> _Choice:
    try:
        return hall3d(conjunction), notes, details
    except PcConvergenceError as error:
        fallback = (
            f"{error}; Monte Carlo used instead (seed {AUTO_MC_SEED}, at most "
            f"{AUTO_MC_SAMPLES:,} trials)"
        )
        result = montecarlo(conjunction, max_samples=AUTO_MC_SAMPLES, seed=AUTO_MC_SEED)
        return result, (*notes, fallback), details


def compute_pc(
    conjunction: Conjunction, method: MethodChoice = "auto", **options: Any
) -> PcResult:
    """Probability of collision of a conjunction.

    Parameters
    ----------
    conjunction
        The conjunction at TCA.
    method
        A method name, or ``"auto"`` (default), which picks:

        1. ``"hall"`` if the relative velocity is zero (both covariances
           must then be known and both orbits elliptic);
        2. ``"alfano"`` if both covariances are unknown, ``"frisbee"`` if one
           is (maximum Pc: upper bounds);
        3. ``"hall"`` if a 2-D assumption is questionable (see
           :func:`applicability`) and both orbits are elliptic, with the 2-D
           Pc kept in ``details["pc_2d"]``;
        4. ``"foster"`` otherwise.

        When ``"hall"`` raises :class:`PcConvergenceError` after its
        encounter scan (the peak-overlap iteration failed where the rate
        matters, the rate cannot be computed at a quadrature node, or Nc is
        not finite), ``"auto"`` falls back to ``"montecarlo"`` with at most
        ``AUTO_MC_SAMPLES`` trials and the seed ``AUTO_MC_SEED``, so
        ``"auto"`` is reproducible. A failure of the encounter scan itself
        is not covered: Monte Carlo runs the same scan
        (:func:`~keepout.pc.hall3d.encounter_scan`) and re-raises it. The 2-D and
        maximum-Pc methods take milliseconds; ``"hall"`` about 1-60 s and the
        Monte Carlo fallback up to about a minute.
    **options
        Passed to an explicitly chosen method (e.g. ``rtol``, ``order``,
        ``window``, ``samples``, ``seed``); not accepted with ``"auto"``.

    Returns
    -------
    PcResult
        ``diagnostics`` start with the conjunction's own (covariance repairs,
        unknown covariances), then applicability warnings, then the method's.
        Applicability warnings are added, whatever the method, whenever the
        encounter plane is defined (non-zero relative velocity) and they
        fire, also with a covariance unknown (maximum Pc): the
        closest-approach offset does not depend on the covariances, and the
        duration from the known covariance alone is a lower bound, so its
        warning still holds. ``details["encounter_duration_s"]`` is added
        only when both covariances are known.

    Raises
    ------
    ValueError
        If ``method`` is unknown, or options are given with ``"auto"``.
    TypeError
        If an option is not accepted by the method.
    CovarianceError
        If the chosen method needs a covariance that is unknown.
    GeometryError
        If the chosen method needs a plane or orbit that is undefined; with
        ``"auto"``, if the relative velocity is zero and a covariance is
        unknown or an orbit is not elliptic.
    PcConvergenceError
        If ``"hall"`` was chosen explicitly and did not converge; if
        ``"montecarlo"`` was chosen and its encounter scan failed; with
        ``"auto"``, if the encounter scan failed (its encounters need more
        detection points than allowed, the rate is significant where the
        screen ruled it negligible, or the rate cannot be computed on the scan
        grid), since the Monte Carlo fallback shares that scan.
    """
    if method != "auto" and method not in _METHODS:
        raise ValueError(
            f"unknown Pc method {method!r}; expected 'auto' or one of {tuple(_METHODS)}"
        )
    if method == "auto":
        if options:
            raise ValueError(f"options {sorted(options)} need an explicit method")
        result, notes, details = _auto(conjunction)
    else:
        result = _METHODS[method](conjunction, **options)
        notes, details = _notes(conjunction)
    return dataclasses.replace(
        result,
        diagnostics=conjunction.diagnostics + notes + result.diagnostics,
        details={**result.details, **details},
    )
