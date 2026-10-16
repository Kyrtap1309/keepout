"""One entry point for every Pc method, with applicability checks."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import (
    Concatenate,
    Final,
    Literal,
    NamedTuple,
    TypeAlias,
    overload,
)

import numpy as np

from keepout.conjunction import Conjunction
from keepout.encounter import Role
from keepout.errors import GeometryError, PcConvergenceError
from keepout.orbits import orbital_period
from keepout.pc._duration import (
    DURATION_LIMIT,
    SIGMA_EXTENT,
    encounter_duration,
    long_encounter,
)
from keepout.pc.elrod import elrod
from keepout.pc.foster import foster
from keepout.pc.hall3d import Encounters, hall3d, integrate
from keepout.pc.maxpc import alfano, dilution, frisbee
from keepout.pc.montecarlo import montecarlo, sample
from keepout.pc.result import Method, PcResult

__all__ = [
    "AUTO_MC_SAMPLES",
    "AUTO_MC_SEED",
    "DURATION_LIMIT",
    "SIGMA_EXTENT",
    "TCA_OFFSET_LIMIT",
    "MethodChoice",
    "applicability",
    "compute_pc",
    "encounter_duration",
]

MethodChoice: TypeAlias = Literal["auto"] | Method
"""Methods :func:`compute_pc` accepts; ``"auto"`` picks one."""

TCA_OFFSET_LIMIT: Final = 0.01
"""Largest relative-position component along the relative velocity, as a
fraction of the relative distance, before the states are reported as not
being at the time of closest approach."""

AUTO_MC_SAMPLES: Final = 1_000_000
"""Trial budget of ``auto``'s Monte Carlo fallback (seconds to a minute)."""

AUTO_MC_SEED: Final = 0
"""Seed of ``auto``'s Monte Carlo fallback, so ``auto`` is reproducible."""


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
    fraction = long_encounter(conjunction, duration)
    if fraction is not None:
        lasts = "at least about" if conjunction.unknown_covariance else "about"
        warnings.append(
            f"2-D Pc assumptions questionable: the encounter lasts {lasts} "
            f"{duration:.3g} s, {100.0 * fraction:.2g}% of the primary's "
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


class _Assessment(NamedTuple):
    """What :func:`compute_pc` adds to every method's result."""

    notes: tuple[str, ...]
    """Applicability warnings."""
    details: dict[str, float]
    """``encounter_duration_s``, when it is meaningful."""


def _assess(conjunction: Conjunction) -> _Assessment:
    """Applicability notes and encounter duration, when they are meaningful.

    Only with an encounter plane (non-zero relative velocity). The notes are
    kept with a covariance unknown (see :func:`applicability`); the duration
    detail is not, since it is then only a lower bound.
    """
    if conjunction.relative_speed == 0.0:
        return _Assessment((), {})
    duration = encounter_duration(conjunction)
    notes = applicability(conjunction, duration)
    details = (
        {} if conjunction.unknown_covariance else {"encounter_duration_s": duration}
    )
    return _Assessment(notes, details)


_Runner: TypeAlias = Callable[Concatenate[Conjunction, ...], PcResult]
"""A method run on a conjunction, with the method's keyword options."""


def _on_plane(method: Callable[..., PcResult]) -> _Runner:
    """Adapt a 2-D method ``f(plane, radius, **options)`` to a conjunction."""

    def run(conjunction: Conjunction, /, **options: object) -> PcResult:
        return method(conjunction.plane, conjunction.hard_body_radius, **options)

    return run


_METHODS: Final[Mapping[Method, _Runner]] = {
    "foster": _on_plane(foster),
    "elrod": _on_plane(elrod),
    "frisbee": _on_plane(frisbee),
    "dilution": _on_plane(dilution),
    "alfano": _on_plane(alfano),
    "hall3d": hall3d,
    "montecarlo": montecarlo,
}


_VELOCITY_METHODS: Final[frozenset[Method]] = frozenset({"hall3d", "montecarlo"})
"""The methods that use the velocity uncertainty (the 3-D ones)."""

_Choice: TypeAlias = tuple[PcResult, tuple[str, ...], dict[str, float]]
"""A method's result, applicability notes and extra details."""


def _bound(conjunction: Conjunction) -> bool:
    return all(
        orbital_period(item.state) is not None
        for item in (conjunction.primary, conjunction.secondary)
    )


def _auto(conjunction: Conjunction) -> _Choice:
    """Pick and run a method; return its result, applicability notes and details."""
    if conjunction.relative_speed == 0.0:
        _require_three_d(conjunction)
        notes: tuple[str, ...] = ("relative velocity is zero: 2-D Pc is undefined",)
        return _three_d(conjunction, notes, {})
    unknown = conjunction.unknown_covariance
    notes, details = _assess(conjunction)
    if len(unknown) == 2:
        return alfano(conjunction.plane, conjunction.hard_body_radius), notes, details
    if unknown:
        return frisbee(conjunction.plane, conjunction.hard_body_radius), notes, details
    if not notes or not _bound(conjunction):
        return foster(conjunction.plane, conjunction.hard_body_radius), notes, details
    position_only = conjunction.position_only_covariance
    if position_only:
        note = (
            f"{_known_only_in_position(conjunction, position_only)}: the 3-D "
            "methods need the velocity uncertainty, so the 2-D Pc is used"
        )
        return (
            foster(conjunction.plane, conjunction.hard_body_radius),
            (*notes, note),
            details,
        )
    two_d = foster(conjunction.plane, conjunction.hard_body_radius).value
    return _three_d(conjunction, notes, {**details, "pc_2d": two_d})


def _known_only_in_position(conjunction: Conjunction, roles: tuple[Role, ...]) -> str:
    """``the covariance of <object> is known only in position (...)``."""
    labels = " and ".join(conjunction.label(role) for role in roles)
    whose = "covariances of" if len(roles) == 2 else "covariance of"
    verb = "are" if len(roles) == 2 else "is"
    return f"the {whose} {labels} {verb} known only in position (zero velocity block)"


def _velocity_notes(conjunction: Conjunction, method: Method) -> tuple[str, ...]:
    """A note when ``method`` needs the velocity uncertainty and a covariance
    has none: whichever way the method was chosen, it takes that velocity as
    exact."""
    position_only = conjunction.position_only_covariance
    if method not in _VELOCITY_METHODS or not position_only:
        return ()
    return (
        f"{_known_only_in_position(conjunction, position_only)}: {method} "
        "takes its velocity as exact",
    )


def _require_three_d(conjunction: Conjunction) -> None:
    """Raise unless a 3-D method applies at zero relative velocity.

    Without an encounter plane only the 3-D methods remain, and they need
    both covariances and elliptic orbits.

    Raises
    ------
    GeometryError
        If a covariance is unknown or an orbit is not elliptic.
    """
    reason = "no Pc method applies: the relative velocity is zero (no encounter plane)"
    unknown = [conjunction.label(role) for role in conjunction.unknown_covariance]
    if len(unknown) == 2:
        raise GeometryError(
            f"{reason} and the covariances of {' and '.join(unknown)} are unknown"
        )
    if unknown:
        raise GeometryError(f"{reason} and the covariance of {unknown[0]} is unknown")
    if not _bound(conjunction):
        raise GeometryError(f"{reason} and an orbit is not elliptic")


def _three_d(
    conjunction: Conjunction,
    notes: tuple[str, ...],
    details: dict[str, float],
) -> _Choice:
    """Hall's 3D-Nc, or Monte Carlo over the same encounter scan if it fails."""
    encounters = Encounters.scan(conjunction, "hall3d", None)
    try:
        return integrate(conjunction, encounters), notes, details
    except PcConvergenceError as error:
        fallback = (
            f"{error}; Monte Carlo used instead (seed {AUTO_MC_SEED}, at most "
            f"{AUTO_MC_SAMPLES:,} trials)"
        )
        result = sample(
            conjunction, encounters, max_samples=AUTO_MC_SAMPLES, seed=AUTO_MC_SEED
        )
        return result, (*notes, fallback), details


@overload
def compute_pc(conjunction: Conjunction, method: Literal["auto"] = ...) -> PcResult: ...


@overload
def compute_pc(
    conjunction: Conjunction, method: Literal["foster"], *, rtol: float = ...
) -> PcResult: ...


@overload
def compute_pc(
    conjunction: Conjunction, method: Literal["elrod"], *, order: int | None = ...
) -> PcResult: ...


@overload
def compute_pc(
    conjunction: Conjunction, method: Literal["frisbee", "dilution", "alfano"]
) -> PcResult: ...


@overload
def compute_pc(
    conjunction: Conjunction,
    method: Literal["hall3d"],
    *,
    window: tuple[float, float] | None = ...,
    rtol: float = ...,
) -> PcResult: ...


@overload
def compute_pc(
    conjunction: Conjunction,
    method: Literal["montecarlo"],
    *,
    window: tuple[float, float] | None = ...,
    samples: int | None = ...,
    rtol: float = ...,
    max_samples: int = ...,
    confidence: float = ...,
    seed: int | None = ...,
) -> PcResult: ...


def compute_pc(
    conjunction: Conjunction, method: MethodChoice = "auto", **options: object
) -> PcResult:
    """Probability of collision of a conjunction.

    Parameters
    ----------
    conjunction
        The conjunction at TCA.
    method
        A method name, or ``"auto"`` (default), which picks:

        1. ``"hall3d"`` if the relative velocity is zero (both covariances
           must then be known and both orbits elliptic);
        2. ``"alfano"`` if both covariances are unknown, ``"frisbee"`` if one
           is (maximum Pc: upper bounds);
        3. ``"hall3d"`` if a 2-D assumption is questionable (see
           :func:`applicability`), both orbits are elliptic and both
           covariances have a velocity block, with the 2-D Pc kept in
           ``details["pc_2d"]``;
        4. ``"foster"`` otherwise, with a note when a 2-D assumption is
           questionable but a covariance is known only in position (zero
           velocity block, e.g. a synthetic covariance; see
           :attr:`~keepout.conjunction.Conjunction.position_only_covariance`):
           the 3-D methods would take its velocity as exact.

        However chosen, ``"hall3d"`` and ``"montecarlo"`` note a covariance
        known only in position: they take its velocity as exact.

        When ``"hall3d"`` raises :class:`PcConvergenceError` after its
        encounter scan (the peak-overlap iteration failed where the rate
        matters, the rate cannot be computed at a quadrature node, or Nc is
        not finite), ``"auto"`` falls back to ``"montecarlo"`` with at most
        ``AUTO_MC_SAMPLES`` trials and the seed ``AUTO_MC_SEED``, so
        ``"auto"`` is reproducible. A failure of the encounter scan itself
        is not covered: Monte Carlo reuses that scan
        (:func:`~keepout.pc._rate.encounter_scan`), so its error propagates.
        The 2-D and maximum-Pc methods take milliseconds; ``"hall3d"`` about
        1-60 s and the Monte Carlo fallback up to about a minute.
    **options
        The keyword-only parameters of an explicitly chosen method (e.g.
        ``rtol``, ``order``, ``window``, ``samples``, ``seed``), as typed by
        the overloads; none with ``"auto"`` or the maximum-Pc methods.

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
        If ``"hall3d"`` was chosen explicitly and did not converge; if
        ``"montecarlo"`` was chosen and its encounter scan failed; with
        ``"auto"``, if the encounter scan failed (its encounters need more
        detection points than allowed, the rate is significant where the
        screen ruled it negligible, or the rate cannot be computed on the scan
        grid), since the Monte Carlo fallback shares that scan.
    """
    if method == "auto":
        if options:
            raise ValueError(f"options {sorted(options)} need an explicit method")
        result, notes, details = _auto(conjunction)
    else:
        run = _METHODS.get(method)
        if run is None:
            raise ValueError(
                f"unknown Pc method {method!r}; expected 'auto' or one of "
                f"{tuple(_METHODS)}"
            )
        notes, details = _assess(conjunction)
        result = run(conjunction, **options)
    notes += _velocity_notes(conjunction, result.method)
    return dataclasses.replace(
        result,
        diagnostics=conjunction.diagnostics + notes + result.diagnostics,
        details={**result.details, **details},
    )
