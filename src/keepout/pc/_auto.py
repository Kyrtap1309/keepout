"""The ``"auto"`` choice of :func:`~keepout.pc.compute_pc`, and the
applicability assessment of 2-D Pc it rests on."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, NamedTuple

import numpy as np

from keepout.errors import GeometryError, PcConvergenceError
from keepout.orbits import orbital_period
from keepout.pc._duration import DURATION_LIMIT, encounter_duration, long_encounter
from keepout.pc._encounters import Encounters, UnusableOrbit
from keepout.pc._plane import MissingCovariance
from keepout.pc.foster import foster
from keepout.pc.hall3d import hall3d_over_scan
from keepout.pc.maxpc import alfano, frisbee
from keepout.pc.montecarlo import montecarlo_over_scan

if TYPE_CHECKING:
    from keepout.conjunction import Conjunction
    from keepout.encounter import Role
    from keepout.pc.result import Method, PcResult

__all__ = [
    "AUTO_MC_SAMPLES",
    "AUTO_MC_SEED",
    "TCA_OFFSET_LIMIT",
    "Assessment",
    "applicability",
    "assess",
    "auto",
    "auto_options",
    "known_only_in_position",
]

AUTO_MC_SAMPLES: Final = 1_000_000
"""Sample budget of ``auto``'s Monte Carlo fallback (seconds to a minute)."""

AUTO_MC_SEED: Final = 0
"""Seed of ``auto``'s Monte Carlo fallback, so ``auto`` is reproducible."""

TCA_OFFSET_LIMIT: Final = 0.01
"""Largest relative-position component along the relative velocity, as a
fraction of the relative distance, before the states are reported as not
being at the time of closest approach."""

_NO_PLANE: Final = (
    "no Pc method applies: the relative velocity is zero (no encounter plane)"
)
"""Start of ``"auto"``'s error when the 3-D methods, the only ones left at
zero relative velocity, cannot run."""


class Assessment(NamedTuple):
    """What :func:`~keepout.pc.compute_pc` adds to a method's result."""

    notes: tuple[str, ...]
    """Applicability warnings (and, for ``"auto"``, why it chose so)."""
    details: dict[str, float]
    """``encounter_duration_s`` when it is meaningful (and ``pc_2d`` when
    ``"auto"`` chose a 3-D method over Foster's)."""


def auto(conjunction: Conjunction) -> tuple[PcResult, Assessment]:
    """The automatic choice: the chosen method's result and its assessment.

    The rules are those of :func:`~keepout.pc.compute_pc`, numbered alike.

    Raises
    ------
    GeometryError
        If the relative velocity is zero and the 3-D methods cannot run.
    """
    # Rule 1: no encounter plane, so a 3-D method.
    if not conjunction.has_plane:
        return _auto_without_plane(conjunction)
    unknown = conjunction.unknown_covariance
    assessment = assess(conjunction)
    notes, details = assessment
    questionable = bool(notes)
    plane, hard_body_radius = conjunction.plane, conjunction.combined_hard_body_radius
    # Rule 2: maximum Pc for unknown covariances.
    if len(unknown) == 2:
        return alfano(plane, hard_body_radius), assessment
    if unknown:
        return frisbee(plane, hard_body_radius), assessment
    if questionable and _both_orbits_elliptic(conjunction):
        position_only = conjunction.position_only_covariance
        if not position_only:
            # Rule 3: a questionable 2-D assumption, and the 3-D methods can run.
            pc_2d = foster(plane, hard_body_radius).value
            encounters = Encounters.locate(conjunction, None)
            return _three_d(conjunction, encounters, notes, {**details, "pc_2d": pc_2d})
        # Rule 4, noting why the 3-D methods are not used.
        note = (
            f"{known_only_in_position(conjunction, position_only)}: the 3-D "
            "methods need the velocity uncertainty, so the 2-D Pc is used"
        )
        return foster(plane, hard_body_radius), Assessment((*notes, note), details)
    # Rule 4.
    return foster(plane, hard_body_radius), assessment


def _auto_without_plane(conjunction: Conjunction) -> tuple[PcResult, Assessment]:
    """The automatic choice at zero relative velocity: a 3-D method.

    Raises
    ------
    GeometryError
        If the 3-D methods cannot run.
    """
    try:
        encounters = Encounters.locate(conjunction, None)
    except MissingCovariance as exc:
        raise GeometryError(
            f"{_NO_PLANE} and {_covariances_of(conjunction, exc.roles)} unknown"
        ) from exc
    except UnusableOrbit as exc:
        raise GeometryError(
            f"{_NO_PLANE} and the 3-D methods cannot use the orbit of "
            f"{conjunction.label(exc.role)}: {exc}"
        ) from exc
    notes = ("relative velocity is zero: 2-D Pc is undefined",)
    return _three_d(conjunction, encounters, notes, {})


def _three_d(
    conjunction: Conjunction,
    encounters: Encounters,
    notes: tuple[str, ...],
    details: dict[str, float],
) -> tuple[PcResult, Assessment]:
    """Hall's 3D-Nc over an encounter scan, or Monte Carlo over it if Hall fails."""
    try:
        return hall3d_over_scan(conjunction, encounters), Assessment(notes, details)
    except PcConvergenceError as exc:
        fallback = (
            f"{exc}; Monte Carlo used instead (seed {AUTO_MC_SEED}, at most "
            f"{AUTO_MC_SAMPLES:,} samples)"
        )
        result = montecarlo_over_scan(
            conjunction, encounters, max_samples=AUTO_MC_SAMPLES, seed=AUTO_MC_SEED
        )
        return result, Assessment((*notes, fallback), details)


def auto_options(method: Method) -> dict[str, object]:
    """The options :func:`auto` runs ``method`` with, where they are not the
    method's defaults: the seed and sample budget of its Monte Carlo
    fallback."""
    if method == "montecarlo":
        return {"max_samples": AUTO_MC_SAMPLES, "seed": AUTO_MC_SEED}
    return {}


def _both_orbits_elliptic(conjunction: Conjunction) -> bool:
    """Whether both objects are on elliptic (bound) orbits, as the 3-D
    methods need."""
    return all(
        orbital_period(item.state) is not None
        for item in (conjunction.primary, conjunction.secondary)
    )


def assess(conjunction: Conjunction) -> Assessment:
    """Applicability notes and encounter duration, when they are meaningful.

    Only with an encounter plane (non-zero relative velocity). The notes are
    kept with a covariance unknown (see :func:`applicability`); the duration
    detail is not, since it is then only a lower bound.
    """
    if not conjunction.has_plane:
        return Assessment((), {})
    duration = encounter_duration(conjunction)
    notes = applicability(conjunction, duration)
    details = (
        {} if conjunction.unknown_covariance else {"encounter_duration_s": duration}
    )
    return Assessment(notes, details)


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
        Its :func:`~keepout.pc.api.encounter_duration` [s].

    Returns
    -------
    tuple of str
        One warning per questionable assumption; empty if none.
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
    along = abs(plane.distance_along_velocity)
    if along > TCA_OFFSET_LIMIT * float(np.linalg.norm(plane.relative_position)):
        warnings.append(
            f"states are {along:.3g} m from closest approach along the relative "
            "velocity; 2-D Pc uses the straight-line closest approach"
        )
    return tuple(warnings)


def _covariances_of(conjunction: Conjunction, roles: tuple[Role, ...]) -> str:
    """``the covariance of <object> is`` or ``the covariances of <object> and
    <object> are``."""
    labels = " and ".join(conjunction.label(role) for role in roles)
    if len(roles) == 2:
        return f"the covariances of {labels} are"
    return f"the covariance of {labels} is"


def known_only_in_position(conjunction: Conjunction, roles: tuple[Role, ...]) -> str:
    """``the covariance of <object> is known only in position (...)``."""
    return (
        f"{_covariances_of(conjunction, roles)} known only in position "
        "(zero velocity block)"
    )
