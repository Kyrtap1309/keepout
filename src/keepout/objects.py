"""Objects in a conjunction: inertial state, covariance and hard-body radius."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy.typing as npt
from astropy.time import Time

from keepout._types import require_non_negative
from keepout.covariance import Covariance, Policy, make_covariance
from keepout.errors import GeometryError
from keepout.frames import to_gcrf_all
from keepout.state import StateVector

__all__ = [
    "EPOCH_TOLERANCE_S",
    "ObjectState",
    "ReportedObject",
    "gcrf_object_states",
    "require_same_epoch",
]

EPOCH_TOLERANCE_S: Final = 1e-3
"""Largest epoch difference [s] treated as "the same epoch"."""


def require_same_epoch(first: Time, second: Time, what: str) -> None:
    """Raise :class:`GeometryError` if two epochs differ by more than the tolerance.

    Parameters
    ----------
    first, second
        Epochs to compare.
    what
        What is compared, for the error message (e.g. ``"state and covariance"``).
    """
    offset = abs(float((first - second).to_value("s")))
    if offset > EPOCH_TOLERANCE_S:
        raise GeometryError(f"{what} epochs differ by {offset:.3g} s")


@dataclass(frozen=True, eq=False)
class ObjectState:
    """One object of a conjunction, in GCRF.

    Parameters
    ----------
    object_id
        Identifier, e.g. the catalogue number.
    state
        GCRF state at the time of closest approach.
    covariance
        GCRF 6x6 covariance at the same epoch, or ``None`` if unknown (only
        the maximum-Pc methods Frisbee and Alfano accept an object without
        covariance; dilution needs both).
    hard_body_radius
        Radius of the sphere enclosing the object [m], if known.

    Raises
    ------
    GeometryError
        If the state or a given covariance is not GCRF, or their epochs differ.
    ValueError
        If ``hard_body_radius`` is negative or not finite.
    """

    object_id: str
    state: StateVector
    covariance: Covariance | None
    hard_body_radius: float | None = None

    def __post_init__(self) -> None:
        covariance = self.covariance
        if self.state.frame != "GCRF" or (
            covariance is not None and covariance.frame != "GCRF"
        ):
            raise GeometryError(
                "ObjectState needs a GCRF state and covariance; convert with "
                "keepout.frames.to_gcrf and Covariance.in_gcrf"
            )
        if covariance is not None:
            require_same_epoch(
                self.state.epoch, covariance.epoch, "state and covariance"
            )
        if self.hard_body_radius is not None:
            require_non_negative(self.hard_body_radius, "hard_body_radius")


@dataclass(frozen=True, eq=False)
class ReportedObject:
    """One object as a conjunction message reports it.

    The state may be in any supported frame and the covariance is in RTN, as
    in a CDM; :func:`gcrf_object_states` turns it into an :class:`ObjectState`.

    Parameters
    ----------
    object_id
        Identifier, e.g. the catalogue number.
    state
        State at the time of closest approach, in any supported frame.
    covariance_rtn
        6x6 RTN covariance [m, m/s] at the epoch of ``state``, unvalidated;
        ``None`` if unknown.
    hard_body_radius
        Radius of the sphere enclosing the object [m], if known.
    """

    object_id: str
    state: StateVector
    covariance_rtn: npt.ArrayLike | None
    hard_body_radius: float | None = None


def gcrf_object_states(
    objects: Sequence[ReportedObject], *, policy: Policy = "repair"
) -> list[ObjectState]:
    """Objects in GCRF, their RTN axes taken from the GCRF states.

    The states are converted together (:func:`keepout.frames.to_gcrf_all`),
    so several ITRF states cost one astropy transformation.

    Parameters
    ----------
    objects
        The objects as reported.
    policy
        Covariance policy; see :func:`keepout.covariance.make_covariance`.

    Returns
    -------
    list of ObjectState
        In the order of ``objects``.

    Raises
    ------
    CovarianceError
        If ``policy`` is ``"raise"`` and a covariance is invalid.
    """
    states = to_gcrf_all([item.state for item in objects])
    converted: list[ObjectState] = []
    for item, state in zip(objects, states, strict=True):
        covariance = None
        if item.covariance_rtn is not None:
            covariance = make_covariance(
                item.covariance_rtn, "RTN", state.epoch, policy=policy
            ).in_gcrf(state)
        converted.append(
            ObjectState(item.object_id, state, covariance, item.hard_body_radius)
        )
    return converted
