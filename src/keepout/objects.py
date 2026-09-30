"""Objects in a conjunction: inertial state, covariance and hard-body radius."""

from __future__ import annotations

import math
from dataclasses import dataclass

from astropy.time import Time

from keepout.covariance import Covariance
from keepout.errors import GeometryError
from keepout.state import StateVector

EPOCH_TOLERANCE_S = 1e-3
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
        radius = self.hard_body_radius
        if radius is not None and not (math.isfinite(radius) and radius >= 0.0):
            raise ValueError(f"hard_body_radius must be finite and >= 0, got {radius}")
