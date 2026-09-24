"""Objects in a conjunction: inertial state, covariance and hard-body radius."""

from __future__ import annotations

import math
from dataclasses import dataclass

from keepout._types import StateVector
from keepout.covariance import Covariance
from keepout.errors import GeometryError

EPOCH_TOLERANCE_S = 1e-3
"""Largest epoch difference [s] treated as "the same epoch"."""


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
        GCRF 6x6 covariance at the same epoch.
    hard_body_radius
        Radius of the sphere enclosing the object [m], if known.

    Raises
    ------
    GeometryError
        If the state or covariance is not GCRF, or their epochs differ.
    ValueError
        If ``hard_body_radius`` is negative or not finite.
    """

    object_id: str
    state: StateVector
    covariance: Covariance
    hard_body_radius: float | None = None

    def __post_init__(self) -> None:
        if self.state.frame != "GCRF" or self.covariance.frame != "GCRF":
            raise GeometryError(
                "ObjectState needs a GCRF state and covariance; convert with "
                "keepout.frames.to_gcrf and Covariance.in_gcrf"
            )
        offset = abs(float((self.state.epoch - self.covariance.epoch).to_value("s")))
        if offset > EPOCH_TOLERANCE_S:
            raise GeometryError(f"state and covariance epochs differ by {offset:.3g} s")
        radius = self.hard_body_radius
        if radius is not None and not (math.isfinite(radius) and radius >= 0.0):
            raise ValueError(f"hard_body_radius must be finite and >= 0, got {radius}")
