"""A conjunction: two objects at the time of closest approach and a combined
hard-body radius."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from astropy.time import Time

from keepout._types import FloatArray, require_positive
from keepout.encounter import EncounterPlane, Role, encounter_plane
from keepout.errors import HardBodyRadiusError
from keepout.objects import ObjectState, require_same_epoch

__all__ = ["Conjunction", "combined_hard_body_radius"]


def combined_hard_body_radius(
    primary: ObjectState, secondary: ObjectState, explicit: float | None = None
) -> float:
    """Pick the combined hard-body radius for a conjunction.

    Parameters
    ----------
    primary, secondary
        The two objects.
    explicit
        Combined radius given by the caller [m]; wins when not ``None``.

    Returns
    -------
    float
        ``explicit`` if given, else the sum of both objects' radii.

    Raises
    ------
    HardBodyRadiusError
        If ``explicit`` is ``None`` and an object has no radius. keepout never
        falls back to a silent default.
    """
    if explicit is not None:
        return explicit
    first, second = primary.hard_body_radius, secondary.hard_body_radius
    if first is None or second is None:
        missing = [
            o.object_id for o in (primary, secondary) if o.hard_body_radius is None
        ]
        raise HardBodyRadiusError(
            f"no hard-body radius for {', '.join(missing)}: pass "
            "hard_body_radius= (combined, in metres) or give HBR for both objects"
        )
    return first + second


@dataclass(frozen=True, eq=False)
class Conjunction:
    """Two objects at the time of closest approach.

    Parameters
    ----------
    primary
        The protected object.
    secondary
        The other object, at the same epoch.
    hard_body_radius
        Combined hard-body radius [m]: the radius of a sphere around the
        primary that the secondary's centre must enter for a collision.

    Raises
    ------
    GeometryError
        If the epochs differ.
    ValueError
        If ``hard_body_radius`` is not finite and positive.
    """

    primary: ObjectState
    secondary: ObjectState
    hard_body_radius: float

    def __post_init__(self) -> None:
        radius = float(self.hard_body_radius)
        require_positive(radius, "hard_body_radius")
        object.__setattr__(self, "hard_body_radius", radius)
        require_same_epoch(
            self.secondary.state.epoch,
            self.primary.state.epoch,
            "primary and secondary",
        )

    @cached_property
    def plane(self) -> EncounterPlane:
        """The encounter plane (built on first use).

        Raises
        ------
        GeometryError
            If the relative velocity is zero: the plane, and 2-D Pc, are then
            undefined (3-D and Monte Carlo Pc are not).
        """
        return encounter_plane(self)

    @property
    def tca(self) -> Time:
        """Time of closest approach (the primary's epoch)."""
        return self.primary.state.epoch

    @property
    def miss_distance(self) -> float:
        """Miss distance on the encounter plane [m]."""
        return self.plane.miss_distance

    @property
    def objects(self) -> tuple[tuple[Role, ObjectState], tuple[Role, ObjectState]]:
        """Both objects with their role, primary first."""
        return ("primary", self.primary), ("secondary", self.secondary)

    @property
    def relative_position(self) -> FloatArray:
        """Secondary minus primary position [m], in the states' frame."""
        position: FloatArray = self.secondary.state.r - self.primary.state.r
        return position

    @property
    def relative_velocity(self) -> FloatArray:
        """Secondary minus primary velocity [m/s], in the states' frame.

        Defined at zero relative speed too, unlike :attr:`plane`.
        """
        velocity: FloatArray = self.secondary.state.v - self.primary.state.v
        return velocity

    @property
    def relative_speed(self) -> float:
        """Relative speed [m/s]."""
        return float(np.linalg.norm(self.relative_velocity))

    def label(self, role: Role) -> str:
        """``"primary (12345)"``: the role and object id, for messages."""
        item = self.primary if role == "primary" else self.secondary
        return f"{role} ({item.object_id})"

    @property
    def unknown_covariance(self) -> tuple[Role, ...]:
        """Objects whose covariance is unknown."""
        return tuple(role for role, item in self.objects if item.covariance is None)

    @property
    def position_only_covariance(self) -> tuple[Role, ...]:
        """Objects whose covariance is known only in position (a zero velocity
        block, :attr:`~keepout.covariance.Covariance.position_only`)."""
        return tuple(
            role
            for role, item in self.objects
            if item.covariance is not None and item.covariance.position_only
        )

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """Covariance repairs and unknown covariances, labelled with role and id.

        Every Pc result for this conjunction repeats them.
        """
        found: list[str] = []
        for role, item in self.objects:
            label = self.label(role)
            if item.covariance is None:
                found.append(f"{label}: covariance unknown")
            else:
                found += [f"{label}: {repair}" for repair in item.covariance.repairs]
        return tuple(found)
