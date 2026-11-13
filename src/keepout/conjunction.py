"""A conjunction: two objects at the time of closest approach and a combined
hard-body radius."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from keepout._validation import positive_float
from keepout.encounter import EncounterPlane, Role, encounter_plane
from keepout.errors import HardBodyRadiusError
from keepout.objects import ObjectState, require_same_epoch
from keepout.state import RelativeMotion, relative_motion

if TYPE_CHECKING:
    from astropy.time import Time

    from keepout._types import FloatArray

__all__ = ["Conjunction", "combined_hard_body_radius"]


def combined_hard_body_radius(
    primary: ObjectState, secondary: ObjectState, explicit: float | None = None
) -> float:
    """The combined hard-body radius for a conjunction.

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
    primary_radius = primary.hard_body_radius
    secondary_radius = secondary.hard_body_radius
    if primary_radius is None or secondary_radius is None:
        missing = [
            f"the {role} ({item.object_id})"
            for role, item in (("primary", primary), ("secondary", secondary))
            if item.hard_body_radius is None
        ]
        raise HardBodyRadiusError(
            f"no hard-body radius for {' and '.join(missing)}",
            "pass combined_hard_body_radius= (in metres) or give both objects one",
        )
    return primary_radius + secondary_radius


@dataclass(frozen=True, eq=False)
class Conjunction:
    """Two objects at the time of closest approach.

    Parameters
    ----------
    primary
        The protected object.
    secondary
        The other object, at the same epoch.
    combined_hard_body_radius
        Combined hard-body radius [m]: the radius of a sphere around the
        primary that the secondary's centre must enter for a collision.

    Raises
    ------
    GeometryError
        If the epochs differ.
    ValueError
        If ``combined_hard_body_radius`` is not finite and positive.
    """

    primary: ObjectState
    secondary: ObjectState
    combined_hard_body_radius: float

    def __post_init__(self) -> None:
        radius = positive_float(
            self.combined_hard_body_radius, "combined_hard_body_radius"
        )
        object.__setattr__(self, "combined_hard_body_radius", radius)
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
    def relative_motion(self) -> RelativeMotion:
        """The secondary's motion relative to the primary, in the states'
        frame."""
        return relative_motion(self.primary.state, self.secondary.state)

    @property
    def relative_position(self) -> FloatArray:
        """Secondary minus primary position [m], in the states' frame."""
        return self.relative_motion.position

    @property
    def relative_velocity(self) -> FloatArray:
        """Secondary minus primary velocity [m/s], in the states' frame.

        Defined at zero relative speed too, unlike :attr:`plane`.
        """
        return self.relative_motion.velocity

    @property
    def relative_speed(self) -> float:
        """Relative speed [m/s]."""
        return self.relative_motion.speed

    @property
    def has_plane(self) -> bool:
        """Whether the encounter :attr:`plane` is defined: the relative speed
        is not zero."""
        return self.relative_speed != 0.0

    def object_of(self, role: Role) -> ObjectState:
        """The object with ``role``.

        Parameters
        ----------
        role
            ``"primary"`` or ``"secondary"``.

        Returns
        -------
        ObjectState
            The primary or the secondary.
        """
        return self.primary if role == "primary" else self.secondary

    def label(self, role: Role) -> str:
        """``"primary (12345)"``: the role and object id, for messages.

        Parameters
        ----------
        role
            ``"primary"`` or ``"secondary"``.

        Returns
        -------
        str
            The role and object id, e.g. ``"primary (12345)"``.
        """
        return f"{role} ({self.object_of(role).object_id})"

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
