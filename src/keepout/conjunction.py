"""A conjunction: two objects at the time of closest approach and a combined
hard-body radius."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from astropy.time import Time

from keepout._types import require_positive
from keepout.covariance import Policy
from keepout.encounter import EncounterPlane, Role, encounter_plane
from keepout.objects import ObjectState, require_same_epoch

if TYPE_CHECKING:
    from keepout.cdm.model import Cdm


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
    ValueError
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
        raise ValueError(
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
        return encounter_plane(self.primary, self.secondary)

    @classmethod
    def from_cdm(
        cls,
        message: Cdm,
        *,
        hard_body_radius: float | None = None,
        policy: Policy = "repair",
    ) -> Conjunction:
        """Build a conjunction from a CDM (``OBJECT1`` is the primary).

        Parameters
        ----------
        message
            The CDM.
        hard_body_radius
            Combined radius [m]; if ``None``, the sum of both objects' ``HBR``
            (CDM 2.0).
        policy
            Covariance policy; see :func:`keepout.covariance.make_covariance`.

        Raises
        ------
        ValueError
            If no radius is given and an object has no ``HBR``.
        """
        primary, secondary = message.object_states(policy=policy)
        radius = combined_hard_body_radius(primary, secondary, hard_body_radius)
        return cls(primary, secondary, radius)

    @property
    def tca(self) -> Time:
        """Time of closest approach (the primary's epoch)."""
        return self.primary.state.epoch

    @property
    def miss_distance(self) -> float:
        """Miss distance on the encounter plane [m]."""
        return self.plane.miss_distance

    @property
    def relative_speed(self) -> float:
        """Relative speed [m/s]."""
        return float(np.linalg.norm(self.secondary.state.v - self.primary.state.v))

    @property
    def unknown_covariance(self) -> tuple[Role, ...]:
        """Objects whose covariance is unknown."""
        roles: tuple[tuple[Role, ObjectState], ...] = (
            ("primary", self.primary),
            ("secondary", self.secondary),
        )
        return tuple(role for role, item in roles if item.covariance is None)

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """Covariance repairs and unknown covariances, labelled with role and id.

        Every Pc result for this conjunction repeats them.
        """
        found: list[str] = []
        for role, item in (("primary", self.primary), ("secondary", self.secondary)):
            if item.covariance is None:
                found.append(f"{role} ({item.object_id}): covariance unknown")
            else:
                found += [
                    f"{role} ({item.object_id}): {repair}"
                    for repair in item.covariance.repairs
                ]
        return tuple(found)
