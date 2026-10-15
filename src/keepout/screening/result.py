"""What screening finds: close approaches, and the objects it did not screen.

A :class:`CloseApproach` is one local minimum of the distance between a
primary and a catalog object inside the primary's screening volume: both
objects' GCRF states at the time of closest approach (TCA), from which the
miss distance, the miss vector in the primary's RTN frame and the relative
velocity follow. Screening on GP data finds geometry, not risk: GP element
sets have no covariance, so :meth:`CloseApproach.to_conjunction` takes the
covariances from elsewhere to build a :class:`~keepout.Conjunction` for a
probability of collision.

A :class:`ScreeningResult` holds the approaches of a screening, in
chronological order, and every element set it did not screen
(:class:`SkippedObject`), each with its :class:`SkipReason`.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import Final, Self

import astropy.units as u
import numpy as np
from astropy.time import Time

from keepout._types import FloatArray, read_only
from keepout.conjunction import Conjunction
from keepout.covariance import Covariance
from keepout.errors import GeometryError
from keepout.frames import relative_position_rtn
from keepout.objects import ObjectState, require_same_epoch
from keepout.omm import Omm
from keepout.screening.volume import ScreeningVolume
from keepout.state import StateVector

__all__ = [
    "CloseApproach",
    "ScreeningResult",
    "ScreeningRole",
    "SkipReason",
    "SkippedObject",
]


class ScreeningRole(enum.StrEnum):
    """Which input of a screening an element set came from; primaries sort
    first."""

    PRIMARY = "primary"
    """Given as a primary."""
    CATALOG = "catalog"
    """Given in the catalog."""


_ROLES: Final = tuple(ScreeningRole)


@dataclass(frozen=True, eq=False)
class CloseApproach:
    """A local minimum of the range of a primary and a catalog object.

    The two states are at the time of closest approach, in GCRF; everything
    else is derived from them.

    Parameters
    ----------
    primary_number
        Catalog number of the primary.
    secondary_number
        Catalog number of the catalog object.
    primary_state
        The primary's GCRF state at TCA.
    secondary_state
        The catalog object's GCRF state at the same epoch.

    Raises
    ------
    GeometryError
        If a state is not in GCRF, or the two epochs differ by more than
        :data:`keepout.objects.EPOCH_TOLERANCE_S`.
    """

    primary_number: int
    secondary_number: int
    primary_state: StateVector
    secondary_state: StateVector

    def __post_init__(self) -> None:
        for state in (self.primary_state, self.secondary_state):
            if state.frame != "GCRF":
                raise GeometryError(
                    f"a close approach needs GCRF states, got {state.frame}"
                )
        require_same_epoch(
            self.secondary_state.epoch,
            self.primary_state.epoch,
            "primary and secondary",
        )

    @cached_property
    def tca(self) -> Time:
        """Time of closest approach (the primary state's epoch), in UTC."""
        tca: Time = self.primary_state.epoch.utc
        return tca

    @property
    def relative_position(self) -> FloatArray:
        """Secondary minus primary position [m], in GCRF."""
        position: FloatArray = self.secondary_state.r - self.primary_state.r
        return position

    @property
    def miss_distance(self) -> float:
        """Distance between the objects at TCA [m]."""
        return float(np.linalg.norm(self.relative_position))

    @cached_property
    def miss_rtn(self) -> FloatArray:
        """The secondary's position relative to the primary at TCA, in the
        primary's RTN frame [m] (radial, transverse, normal); read-only."""
        primary = self.primary_state
        miss = relative_position_rtn(primary.r, primary.v, self.secondary_state.r)
        read_only(miss)
        return miss

    @property
    def relative_velocity(self) -> FloatArray:
        """Secondary minus primary velocity [m/s], in GCRF."""
        velocity: FloatArray = self.secondary_state.v - self.primary_state.v
        return velocity

    @property
    def relative_speed(self) -> float:
        """Relative speed at TCA [m/s]."""
        return float(np.linalg.norm(self.relative_velocity))

    def to_conjunction(
        self,
        *,
        hard_body_radius: float,
        primary_covariance: Covariance | None = None,
        secondary_covariance: Covariance | None = None,
    ) -> Conjunction:
        """The approach as a conjunction, with covariances from elsewhere.

        Parameters
        ----------
        hard_body_radius
            Combined hard-body radius of the two objects [m].
        primary_covariance, secondary_covariance
            Each object's covariance at TCA, in GCRF or in its own RTN frame
            (its RTN axes are taken from its GCRF state here); ``None`` where
            unknown (only the maximum-Pc methods accept that).

        Returns
        -------
        Conjunction
            The two objects at TCA, identified by their catalog numbers
            (as text), with GCRF covariances.

        Raises
        ------
        GeometryError
            If a covariance's epoch differs from the TCA by more than
            :data:`keepout.objects.EPOCH_TOLERANCE_S`.
        ValueError
            If ``hard_body_radius`` is not finite and positive.
        """
        objects = []
        for number, state, covariance in (
            (self.primary_number, self.primary_state, primary_covariance),
            (self.secondary_number, self.secondary_state, secondary_covariance),
        ):
            gcrf = None if covariance is None else covariance.in_gcrf(state)
            objects.append(ObjectState(str(number), state, gcrf))
        return Conjunction(objects[0], objects[1], hard_body_radius)


class SkipReason(enum.StrEnum):
    """Why screening left an element set out."""

    DUPLICATE = "duplicate"
    """Another element set of the same object, with a newer epoch, was used."""
    STALE = "stale"
    """The window ends too long after the element epoch."""
    FUTURE_EPOCH = "future epoch"
    """The window starts too long before the element epoch."""
    UNSUPPORTED = "unsupported"
    """SGP4 cannot be initialised with the elements."""
    NO_VOLUME = "no volume"
    """The volume rule gives no volume for this primary."""
    PROPAGATION_FAILED = "propagation failed"
    """SGP4 reported an error inside the window."""


_REASONS: Final = tuple(SkipReason)


@dataclass(frozen=True, slots=True)
class SkippedObject:
    """An element set screening left out, and why.

    Parameters
    ----------
    elements
        The element set.
    role
        Whether it was given as a primary or in the catalog.
    reason
        Why it was left out.
    detail
        The particulars, for people: e.g. the SGP4 error and when it
        occurred.
    """

    elements: Omm
    role: ScreeningRole
    reason: SkipReason
    detail: str

    @property
    def catalog_number(self) -> int:
        """Catalog number of the object."""
        return self.elements.catalog_number


def _chronological(approach: CloseApproach) -> tuple[float, float, int, int]:
    tca = approach.tca
    return (
        float(tca.jd1),
        float(tca.jd2),
        approach.primary_number,
        approach.secondary_number,
    )


def _skip_order(skipped: SkippedObject) -> tuple[int, int, float, float, int]:
    epoch = skipped.elements.epoch
    return (
        _ROLES.index(skipped.role),
        skipped.catalog_number,
        float(epoch.jd1),
        float(epoch.jd2),
        _REASONS.index(skipped.reason),
    )


@dataclass(frozen=True, eq=False, slots=True)
class ScreeningResult:
    """The close approaches of a screening and the objects it left out.

    Parameters
    ----------
    approaches
        Every local minimum of a pair's range inside the primary's volume
        whose TCA lies in the window; kept in chronological order (then by
        primary and secondary catalog number).
    skipped
        Every element set not screened, with the reason; kept sorted by
        role (primaries first), catalog number, epoch and reason.
    start
        Start of the window.
    duration
        Length of the window [s]; TCAs lie in ``(start, start + duration]``.
    volumes
        The volume each screened primary was screened with, by catalog
        number, in the order the primaries were given (a read-only mapping).
    catalog_count
        Number of catalog element sets screened (not skipped).
    """

    approaches: tuple[CloseApproach, ...]
    skipped: tuple[SkippedObject, ...]
    start: Time
    duration: float
    volumes: Mapping[int, ScreeningVolume]
    catalog_count: int

    def __post_init__(self) -> None:
        approaches = tuple(sorted(self.approaches, key=_chronological))
        object.__setattr__(self, "approaches", approaches)
        object.__setattr__(
            self, "skipped", tuple(sorted(self.skipped, key=_skip_order))
        )
        object.__setattr__(self, "volumes", MappingProxyType(dict(self.volumes)))

    def __reduce__(self) -> tuple[type[Self], tuple[object, ...]]:
        # A mapping proxy does not pickle: pickle the plain mapping.
        return type(self), (
            self.approaches,
            self.skipped,
            self.start,
            self.duration,
            dict(self.volumes),
            self.catalog_count,
        )

    @property
    def end(self) -> Time:
        """End of the window."""
        end: Time = self.start + self.duration * u.s
        return end
