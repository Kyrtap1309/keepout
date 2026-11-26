"""State vectors: position and velocity at an epoch, in an explicit frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, TypeAlias, get_args

import numpy as np
import numpy.typing as npt

from keepout._linalg import as_vector
from keepout._validation import require_choice
from keepout.errors import GeometryError, UnsupportedFrame

if TYPE_CHECKING:
    from astropy.time import Time

    from keepout._types import FloatArray

__all__ = [
    "REF_FRAMES",
    "RefFrame",
    "RelativeMotion",
    "StateVector",
    "as_ref_frame",
    "relative_motion",
    "require_frame",
]

RefFrame: TypeAlias = Literal["GCRF", "EME2000", "ITRF", "TEME"]
"""Reference frames a state vector can be expressed in."""

REF_FRAMES: Final[tuple[RefFrame, ...]] = get_args(RefFrame)
"""Every :data:`RefFrame`: ``("GCRF", "EME2000", "ITRF", "TEME")``."""


def as_ref_frame(name: str | None) -> RefFrame:
    """``name`` as a supported reference frame.

    Parameters
    ----------
    name
        Frame name, e.g. a CDM ``REF_FRAME`` value; ``None`` if absent.

    Returns
    -------
    RefFrame
        The frame.

    Raises
    ------
    UnsupportedFrame
        If ``name`` is not one of :data:`REF_FRAMES`.
    """
    return require_choice(name, REF_FRAMES, "reference frame", UnsupportedFrame)


@dataclass(frozen=True, eq=False, init=False, slots=True)
class StateVector:
    """Position and velocity at an epoch, in an explicit frame (SI units).

    Parameters
    ----------
    epoch
        Epoch of the state.
    r
        Position [m].
    v
        Velocity [m/s].
    frame
        Reference frame of ``r`` and ``v``.

    Raises
    ------
    ValueError
        If ``r`` or ``v`` is not a finite 3-vector.
    UnsupportedFrame
        If ``frame`` is not one of :data:`REF_FRAMES` (also a ``ValueError``).
    """

    epoch: Time
    r: FloatArray
    v: FloatArray
    frame: RefFrame

    # A hand-written __init__ so callers may pass any array-like while the
    # attributes stay typed as read-only float64 arrays.
    def __init__(
        self, epoch: Time, r: npt.ArrayLike, v: npt.ArrayLike, frame: RefFrame
    ) -> None:
        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(self, "r", as_vector(r, "r"))
        object.__setattr__(self, "v", as_vector(v, "v"))
        object.__setattr__(self, "frame", as_ref_frame(frame))


def require_frame(state: StateVector, frame: RefFrame, what: str) -> None:
    """Raise :class:`GeometryError` unless ``state`` is in ``frame``.

    Parameters
    ----------
    state
        The state to check.
    frame
        The frame it must be in.
    what
        What needs the state, for the error: ``"a close approach"``.

    Raises
    ------
    GeometryError
        If ``state`` is in another frame.
    """
    if state.frame != frame:
        hint = "; convert with keepout.frames.to_gcrf" if frame == "GCRF" else ""
        raise GeometryError(f"{what} needs a {frame} state, got {state.frame}{hint}")


class RelativeMotion(NamedTuple):
    """A secondary object's motion relative to a primary, in their frame."""

    position: FloatArray
    """Secondary minus primary position [m]."""
    velocity: FloatArray
    """Secondary minus primary velocity [m/s]."""

    @property
    def distance(self) -> float:
        """Length of :attr:`position` [m]."""
        return float(np.linalg.norm(self.position))

    @property
    def speed(self) -> float:
        """Length of :attr:`velocity` [m/s]."""
        return float(np.linalg.norm(self.velocity))


def relative_motion(primary: StateVector, secondary: StateVector) -> RelativeMotion:
    """The motion of ``secondary`` relative to ``primary`` (same frame).

    Parameters
    ----------
    primary, secondary
        The two states.

    Returns
    -------
    RelativeMotion
        The relative position [m] and velocity [m/s], in the frame of the states.

    Raises
    ------
    GeometryError
        If the states are in different frames.
    """
    require_frame(secondary, primary.frame, "relative_motion")
    position: FloatArray = secondary.r - primary.r
    velocity: FloatArray = secondary.v - primary.v
    return RelativeMotion(position, velocity)
