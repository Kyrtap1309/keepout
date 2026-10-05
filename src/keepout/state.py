"""State vectors: position and velocity at an epoch, in an explicit frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, get_args

import numpy.typing as npt
from astropy.time import Time

from keepout._types import FloatArray, as_vector
from keepout.errors import UnsupportedFrame

__all__ = ["REF_FRAMES", "RefFrame", "StateVector", "as_ref_frame"]

RefFrame: TypeAlias = Literal["GCRF", "EME2000", "ITRF"]
"""Reference frames a state vector can be expressed in."""

REF_FRAMES: Final[tuple[RefFrame, ...]] = get_args(RefFrame)


def as_ref_frame(name: str | None) -> RefFrame:
    """``name`` as a supported reference frame.

    Parameters
    ----------
    name
        Frame name, e.g. a CDM ``REF_FRAME`` value; ``None`` if absent.

    Raises
    ------
    UnsupportedFrame
        If ``name`` is not one of :data:`REF_FRAMES`.
    """
    for frame in REF_FRAMES:
        if name == frame:
            return frame
    raise UnsupportedFrame(
        f"unsupported reference frame {name!r}; expected one of {', '.join(REF_FRAMES)}"
    )


@dataclass(frozen=True, eq=False, init=False)
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
