"""State vectors: position and velocity at an epoch, in an explicit frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, get_args

import numpy.typing as npt
from astropy.time import Time

from keepout._types import FloatArray, as_vector

__all__ = ["REF_FRAMES", "RefFrame", "StateVector"]

RefFrame: TypeAlias = Literal["GCRF", "EME2000", "ITRF"]
"""Reference frames a state vector can be expressed in."""

REF_FRAMES: tuple[RefFrame, ...] = get_args(RefFrame)


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
        if frame not in REF_FRAMES:
            raise ValueError(
                f"unsupported frame {frame!r}; expected one of {REF_FRAMES}"
            )
        object.__setattr__(self, "frame", frame)
