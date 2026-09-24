"""Shared domain types: array aliases, reference frames and state vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from astropy.time import Time

FloatArray: TypeAlias = npt.NDArray[np.float64]
"""A float64 NumPy array."""

RefFrame: TypeAlias = Literal["GCRF", "EME2000", "ITRF"]
"""Reference frames a state vector can be expressed in."""

REF_FRAMES: tuple[RefFrame, ...] = ("GCRF", "EME2000", "ITRF")


def as_vector(value: npt.ArrayLike, name: str) -> FloatArray:
    """Return a read-only, finite float64 copy of a 3-vector.

    Parameters
    ----------
    value
        Anything NumPy can turn into a shape ``(3,)`` array.
    name
        Name used in error messages.

    Raises
    ------
    ValueError
        If the shape is not ``(3,)`` or a component is not finite.
    """
    array = np.array(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.flags.writeable = False
    return array


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
