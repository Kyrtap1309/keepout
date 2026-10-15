"""The interface every propagator gives: states of many objects at many times.

A :class:`Propagator` holds N objects and returns, for a reference epoch and
an array of time offsets, an :class:`Ephemeris`: positions [m], velocities
[m/s] and an error code for every object at every time, in one inertial
frame. It also bounds each object's distance from the Earth's centre over a
time span (:class:`RadiusBounds`), which screening's apogee/perigee filter
needs before anything is propagated. Screening depends on this interface
only, so that SGP4 (:class:`~keepout.propagation.sgp4_prop.Sgp4Propagator`)
and a numerical propagator can be used alike.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol, Self, TypeAlias, runtime_checkable

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.time import Time

from keepout._types import BoolArray, FloatArray, read_only
from keepout.state import RefFrame, as_ref_frame

__all__ = [
    "CHUNK_STATES",
    "ERROR_MESSAGES",
    "NO_ERROR",
    "Ephemeris",
    "ErrorArray",
    "Propagator",
    "RadiusBounds",
    "describe_error",
    "propagate_in_chunks",
    "time_offsets",
]

ErrorArray: TypeAlias = npt.NDArray[np.uint8]
"""Error codes, one per object and time; see :data:`ERROR_MESSAGES`."""

NO_ERROR: Final = 0
"""Error code of a state computed without error."""

ERROR_MESSAGES: Final[Mapping[int, str]] = MappingProxyType(
    {
        1: "mean eccentricity outside the range 0 to 1",
        2: "mean motion below zero",
        3: "perturbed eccentricity outside the range 0 to 1",
        4: "semi-latus rectum below zero",
        5: "orbit below the Earth's surface (no longer used by SGP4)",
        6: "decayed: orbit radius below one Earth radius",
    }
)
"""What each non-zero error code means. Codes 1 to 6 are SGP4's (Vallado et
al., "Revisiting Spacetrack Report #3", AIAA 2006-6753); other propagators
report an object that has re-entered with code 6."""

CHUNK_STATES: Final = 1_000_000
"""Default number of states (objects x times) per chunk in
:func:`propagate_in_chunks`: about 49 MB of positions, velocities and error
codes, and as fast per state as larger chunks."""


def describe_error(code: int) -> str:
    """Text of an error code, for messages and reports.

    Parameters
    ----------
    code
        An error code of an :class:`Ephemeris`.

    Returns
    -------
    str
        ``"no error"`` for :data:`NO_ERROR`, else ``"error N: meaning"``
        (``"unknown"`` for a code not in :data:`ERROR_MESSAGES`).
    """
    if code == NO_ERROR:
        return "no error"
    return f"error {code}: {ERROR_MESSAGES.get(code, 'unknown')}"


def time_offsets(epoch: Time, offsets: npt.ArrayLike) -> FloatArray:
    """Check a time grid and return its offsets as a read-only array.

    Parameters
    ----------
    epoch
        Reference epoch; one time.
    offsets
        Seconds elapsed since ``epoch`` (SI seconds, as ``epoch + offsets *
        u.s`` counts them, leap seconds included); a number is one offset.

    Returns
    -------
    FloatArray
        The offsets, shape ``(t,)``, as a read-only float64 copy.

    Raises
    ------
    ValueError
        If ``epoch`` is not a single time, or the offsets are not finite
        numbers in at most one dimension.
    """
    if not epoch.isscalar:
        raise ValueError(f"epoch must be a single time, got shape {epoch.shape}")
    seconds = np.array(offsets, dtype=np.float64, ndmin=1)
    if seconds.ndim != 1:
        raise ValueError(f"offsets must be one-dimensional, got shape {seconds.shape}")
    if not np.all(np.isfinite(seconds)):
        raise ValueError("offsets must be finite")
    read_only(seconds)
    return seconds


def _view(array: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    """A read-only view: no copy, and the caller's array stays writeable."""
    view = array.view()
    read_only(view)
    return view


@dataclass(frozen=True, eq=False, slots=True)
class Ephemeris:
    """States of N objects at T times, as a propagator returns them.

    The arrays are read-only views of those given (not copies).

    Parameters
    ----------
    epoch
        Reference epoch of the time grid.
    offsets
        Times as seconds elapsed since ``epoch``, shape ``(T,)``; see
        :func:`time_offsets`.
    r
        Positions [m], shape ``(N, T, 3)``; NaN where ``errors`` is non-zero.
    v
        Velocities [m/s], shape ``(N, T, 3)``; NaN where ``errors`` is
        non-zero.
    errors
        Error code of each state, shape ``(N, T)``: :data:`NO_ERROR` or a
        code of :data:`ERROR_MESSAGES`.
    frame
        Inertial frame of ``r`` and ``v``.

    Raises
    ------
    ValueError
        If the time grid is invalid (see :func:`time_offsets`), the shapes
        do not match, or ``errors`` are not 8-bit unsigned integers.
    UnsupportedFrame
        If ``frame`` is not a supported reference frame.
    """

    epoch: Time
    offsets: FloatArray
    r: FloatArray
    v: FloatArray
    errors: ErrorArray
    frame: RefFrame

    def __post_init__(self) -> None:
        offsets = time_offsets(self.epoch, self.offsets)
        r = np.asarray(self.r, dtype=np.float64)
        v = np.asarray(self.v, dtype=np.float64)
        errors = np.asarray(self.errors)
        if r.ndim != 3 or r.shape[1:] != (len(offsets), 3):
            raise ValueError(
                f"r must have shape (N, {len(offsets)}, 3) for {len(offsets)} "
                f"offsets, got {r.shape}"
            )
        if v.shape != r.shape:
            raise ValueError(f"v must have the shape of r {r.shape}, got {v.shape}")
        if errors.shape != r.shape[:2]:
            raise ValueError(
                f"errors must have shape {r.shape[:2]}, got {errors.shape}"
            )
        if errors.dtype != np.uint8:
            raise ValueError(f"errors must be of type uint8, got {errors.dtype}")
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "r", _view(r))
        object.__setattr__(self, "v", _view(v))
        object.__setattr__(self, "errors", _view(errors))
        object.__setattr__(self, "frame", as_ref_frame(self.frame))

    @property
    def times(self) -> Time:
        """The times of the grid, shape ``(T,)``."""
        times: Time = self.epoch + self.offsets * u.s  # astropy.units is untyped
        return times

    @property
    def valid(self) -> BoolArray:
        """Where a state was computed without error, shape ``(N, T)``."""
        valid: BoolArray = self.errors == NO_ERROR
        return valid


@dataclass(frozen=True, eq=False, slots=True)
class RadiusBounds:
    """Bounds on the distance of N objects from the Earth's centre.

    Over a time span, every state a propagator gives for object ``i`` (at a
    time in the span, without an error) has ``lowest[i] <= |r| <=
    highest[i]``. The arrays are read-only copies of those given.

    Parameters
    ----------
    lowest
        Lower bounds [m], shape ``(N,)``, finite and >= 0.
    highest
        Upper bounds [m], shape ``(N,)``, >= ``lowest``; ``inf`` for an
        object the propagator cannot bound.

    Raises
    ------
    ValueError
        If the arrays are not one-dimensional of the same length, or a bound
        is out of range.
    """

    lowest: FloatArray
    highest: FloatArray

    def __post_init__(self) -> None:
        lowest = np.array(self.lowest, dtype=np.float64)
        highest = np.array(self.highest, dtype=np.float64)
        if lowest.ndim != 1 or highest.shape != lowest.shape:
            raise ValueError(
                "lowest and highest must be one-dimensional of the same length, "
                f"got shapes {lowest.shape} and {highest.shape}"
            )
        if not np.all(np.isfinite(lowest) & (lowest >= 0.0)):
            raise ValueError("lowest must be finite and >= 0")
        if not np.all(highest >= lowest):
            raise ValueError("highest must be >= lowest")
        read_only(lowest, highest)
        object.__setattr__(self, "lowest", lowest)
        object.__setattr__(self, "highest", highest)

    def __len__(self) -> int:
        return len(self.lowest)


@runtime_checkable
class Propagator(Protocol):
    """States of a fixed set of objects at any times.

    Contract, relied on by screening:

    * ``propagate`` returns positions [m] and velocities [m/s] in
      :attr:`frame` for every object (in order) at every offset;
    * a failed state has a non-zero error code and NaN position and
      velocity, and does not affect any other state;
    * each state depends only on its object and its time: a :meth:`subset`
      of the objects, or a part of the offsets, gives exactly the numbers of
      the matching part of a whole call, so chunking a time grid or splitting
      the work between processes cannot change a result;
    * :meth:`radius_bounds` holds every such state's distance from the
      Earth's centre, without propagating (screening filters on it);
    * a propagator can be pickled, to be sent to worker processes.
    """

    @property
    def frame(self) -> RefFrame:
        """Inertial frame of the states."""
        ...

    def __len__(self) -> int:
        """Number of objects."""
        ...

    def subset(self, indices: npt.ArrayLike) -> Self:
        """A propagator of some of the objects, without re-initialising them.

        Parameters
        ----------
        indices
            Positions of the objects to keep, in the order wanted; a
            one-dimensional array of integers in ``[0, len(self))``.
        """
        ...

    def propagate(self, epoch: Time, offsets: npt.ArrayLike) -> Ephemeris:
        """States of every object at ``epoch + offsets``.

        Parameters
        ----------
        epoch
            Reference epoch; one time, in any time scale.
        offsets
            Seconds elapsed since ``epoch``; see :func:`time_offsets`.
        """
        ...

    def radius_bounds(self, start: Time, duration: float) -> RadiusBounds:
        """Bounds on each object's distance from the Earth's centre over a span.

        Every state :meth:`propagate` gives at a time from ``start`` to
        ``start + duration`` (both included) is within its object's bounds,
        unless it has an error.

        Parameters
        ----------
        start
            Start of the span; one time, in any time scale.
        duration
            Length of the span [s], finite and >= 0.
        """
        ...


def propagate_in_chunks(
    propagators: Sequence[Propagator],
    epoch: Time,
    offsets: npt.ArrayLike,
    *,
    max_states: int = CHUNK_STATES,
    overlap: bool = False,
) -> Iterator[tuple[Ephemeris, ...]]:
    """Propagate on a time grid one chunk of consecutive times at a time.

    Memory stays bounded by ``max_states`` whatever the grid, and the chunks
    together hold exactly the states of one call on the whole grid.

    Parameters
    ----------
    propagators
        The objects, in one or more propagators: each chunk holds the states
        of all of them at the same times.
    epoch
        Reference epoch of the grid.
    offsets
        Seconds elapsed since ``epoch``; see :func:`time_offsets`.
    max_states
        Most states (objects of all the propagators x times) in one chunk; a
        chunk always has at least one time (two with ``overlap``).
    overlap
        Whether each chunk after the first starts at the last time of the one
        before (propagated again), so that every two consecutive times of the
        grid lie in one chunk.

    Returns
    -------
    Iterator[tuple[Ephemeris, ...]]
        The states at consecutive parts of ``offsets``, in order, one
        :class:`Ephemeris` per propagator, each chunk computed when the
        iterator reaches it; nothing for an empty grid.

    Raises
    ------
    ValueError
        If ``max_states`` is below 1 or the time grid is invalid (checked at
        the call, before any chunk).
    """
    if max_states < 1:
        raise ValueError(f"max_states must be at least 1, got {max_states}")
    seconds = time_offsets(epoch, offsets)
    objects = sum(len(propagator) for propagator in propagators)
    size = max(2 if overlap else 1, max_states // max(1, objects))
    advance = size - 1 if overlap else size
    # With overlap, the last time starts no chunk: the one before holds it.
    end = len(seconds) - 1 if overlap and len(seconds) > 1 else len(seconds)
    return (
        tuple(
            propagator.propagate(epoch, seconds[start : start + size])
            for propagator in propagators
        )
        for start in range(0, end, advance)
    )
