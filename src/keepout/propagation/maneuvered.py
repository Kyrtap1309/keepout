"""An orbit after a burn: a propagator's states plus a Cowell difference.

:class:`ManeuveredPropagator` keeps what its base propagator says of an object
and adds what a burn changes: the difference of two Cowell orbits from the
base's state at the burn, one with the burn and one without. Each state stays
in the base's theory (SGP4 for an element set) and screening needs nothing
else to change.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Self

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.time import Time

from keepout._types import FloatArray, override
from keepout._validation import (
    require_non_negative,
    require_positive,
    require_single_time,
)
from keepout.errors import PropagationError
from keepout.frames import frame_to_gcrf
from keepout.propagation.base import (
    NO_ERROR,
    Ephemeris,
    Propagator,
    RadiusBounds,
    describe_error,
    time_offsets,
)
from keepout.propagation.cowell import SPAN_SLACK, integrate
from keepout.propagation.forces import ForceModel, earth_pole
from keepout.state import RefFrame
from keepout.units import elapsed_seconds

if TYPE_CHECKING:
    from scipy.interpolate import CubicHermiteSpline

__all__ = ["ManeuveredPropagator"]


def _gcrf_to_frame(frame: RefFrame, epochs: Time) -> FloatArray:
    """The rotations from GCRF to ``frame`` at ``epochs``, shape ``(T, 3, 3)``,
    read off unit vectors."""
    images, _ = frame_to_gcrf(frame, epochs.reshape(-1, 1), np.eye(3), np.zeros((3, 3)))
    return images  # rows: the frame's axes in GCRF


def _burn_difference(
    state: FloatArray,
    frame: RefFrame,
    burn_time: Time,
    delta_v: float,
    nodes: FloatArray,
) -> CubicHermiteSpline:
    """What a burn adds to a position in ``frame``, as a spline over ``nodes``
    (seconds after the burn) from ``state``, the GCRF state at the burn.

    Raises
    ------
    PropagationError
        If the solver fails, or either orbit falls below the Earth's surface.
    """
    # Imported here: only a burn needs it, and it needs scipy anyway.
    from scipy.interpolate import CubicHermiteSpline

    velocity = state[3:]
    burned = state.copy()
    burned[3:] += delta_v * velocity / np.linalg.norm(velocity)
    # No drag: element sets carry no CD·A/m to trust, and a difference over a
    # week changes by far less than a screening volume (Assumptions).
    force = ForceModel(earth_pole(burn_time))
    # Tighter than integrate's defaults, which leave 2 m to 6 m after a week of
    # 1 m/s on an eccentric orbit: sharing steps does not cancel the error there.
    rows = integrate(
        force,
        np.stack((state, burned)),
        nodes,
        rtol=1e-12,
        atol_position=1e-5,
        atol_velocity=1e-8,
    )
    difference = rows[:, 1] - rows[:, 0]
    # A rotation per node: TEME turns by 0.9 arcsec against GCRF in a week
    # (4.4e-6 rad, precession of the equinox), 8 m on the 1800 km of a 1 m/s
    # burn. The rate of that turn is left out of the velocity: 2e-5 m/s there.
    rotations = _gcrf_to_frame(frame, burn_time + nodes * u.s)
    position = (rotations @ difference[:, :3, None])[..., 0]
    speed = (rotations @ difference[:, 3:, None])[..., 0]
    return CubicHermiteSpline(nodes, position, speed)


class ManeuveredPropagator:
    """The states of one object with an along-track burn, in its base's frame.

    Before the burn a state is the base's, exactly. From the burn on it is
    the base's plus a difference: from the base's state at the burn, in GCRF,
    the orbits with and without an impulse along the velocity go through one
    :func:`~keepout.propagation.cowell.integrate` to nodes at most ``step``
    apart, with zonal gravity to degree 6 and no drag, and a cubic Hermite
    spline of the position difference is kept, with the velocity difference
    as its slope. The difference therefore depends on the object and the time
    alone, as :class:`~keepout.propagation.base.Propagator` requires. The
    velocity is the spline's slope, continuous over the nodes. An error of the
    base stays, with its NaN state, and the integration ignores it; the state
    at the burn must have none.

    Against a Cowell integration of the whole orbit after the burn the
    difference is off by metres at 1 m/s over 7 days; the measurements are
    under "Re-screen accuracy" in ``docs/assumptions.md``.

    Times after ``burn_time + horizon`` have no state: :meth:`propagate` and
    :meth:`radius_bounds` refuse them, so that a window past the horizon never
    goes unnoticed. Screening reads positions
    :data:`~keepout.screening.tca.SEARCH_REACH` past the end of a window, so
    ``horizon`` must reach that far; :func:`~keepout.screening.screen`
    refuses one that does not.

    A burn at a time inside the window of a screening is a kink in the
    primary's path, which the search's bound on accelerations does not
    include; up to half a grid step's worth of m/s (60 m/s by default) it is
    within :data:`~keepout.screening.tca.PERTURBATION`. A burn at the start of
    the window is no kink in the grid's states, but a central difference
    reads positions up to 2 s before the window, so ``f`` within 2 s after
    the burn straddles it: positions stay continuous, and the TCA of such a
    minimum moves by about ``2 s * delta_v / relative speed``.

    A pickled propagator holds its base and its spline, about 150 kB per day
    at 60 s; only the base is initialised again.

    Parameters
    ----------
    base
        The object without the burn; one object.
    burn_time
        Time of the impulse; one time, in any time scale.
    delta_v
        Signed Δv along the velocity [m/s]; positive is prograde. Zero
        integrates nothing and gives the base's states exactly.
    horizon
        Seconds after ``burn_time`` the states go up to, finite and > 0. To
        screen a window of ``duration`` from the burn, ``duration +
        SEARCH_REACH``.
    step
        Longest time between nodes [s], finite and > 0.

    Raises
    ------
    ValueError
        If ``base`` does not hold one object, ``burn_time`` is not one time,
        ``delta_v`` is not finite or ``horizon`` or ``step`` is out of range.
    UnsupportedFrame
        If the base's frame is not a supported reference frame.
    PropagationError
        If the base has no state at the burn, the solver fails, or an orbit
        falls below the Earth's surface before the horizon.
    """

    __slots__ = ("_base", "_burn_time", "_difference", "_horizon")

    _base: Propagator
    _burn_time: Time
    _difference: CubicHermiteSpline | None
    _horizon: float

    def __init__(
        self,
        base: Propagator,
        burn_time: Time,
        delta_v: float,
        horizon: float,
        *,
        step: float = 60.0,
    ) -> None:
        if len(base) != 1:
            raise ValueError(f"base must hold one object, got {len(base)}")
        require_single_time(burn_time, "burn_time")
        if not math.isfinite(delta_v):
            raise ValueError(f"delta_v must be finite, got {delta_v}")
        require_positive(horizon, "horizon")
        require_positive(step, "step")
        at_burn = base.propagate(burn_time, 0.0)
        code = int(at_burn.errors[0, 0])
        if code != NO_ERROR:
            raise PropagationError(
                f"no state to burn from at {burn_time.utc.isot}: {describe_error(code)}"
            )
        r, v = frame_to_gcrf(base.frame, burn_time, at_burn.r[0, 0], at_burn.v[0, 0])
        nodes = np.linspace(0.0, horizon, math.ceil(horizon / step) + 1)
        self._base = base
        self._burn_time = burn_time
        self._horizon = float(horizon)
        self._difference = (
            None
            if delta_v == 0.0
            else _burn_difference(
                np.concatenate((r, v)), base.frame, burn_time, delta_v, nodes
            )
        )

    @property
    def frame(self) -> RefFrame:
        """Frame of the states: the base's."""
        return self._base.frame

    def __len__(self) -> int:
        return len(self._base)

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self)} objects)"

    def subset(self, indices: npt.ArrayLike) -> Self:
        """A propagator of some of the objects, sharing the spline.

        Parameters
        ----------
        indices
            Positions of the objects to keep, in the order wanted; a
            one-dimensional array of integers in ``[0, len(self))``
            (repeats allowed, each the same object).

        Returns
        -------
        ManeuveredPropagator
            The chosen objects; nothing is integrated again.

        Raises
        ------
        ValueError
            If ``indices`` are not integers in one dimension.
        IndexError
            If an index is outside ``[0, len(self))``.
        """
        subset = object.__new__(type(self))
        subset._base = self._base.subset(indices)
        subset._burn_time = self._burn_time
        subset._difference = self._difference
        subset._horizon = self._horizon
        return subset

    def propagate(self, epoch: Time, offsets: npt.ArrayLike) -> Ephemeris:
        """States of the object at ``epoch + offsets``, in the base's frame.

        Parameters
        ----------
        epoch
            Reference epoch; one time, in any time scale.
        offsets
            Seconds elapsed since ``epoch``; a number is one offset.

        Returns
        -------
        Ephemeris
            Positions [m] and velocities [m/s], shape ``(N, T, 3)``, and the
            base's error codes, shape ``(N, T)``.

        Raises
        ------
        ValueError
            If ``epoch`` is not one time, the offsets are not finite numbers
            in at most one dimension, or one is more than 1 microsecond after
            the horizon.
        """
        seconds = time_offsets(epoch, offsets)
        times = seconds + elapsed_seconds(self._burn_time, epoch)
        self._require_within_horizon(times)
        states = self._base.propagate(epoch, seconds)
        if self._difference is None:
            return states
        after = times >= 0.0
        burned = np.clip(times[after], None, self._horizon)
        r, v = states.r.copy(), states.v.copy()
        r[:, after] += self._difference(burned)
        v[:, after] += self._difference(burned, nu=1)
        return Ephemeris(epoch, seconds, r, v, states.errors, self.frame)

    def radius_bounds(self, start: Time, duration: float) -> RadiusBounds:
        """The base's bounds on the radius, widened by the burn's difference.

        Parameters
        ----------
        start
            Start of the span; one time, in any time scale.
        duration
            Length of the span [s], finite and >= 0.

        Returns
        -------
        RadiusBounds
            The base's bounds over the span, moved out by the largest
            difference of a position from the burn on, ``lowest`` clipped at
            0. Most of a difference lies along the track and leaves the radius
            alone, so these are wide: kilometres per mm/s after a week.

        Raises
        ------
        ValueError
            If ``start`` is not one time, ``duration`` is not finite and
            >= 0, or the span ends more than 1 microsecond after the horizon.
        """
        require_single_time(start, "start")
        require_non_negative(duration, "duration")
        first = elapsed_seconds(self._burn_time, start)
        self._require_within_horizon(np.array([first + duration]))
        bounds = self._base.radius_bounds(start, duration)
        widening = self._largest_difference(first, first + duration)
        lowest = np.maximum(bounds.lowest - widening, 0.0)
        return RadiusBounds(lowest, bounds.highest + widening)

    def _require_within_horizon(self, times: FloatArray) -> None:
        """A check of seconds after the burn.

        Raises
        ------
        ValueError
            If one is more than 1 microsecond after the horizon.
        """
        if np.any(times > self._horizon + SPAN_SLACK):
            raise ValueError(
                f"times after the horizon, {self._horizon} s after the burn: "
                f"up to {times.max()} s"
            )

    def _largest_difference(self, first: float, last: float) -> float:
        """A bound on the difference over seconds ``first`` to ``last`` after
        the burn, between the nodes too: on a node interval of length ``h`` a
        cubic Hermite spline stays within ``h / 4`` times the larger slope of
        the larger value."""
        difference = self._difference
        if difference is None or last < 0.0:
            return 0.0
        nodes: FloatArray = difference.x
        low = int(np.searchsorted(nodes, first, side="right")) - 1
        low = min(max(low, 0), len(nodes) - 2)
        high = int(np.searchsorted(nodes, last, side="left")) + 1
        kept = nodes[low : max(high, low + 2)]
        values = np.linalg.norm(difference(kept), axis=1)
        slopes = np.linalg.norm(difference(kept, 1), axis=1)
        reach = np.maximum(values[:-1], values[1:])
        reach += np.diff(kept) / 4.0 * np.maximum(slopes[:-1], slopes[1:])
        return float(reach.max())
