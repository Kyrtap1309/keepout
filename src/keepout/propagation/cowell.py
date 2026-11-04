"""Cowell integration, in one batch or behind the
:class:`~keepout.propagation.base.Propagator` interface.

:func:`integrate` carries N states through one ``solve_ivp`` (DOP853), so they
share their steps: the fast way to integrate many states of one force model.
:class:`CowellPropagator` integrates each object alone, once, and keeps a
spline, because a batch's steps depend on the other objects and an
integration's steps on where it ends, and the interface wants a state that
depends on its object and its time only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Final, Literal, NamedTuple, Self

import numpy as np
import numpy.typing as npt
from astropy.time import Time

from keepout._linalg import as_axis
from keepout._types import FloatArray, override
from keepout._validation import (
    require_non_negative,
    require_positive,
    require_single_time,
)
from keepout.errors import PropagationError
from keepout.propagation.base import (
    DECAYED,
    Ephemeris,
    RadiusBounds,
    object_indices,
    time_offsets,
)
from keepout.propagation.forces import EARTH_RADIUS, ForceModel
from keepout.units import elapsed_seconds

if TYPE_CHECKING:
    from scipy.interpolate import CubicHermiteSpline

__all__ = ["SPAN_SLACK", "CowellPropagator", "integrate"]

SPAN_SLACK: Final = 1e-6
"""How far past an end of the span a time may lie [s]: the difference of two
epochs rounds, so a time at the end can land just outside it."""


def _radii(state: FloatArray) -> FloatArray:
    """Distance from the Earth's centre of each object of a flat state."""
    radii: FloatArray = np.linalg.norm(state.reshape(-1, 6)[:, :3], axis=1)
    return radii


class _Surface:
    """The ``solve_ivp`` event of the lowest object going through the surface."""

    terminal: ClassVar[bool] = True
    direction: ClassVar[float] = -1.0

    def __call__(self, _time: float, state: FloatArray) -> float:
        return float(_radii(state).min() - EARTH_RADIUS)


def _states_array(states: npt.ArrayLike) -> FloatArray:
    """States as an array of shape ``(N, 6)``.

    Raises
    ------
    ValueError
        If the shape is another, or a number is not finite.
    """
    array = np.asarray(states, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 6:
        raise ValueError(f"states must have shape (N, 6), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("states must be finite")
    return array


def _stops_array(stops: npt.ArrayLike) -> FloatArray:
    """Stops as a strictly monotonic array of shape ``(T,)``, T >= 1.

    Raises
    ------
    ValueError
        If they are empty, not one-dimensional, not finite or not strictly
        monotonic.
    """
    array = as_axis(stops, "stops")
    gaps = np.diff(array)
    if not (np.all(gaps > 0.0) or np.all(gaps < 0.0)):
        raise ValueError("stops must be strictly monotonic")
    return array


def integrate(
    force: ForceModel,
    states: FloatArray,
    stops: FloatArray,
    *,
    rtol: float = 1e-11,
    atol_position: float = 1e-4,
    atol_velocity: float = 1e-7,
    stop_at_surface: bool = False,
) -> FloatArray:
    """States of N objects at some times, in one DOP853 integration.

    The integration goes from time 0 to the first stop, whichever way that
    is, then along the stops. All the objects share their steps, and the
    solver's error norm is a root mean square over them, so one difficult
    object in a large batch is held to a looser tolerance than alone. The
    tolerances may not bound the error of drag on an orbit that crosses bands
    of the density table (see "The atmosphere" in ``docs/maneuver.md``).

    Parameters
    ----------
    force
        The forces on every object.
    states
        Positions [m] and velocities [m/s] at time 0, shape ``(N, 6)``, N >= 1.
    stops
        Times [s], shape ``(T,)``, strictly monotonic; the first may have
        either sign.
    rtol
        Relative tolerance of the solver.
    atol_position
        Absolute tolerance of the positions [m].
    atol_velocity
        Absolute tolerance of the velocities [m/s].
    stop_at_surface
        For one object: a state below :data:`EARTH_RADIUS` ends the
        integration, instead of raising.

    Returns
    -------
    FloatArray
        The states at the stops, shape ``(T, N, 6)``; with ``stop_at_surface``,
        NaN at the stops the integration did not reach.

    Raises
    ------
    ValueError
        If the shapes are wrong, a number is not finite, ``stops`` are not
        strictly monotonic or ``stop_at_surface`` is given more than one
        object.
    PropagationError
        If the solver fails, or a state falls below :data:`EARTH_RADIUS`
        without ``stop_at_surface``.
    """
    # Imported here: scipy.integrate takes 0.2 s to import.
    from scipy.integrate import solve_ivp

    initial = _states_array(states)
    times = _stops_array(stops)
    count = len(initial)
    if count == 0:
        raise ValueError("states must hold at least one object")
    if stop_at_surface and count != 1:
        raise ValueError(f"stop_at_surface takes one object, got {count}")
    tolerance = np.tile(np.repeat((atol_position, atol_velocity), 3), count)
    rows = np.full((len(times), count, 6), np.nan)

    def leg(start: float, state: FloatArray, ends: FloatArray) -> FloatArray:
        """The flat states at ``ends``, as far as the integration got.

        Raises
        ------
        PropagationError
            If the solver fails, or the surface ends the leg unasked.
        """
        solution = solve_ivp(
            lambda _, y: force.derivative(y),
            (start, float(ends[-1])),
            state,
            method="DOP853",
            t_eval=ends,
            events=_Surface(),
            rtol=rtol,
            atol=tolerance,
        )
        if solution.status < 0:
            raise PropagationError(f"integration failed: {solution.message}")
        if solution.status > 0 and not stop_at_surface:
            # Lists, not None: solve_ivp was given events.
            t_events: list[FloatArray] = solution.t_events or []
            y_events: list[FloatArray] = solution.y_events or []
            when = t_events[0][0]
            lowest = np.argmin(_radii(y_events[0][0]))
            raise PropagationError(
                f"object {lowest} fell below the Earth's surface at t = {when:.1f} s"
            )
        # An empty list, not an array, if nothing was reached.
        reached: FloatArray = np.reshape(solution.y, (state.size, -1)).T
        return reached

    flat = initial.ravel()
    below = _radii(flat) < EARTH_RADIUS
    if below.any():
        if stop_at_surface:
            return rows
        raise PropagationError(
            f"object {np.argmax(below)} starts below the Earth's surface"
        )
    reached = flat[None] if times[0] == 0.0 else leg(0.0, flat, times[:1])
    if len(reached) == 1 and len(times) > 1:
        rest = leg(float(times[0]), reached[0], times[1:])
        reached = np.concatenate((reached, rest))
    rows[: len(reached)] = reached.reshape(-1, count, 6)
    return rows


class _Arc(NamedTuple):
    """One object's integrated trajectory."""

    spline: CubicHermiteSpline | None
    """Its state through time; ``None`` if it re-entered before a second node."""
    end: float
    """Where its valid states end (exclusive): ``inf`` if it stays up, else
    the last node before re-entry."""


def _arc(force: ForceModel, state: FloatArray, nodes: FloatArray) -> _Arc:
    """The trajectory of one object, integrated through ``nodes``."""
    # Imported here: only an integration needs it, and it needs scipy anyway.
    from scipy.interpolate import CubicHermiteSpline

    rows = integrate(force, state[None], nodes, stop_at_surface=True)[:, 0]
    count = int(np.count_nonzero(np.isfinite(rows[:, 0])))
    if count < 2:
        return _Arc(None, -math.inf)
    kept = rows[:count]
    slopes = force.derivative(kept.ravel()).reshape(-1, 6)
    spline = CubicHermiteSpline(nodes[:count], kept, slopes)
    return _Arc(spline, math.inf if count == len(nodes) else float(nodes[count - 1]))


class CowellPropagator:
    """States of objects integrated once, over a fixed span of time.

    At construction each object is integrated alone with :func:`integrate` to
    nodes at most ``step`` apart, from the start of the span to its end, and a
    cubic Hermite spline is kept through them, of the positions (the
    velocities being their slopes) and of the velocities (the accelerations
    being theirs). :meth:`propagate` only evaluates the splines, so a state
    depends on its object and its time alone, as
    :class:`~keepout.propagation.base.Propagator` requires; integrating per
    call could not give that, as ``solve_ivp``'s steps depend on where the
    integration ends.

    The cubic's error goes as ``step⁴``. In LEO, at 500 km, it peaks between
    nodes at 0.3 mm and 3e-7 m/s for the default 10 s, 22 mm at 30 s, 0.36 m
    at 60 s; above that, it falls as the mean motion to the fourth.

    A span that does not start at ``epoch`` is reached by integrating to its
    start first, so an object that re-enters on the way has no valid state.
    An object that re-enters has error code
    :data:`~keepout.propagation.base.DECAYED` and NaN states from the last
    node before re-entry on (the spline ends there). A pickled propagator
    holds its splines, about 1.7 MB per object for a day at 10 s, and
    integrates nothing again.

    Parameters
    ----------
    epoch
        The time of ``states`` and the reference of ``span``; one time, in any
        time scale.
    states
        GCRF positions [m] and velocities [m/s] at ``epoch``, shape ``(N, 6)``.
    forces
        The forces on each object, N of them.
    span
        First and last offset [s] from ``epoch`` that :meth:`propagate` can
        give, finite and increasing.
    step
        Longest time between nodes [s], finite and > 0.

    Raises
    ------
    ValueError
        If ``epoch`` is not one time, the shapes do not match, a number is
        not finite or ``span`` or ``step`` is out of range.
    PropagationError
        If the solver fails on an object.
    """

    __slots__ = ("_arcs", "_epoch", "_span")

    _arcs: tuple[_Arc, ...]
    _epoch: Time
    _span: tuple[float, float]

    def __init__(
        self,
        epoch: Time,
        states: FloatArray,
        forces: Sequence[ForceModel],
        span: tuple[float, float],
        *,
        step: float = 10.0,
    ) -> None:
        require_single_time(epoch, "epoch")
        initial = _states_array(states)
        if len(forces) != len(initial):
            raise ValueError(
                f"forces must hold one model per object: {len(forces)} for "
                f"{len(initial)}"
            )
        start, stop = span
        if not (math.isfinite(start) and math.isfinite(stop) and start < stop):
            raise ValueError(f"span must be finite and increasing, got {span}")
        require_positive(step, "step")
        nodes = np.linspace(start, stop, math.ceil((stop - start) / step) + 1)
        self._epoch = epoch
        self._span = (start, stop)
        self._arcs = tuple(
            _arc(force, state, nodes)
            for force, state in zip(forces, initial, strict=True)
        )

    @property
    def frame(self) -> Literal["GCRF"]:
        """Frame of the states: GCRF, as they were given."""
        return "GCRF"

    def __len__(self) -> int:
        return len(self._arcs)

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self)} objects)"

    def subset(self, indices: npt.ArrayLike) -> Self:
        """A propagator of some of the objects, sharing their splines.

        Parameters
        ----------
        indices
            Positions of the objects to keep, in the order wanted; a
            one-dimensional array of integers in ``[0, len(self))``
            (repeats allowed).

        Returns
        -------
        CowellPropagator
            The chosen objects; nothing is integrated again.

        Raises
        ------
        ValueError
            If ``indices`` are not integers in one dimension.
        IndexError
            If an index is outside ``[0, len(self))``.
        """
        chosen = object_indices(indices, len(self))
        subset = object.__new__(type(self))
        subset._epoch = self._epoch
        subset._span = self._span
        subset._arcs = tuple(self._arcs[i] for i in chosen)
        return subset

    def propagate(self, epoch: Time, offsets: npt.ArrayLike) -> Ephemeris:
        """GCRF states of every object at ``epoch + offsets``.

        Parameters
        ----------
        epoch
            Reference epoch; one time, in any time scale.
        offsets
            Seconds elapsed since ``epoch``; a number is one offset.

        Returns
        -------
        Ephemeris
            Positions [m] and velocities [m/s] in GCRF, shape ``(N, T, 3)``,
            and the error codes, shape ``(N, T)``: 6 where an object has
            re-entered.

        Raises
        ------
        ValueError
            If ``epoch`` is not one time, the offsets are not finite numbers
            in at most one dimension, or one is more than 1 microsecond outside the
            span.
        """
        seconds = time_offsets(epoch, offsets)
        times = seconds + elapsed_seconds(self._epoch, epoch)
        first, last = self._span
        if np.any((times < first - SPAN_SLACK) | (times > last + SPAN_SLACK)):
            raise ValueError(f"offsets outside the span [{first}, {last}] s")
        times = np.clip(times, first, last)
        states = np.full((len(self), len(times), 6), np.nan)
        errors = np.zeros((len(self), len(times)), dtype=np.uint8)
        for index, arc in enumerate(self._arcs):
            valid = times < arc.end
            errors[index, ~valid] = DECAYED
            if arc.spline is not None:
                states[index, valid] = arc.spline(times[valid])
        return Ephemeris(
            epoch, seconds, states[..., :3], states[..., 3:], errors, self.frame
        )

    def radius_bounds(self, start: Time, duration: float) -> RadiusBounds:
        """No bounds: ``0`` and ``inf``, as a numerical orbit has no formula.

        Parameters
        ----------
        start
            Start of the span; one time, in any time scale.
        duration
            Length of the span [s], finite and >= 0.

        Returns
        -------
        RadiusBounds
            Lowest ``0`` and highest ``inf`` for every object.

        Raises
        ------
        ValueError
            If ``start`` is not one time or ``duration`` is not finite and
            >= 0.
        """
        require_single_time(start, "start")
        require_non_negative(duration, "duration")
        return RadiusBounds(np.zeros(len(self)), np.full(len(self), np.inf))
