"""The minima of a pair in the grid intervals that passed the sieves of
:class:`~keepout.screening.tca.ApproachSearch`: ``f = rho . drho/dt`` from
positions, its roots, and the failures found on the way (see
:mod:`keepout.screening.tca`)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, NamedTuple

import numpy as np

from keepout._types import FloatArray, IndexArray
from keepout.propagation import NO_ERROR, ErrorArray, States
from keepout.screening._sieve import dot
from keepout.screening.search_result import ObjectFailure, RangeMinimum, keep_earliest
from keepout.screening.volume import ScreeningVolume

__all__ = [
    "DERIVATIVE_STEP",
    "SLOW_STEP",
    "TCA_TOLERANCE",
    "Failed",
    "Failures",
    "Interval",
    "Pair",
    "interval_samples",
    "minima_inside_volume",
    "record_first_failures",
    "require_no_failure",
    "roots",
]

SLOW_STEP: Final = 10.0
"""Longest time between the samples of an interval whose minima the grid
cannot decide (a slow pair) [s]."""

DERIVATIVE_STEP: Final = 1.0
"""Step of the central difference that gives ``rho . drho/dt`` [s]: four
positions at -2, -1, +1 and +2 steps. On SGP4 the error is below 1e-4
m^2/s, mostly SGP4's own rounding."""

TCA_TOLERANCE: Final = 1e-6
"""Absolute tolerance of the TCA [s] given to Brent's method."""

_CENTRAL_OFFSETS: Final = np.array([-2.0, -1.0, 1.0, 2.0]) * DERIVATIVE_STEP
"""Offsets [s] of the four positions of the central difference that gives
``f``, from the time it is computed at."""

_CENTRAL_WEIGHTS: Final = np.array([1.0, -8.0, 8.0, -1.0]) / (24.0 * DERIVATIVE_STEP)
"""Weights of ``|rho|^2`` at :data:`_CENTRAL_OFFSETS`: its fourth-order
central difference, halved, ``f`` [m^2/s]."""

_ONE_SIDED_WEIGHTS: Final = np.array([-25.0, 48.0, -36.0, 16.0, -3.0]) / 24.0
"""Fourth-order one-sided difference of ``|rho|^2`` over positions 0 to 4
steps away, halved, per unit step: ``f`` at a window's ends when the
central difference reaches a failed state outside the window."""


class Failed(Exception):
    """Propagation failed, in the window, while refining a pair."""

    def __init__(
        self, primary: ObjectFailure | None, secondary: ObjectFailure | None
    ) -> None:
        super().__init__()
        self.primary = primary
        self.secondary = secondary


def _first_failure(errors: ErrorArray, stamps: FloatArray) -> ObjectFailure | None:
    """The earliest failed state among ``errors`` at ``stamps``, or None when
    none failed."""
    bad = np.flatnonzero(errors != NO_ERROR)
    if bad.size == 0:
        return None
    first = bad[np.argmin(stamps[bad])]
    return ObjectFailure(float(stamps[first]), int(errors[first]))


def require_no_failure(
    primary_errors: ErrorArray, secondary_errors: ErrorArray, stamps: FloatArray
) -> None:
    """A check that neither object of a pair failed at any of ``stamps``.

    Raises
    ------
    Failed
        If ``primary_errors`` or ``secondary_errors`` holds an SGP4 error
        code; it carries the earliest failure of each.
    """
    failed = (
        _first_failure(primary_errors, stamps),
        _first_failure(secondary_errors, stamps),
    )
    if failed != (None, None):
        raise Failed(*failed)


@dataclass(frozen=True, slots=True)
class Failures:
    """The earliest failure seen of each primary and catalog object, by
    position."""

    primary: dict[int, ObjectFailure] = field(default_factory=dict)
    catalog: dict[int, ObjectFailure] = field(default_factory=dict)

    def note(self, primary: int, secondary: int, failed: Failed) -> None:
        """A pair's failure found while refining it, kept as each failed
        object's earliest."""
        if failed.primary is not None:
            keep_earliest(self.primary, primary, failed.primary)
        if failed.secondary is not None:
            keep_earliest(self.catalog, secondary, failed.secondary)


def record_first_failures(
    failures: dict[int, ObjectFailure],
    positions: IndexArray,
    errors: ErrorArray,
    times: FloatArray,
) -> None:
    """The first failed state of each object of a chunk (in time order),
    kept as its earliest failure."""
    failed = errors != NO_ERROR
    for row in np.flatnonzero(np.any(failed, axis=1)):
        first = int(np.argmax(failed[row]))
        failure = ObjectFailure(float(times[first]), int(errors[row, first]))
        keep_earliest(failures, int(positions[row]), failure)


def _stencil_sum(rho: FloatArray, weights: FloatArray) -> FloatArray:
    """``sum_k weights[k] |rho_k|^2`` over each run of ``len(weights)``
    consecutive relative positions, added term by term in order: the same
    bits whatever the number of runs (a matrix product might not)."""
    squared = dot(rho, rho).reshape(-1, len(weights))
    total: FloatArray = squared[:, 0] * weights[0]
    for k in range(1, len(weights)):
        total = total + squared[:, k] * weights[k]
    return total


@dataclass(eq=False, slots=True)
class Pair:
    """One primary and one catalog object, evaluated at any times.

    ``sample`` gives the states of both (rows 0 and 1) at offsets from the
    start of the window, which is ``duration`` long.
    """

    sample: States
    duration: float
    _known_f: dict[float, float] = field(default_factory=dict, init=False)
    """``f`` at each time it was computed at."""

    def states(
        self, stamps: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Positions and velocities of both at ``stamps``, ``(n, 3)`` each."""
        errors, r, v = self.sample(stamps)
        if errors.any():
            require_no_failure(errors[0], errors[1], stamps)
        return r[0], v[0], r[1], v[1]

    def f_values(self, times: FloatArray) -> FloatArray:
        """``f`` at ``times``, from the values already known if all are,
        else computed (and kept)."""
        listed = times.tolist()
        if all(time in self._known_f for time in listed):
            return np.array([self._known_f[time] for time in listed])
        values = self._computed_f(times)
        self._known_f.update(zip(listed, values.tolist(), strict=True))
        return values

    def f_at(self, time: float) -> float:
        """``f`` at one time, for Brent's method (which starts at the ends
        of its bracket, already known)."""
        known = self._known_f.get(time)
        if known is not None:
            return known
        return float(self._computed_f(np.array([time]))[0])

    def _computed_f(self, times: FloatArray) -> FloatArray:
        """``f = rho . drho/dt`` at ``times`` (in the window) from positions
        only, whether known or not.

        Raises :class:`Failed` for a failure in the window only: a time
        whose central difference reaches a failed state outside the window
        takes the one-sided difference inside it.
        """
        stamps = (times[:, None] + _CENTRAL_OFFSETS).reshape(-1)
        errors, r, _ = self.sample(stamps)
        if not errors.any():
            return _stencil_sum(r[1] - r[0], _CENTRAL_WEIGHTS)
        outside = (stamps < 0.0) | (stamps > self.duration)
        inside = ~outside
        require_no_failure(errors[0][inside], errors[1][inside], stamps[inside])
        derivative = _stencil_sum(r[1] - r[0], _CENTRAL_WEIGHTS)
        failed = (errors[0] != NO_ERROR) | (errors[1] != NO_ERROR)
        reaching = (failed & outside).reshape(-1, len(_CENTRAL_OFFSETS))
        edge = np.flatnonzero(np.any(reaching, axis=1))
        if edge.size > 0:
            derivative[edge] = self._one_sided_f(times[edge])
        return derivative

    def _one_sided_f(self, times: FloatArray) -> FloatArray:
        """``f`` at ``times`` from five positions inside the window, towards
        its far end (closer together than DERIVATIVE_STEP in a window
        shorter than four of them)."""
        after = self.duration - times
        forward = after >= times
        room = np.where(forward, after, times)
        spacing = np.minimum(DERIVATIVE_STEP, room / 4.0) * np.where(forward, 1.0, -1.0)
        nodes = times[:, None] + spacing[:, None] * np.arange(len(_ONE_SIDED_WEIGHTS))
        stamps = np.clip(nodes, 0.0, self.duration).reshape(-1)
        r_primary, _, r_secondary, _ = self.states(stamps)
        derivative: FloatArray = (
            _stencil_sum(r_secondary - r_primary, _ONE_SIDED_WEIGHTS) / spacing
        )
        return derivative


class Interval(NamedTuple):
    """A grid interval of a pair that passed the sieves."""

    first: float
    last: float
    monotone: bool
    """Whether ``f`` is monotone over it: its sign at the ends decides."""


def interval_samples(interval: Interval) -> FloatArray:
    """Where ``f`` is evaluated in an interval: its ends, or every SLOW_STEP."""
    first, last = interval.first, interval.last
    if interval.monotone:
        return np.array([first, last])
    count = math.ceil((last - first) / SLOW_STEP)
    samples = first + (last - first) * (np.arange(count + 1) / count)
    samples[-1] = last
    return samples


def roots(pair: Pair, samples: FloatArray) -> tuple[list[float], Failed | None]:
    """The TCAs of a pair in one interval, sampled at ``samples``, in order,
    and the failure that stopped the search, if one did."""
    # Imported here: scipy.optimize takes a noticeable time to import.
    from scipy.optimize import brentq

    try:
        values = pair.f_values(samples)
    except Failed as exc:
        return [], exc
    tcas: list[float] = []
    for j in np.flatnonzero((values[:-1] < 0.0) & (values[1:] >= 0.0)):
        try:
            tca = brentq(pair.f_at, samples[j], samples[j + 1], xtol=TCA_TOLERANCE)
        except Failed as exc:
            return tcas, exc
        tcas.append(float(tca))
    return tcas, None


def minima_inside_volume(
    volume: ScreeningVolume,
    primary: int,
    secondary: int,
    tcas: FloatArray,
    errors: ErrorArray,
    states: tuple[FloatArray, ...],
    failed: Failed | None,
) -> list[RangeMinimum]:
    """The minima of a pair in one interval: those of its TCAs (states at
    each given) at which the other object is inside the primary's
    ``volume``.

    Raises, as if each TCA's state were propagated after its root was
    found, :class:`Failed` for the first TCA whose state failed, else the
    failure that stopped the search for roots (``failed``).
    """
    bad = np.flatnonzero(np.any(errors != NO_ERROR, axis=0))
    if bad.size > 0:
        rows = slice(int(bad[0]), int(bad[0]) + 1)
        require_no_failure(errors[0, rows], errors[1, rows], tcas[rows])
    if failed is not None:
        raise failed
    r_primary, v_primary, r_secondary, v_secondary = states
    inside = volume.contains_position(r_primary, v_primary, r_secondary)
    rho = r_secondary - r_primary
    distance = np.sqrt(dot(rho, rho))
    return [
        RangeMinimum(
            primary,
            secondary,
            float(tcas[k]),
            float(distance[k]),
            r_primary[k],
            v_primary[k],
            r_secondary[k],
            v_secondary[k],
        )
        for k in np.flatnonzero(inside)
    ]
