"""The effect of an along-track burn at TCA, as a difference of two orbits.

:func:`burn_effects` integrates the whole grid of burns in one batch, beside a
reference without a burn. The solver's steps are shared, so what it gets wrong
about the orbit is the same with and without the burn, and leaves the
difference.

A burn whose orbit falls below the Earth's surface, or that the solver fails
on, leaves the others: the batch is split until the failing Δv are alone, the
others are integrated again without them, and each burn of a failing Δv alone.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt

from keepout._linalg import as_axis
from keepout._types import BoolArray, FloatArray, read_only
from keepout._validation import require_positive
from keepout.errors import PropagationError
from keepout.propagation.cowell import integrate
from keepout.state import StateVector, require_frame

if TYPE_CHECKING:
    from keepout.propagation.forces import ForceModel

__all__ = ["BurnEffects", "burn_effects", "closest_approach_shift"]

_MAX_NEWTON_ITERATIONS: Final = 8
"""Most Newton steps of :func:`closest_approach_shift`."""

_NEWTON_TOLERANCE: Final = 1e-6
"""Newton step [s] at which a shift has converged, 15 mm at orbital speed; the
range at a minimum changes with its square."""


@dataclass(frozen=True, eq=False, slots=True)
class BurnEffects:
    """The change of an object's state at TCA, for each burn of a grid.

    The arrays are read-only copies of those given.

    Parameters
    ----------
    burn_leads
        Times of the burns before TCA [s], shape ``(B,)``, descending.
    delta_v
        Signed Δv along the velocity [m/s], shape ``(D,)``.
    dr
        Change of the position at TCA [m], shape ``(B, D, 3)``; NaN where the
        burn failed.
    dv
        Change of the velocity at TCA [m/s], shape ``(B, D, 3)``; NaN where
        the burn failed.

    Raises
    ------
    ValueError
        If ``dr`` or ``dv`` does not have the shape ``(B, D, 3)``.
    """

    burn_leads: FloatArray
    delta_v: FloatArray
    dr: FloatArray
    dv: FloatArray

    def __post_init__(self) -> None:
        leads = np.array(self.burn_leads, dtype=np.float64)
        changes = np.array(self.delta_v, dtype=np.float64)
        dr = np.array(self.dr, dtype=np.float64)
        dv = np.array(self.dv, dtype=np.float64)
        shape = (*leads.shape, *changes.shape, 3)
        if dr.shape != shape or dv.shape != shape:
            raise ValueError(
                f"dr and dv must have shape {shape}, got {dr.shape} and {dv.shape}"
            )
        read_only(leads, changes, dr, dv)
        object.__setattr__(self, "burn_leads", leads)
        object.__setattr__(self, "delta_v", changes)
        object.__setattr__(self, "dr", dr)
        object.__setattr__(self, "dv", dv)

    @property
    def failed(self) -> BoolArray:
        """The cells whose burn failed, shape ``(B, D)``: its orbit fell below
        the Earth's surface, or the solver failed on it."""
        # A ufunc's reduce, not .any(): the oldest NumPy stubs type .any(axis=...)
        # as a scalar or an array.
        failed: BoolArray = np.logical_or.reduce(
            np.isnan(self.dr) | np.isnan(self.dv), axis=-1
        )
        read_only(failed)
        return failed


def burn_effects(
    state: StateVector,
    burn_leads: npt.ArrayLike,
    delta_v: npt.ArrayLike,
    force: ForceModel,
) -> BurnEffects:
    """The change of a state at TCA caused by each burn of a grid.

    The state goes back to the earliest burn, then forward to TCA in one
    batch, one leg per burn time, each adding that burn time's cells. An
    impulse adds ``delta_v`` along the velocity of the orbit without a burn.
    Each change is a cell minus that orbit, carried in the same batch, and a
    cell starts from it, rather than from the state on the way back: the two
    differ by the integration error, cm to dm, which would then be left in
    the difference. A Δv of 0 is not integrated, as equal states do not stay
    bit for bit equal in a batch, and gives exactly 0.

    Where the batch fails, it is split in two halves of its Δv, each
    integrated again from the earliest burn, until a Δv that fails is alone.
    The other Δv are then integrated again in one batch, and so usually come
    out as ``burn_effects`` gives them without the failing ones; should that
    batch fail too, they keep the values of the halves they passed in, whose
    steps depended on those smaller batches. Each burn of a failing Δv is
    integrated alone, as for a grid of that one cell, and is NaN only if it
    fails then. Nothing more is integrated when no burn fails.

    The orbit without a burn thus ends cm to dm from ``state``, which the
    changes do not include. With drag the grid's steps cannot cancel all the
    error of an orbit crossing a band of the density table, and the steps
    depend on the rest of the batch (see :func:`~keepout.propagation.cowell.integrate`).

    Parameters
    ----------
    state
        The object at TCA, in GCRF.
    burn_leads
        Times of the burns before TCA [s], shape ``(B,)``, descending and
        >= 0.
    delta_v
        Signed Δv along the velocity [m/s], shape ``(D,)``; positive is
        prograde.
    force
        The forces on the object.

    Returns
    -------
    BurnEffects
        The changes at the epoch of ``state``, shape ``(B, D, 3)``; NaN in
        the cells that failed.

    Raises
    ------
    GeometryError
        If ``state`` is not in GCRF.
    ValueError
        If ``burn_leads`` or ``delta_v`` is empty, not one-dimensional or not
        finite, or ``burn_leads`` is not descending and >= 0.
    PropagationError
        If the solver fails on the orbit without a burn, or that orbit falls
        below the Earth's surface.
    """
    require_frame(state, "GCRF", "a burn")
    leads = as_axis(burn_leads, "burn_leads")
    changes = as_axis(delta_v, "delta_v")
    if np.any(np.diff(leads) >= 0.0) or leads[-1] < 0.0:
        raise ValueError(f"burn_leads must be descending and >= 0, got {leads}")
    burning = changes != 0.0
    at_tca = np.concatenate((state.r, state.v))[None]
    start = integrate(force, at_tca, -leads[:1])[0, 0]
    difference = np.zeros((len(leads), len(changes), 6))
    difference[:, burning] = _integrate_burns(
        force, at_tca, start, leads, changes[burning]
    )
    return BurnEffects(leads, changes, difference[..., :3], difference[..., 3:])


def _integrate_burns(
    force: ForceModel,
    at_tca: FloatArray,
    start: FloatArray,
    leads: FloatArray,
    burns: FloatArray,
) -> FloatArray:
    """The changes at TCA of the burns, NaN for each burn that fails.

    One batch of all of ``burns``; where it fails, the failing Δv found by
    :func:`_halve`, the others in one batch again, and each burn of a failing
    Δv alone.

    Parameters
    ----------
    force
        The forces on the object.
    at_tca
        The state at TCA, shape ``(1, 6)``.
    start
        The state without a burn at the earliest burn, shape ``(6,)``.
    leads
        Times of the burns before TCA [s], shape ``(B,)``, descending.
    burns
        Signed Δv along the velocity [m/s], shape ``(D,)``, without 0.

    Returns
    -------
    FloatArray
        The changes of position and velocity at TCA, shape ``(B, D, 6)``.

    Raises
    ------
    PropagationError
        If the solver fails on the orbit without a burn, or it falls below
        the Earth's surface.
    """
    try:
        return _burns_at_tca(force, start, leads, burns)
    except PropagationError:
        if len(burns) == 0:
            raise
    changes, failing = _halve(force, start, leads, burns)
    survivors = ~failing
    if failing.any() and survivors.any():
        # Should each half pass and their union not, the halves are kept.
        with contextlib.suppress(PropagationError):
            changes[:, survivors] = _burns_at_tca(force, start, leads, burns[survivors])
    if failing.any():
        starts = [
            integrate(force, at_tca, -leads[[row]])[0, 0] for row in range(len(leads))
        ]
        for column in np.flatnonzero(failing):
            for row, alone in enumerate(starts):
                try:
                    changes[row, column] = _burns_at_tca(
                        force, alone, leads[[row]], burns[[column]]
                    )[0, 0]
                except PropagationError:
                    changes[row, column] = np.nan
    return changes


def _halve(
    force: ForceModel, start: FloatArray, leads: FloatArray, burns: FloatArray
) -> tuple[FloatArray, BoolArray]:
    """The changes at TCA of burns that fail in one batch, from each half of
    them in turn, and the Δv that fail alone.

    Parameters are those of :func:`_integrate_burns`; ``burns`` is not empty.

    Returns
    -------
    tuple[FloatArray, BoolArray]
        The changes of position and velocity at TCA, shape ``(B, D, 6)``,
        NaN for a failing Δv; and whether each Δv fails, shape ``(D,)``.
    """
    if len(burns) == 1:
        return np.full((len(leads), 1, 6), np.nan), np.ones(1, dtype=np.bool_)
    half = len(burns) // 2
    changes: list[FloatArray] = []
    failing: list[BoolArray] = []
    for part in (burns[:half], burns[half:]):
        try:
            part_changes = _burns_at_tca(force, start, leads, part)
        except PropagationError:
            part_changes, part_failing = _halve(force, start, leads, part)
        else:
            part_failing = np.zeros(len(part), dtype=np.bool_)
        changes.append(part_changes)
        failing.append(part_failing)
    return np.concatenate(changes, axis=1), np.concatenate(failing)


def _burns_at_tca(
    force: ForceModel, start: FloatArray, leads: FloatArray, burns: FloatArray
) -> FloatArray:
    """The changes at TCA of the burns, in one batch with the orbit without a
    burn: one leg per burn time, each adding that burn time's cells.

    Parameters are those of :func:`_integrate_burns`.

    Returns
    -------
    FloatArray
        The changes of position and velocity at TCA, shape ``(B, D, 6)``.

    Raises
    ------
    PropagationError
        If the solver fails, or a state falls below the Earth's surface.
    """
    batch = start[None]
    for gap in -np.diff(leads, append=0.0):
        reference = batch[0]
        along_track = reference[3:] / np.linalg.norm(reference[3:])
        cells = np.tile(reference, (len(burns), 1))
        cells[:, 3:] += burns[:, None] * along_track
        batch = integrate(force, np.vstack((batch, cells)), gap[None])[0]
    changes: FloatArray = (batch[1:] - batch[0]).reshape(len(leads), -1, 6)
    return changes


def closest_approach_shift(
    r: FloatArray, v: FloatArray, a: FloatArray, *, limit: float
) -> FloatArray:
    """Time of the nearest minimum of the range along a parabola, if any.

    Newton from time 0 on half the slope of the squared range, ``p · dp/dt``
    of ``p = r + v t + a t²/2``, until it converges, for at most 8 iterations.
    In a slow, nearly co-orbital encounter the curvature ``|dp/dt|² + p · a``
    is near 0 or negative, Newton jumps far along a parabola that no longer
    holds, or to a maximum: such a cell gets NaN, never a time that is not a
    minimum. A finite result is a minimum of the expansion within ``limit``.

    Parameters
    ----------
    r
        Relative position [m], shape ``(..., 3)``.
    v
        Relative velocity [m/s], shape ``(..., 3)``.
    a
        Relative acceleration [m/s²], shape ``(..., 3)``.
    limit
        How far from time 0 the expansion is trusted [s], finite and > 0.

    Returns
    -------
    FloatArray
        The time [s], in the broadcast shape of the inputs without the last
        axis; NaN where the curvature at an iterate is not positive, an
        iterate is further than ``limit`` or Newton has not converged.

    Raises
    ------
    ValueError
        If ``limit`` is not finite and > 0, the last axis of an input is not
        of length 3, or the shapes do not broadcast.
    """
    require_positive(limit, "limit")
    if any(vector.shape[-1:] != (3,) for vector in (r, v, a)):
        raise ValueError(
            f"r, v and a must have shape (..., 3), got {r.shape}, {v.shape}, {a.shape}"
        )
    shape = np.broadcast_shapes(r.shape, v.shape, a.shape)[:-1]
    time = np.zeros(shape)
    step = np.full(shape, np.nan)
    for _ in range(_MAX_NEWTON_ITERATIONS):
        t = time[..., None]
        rate = v + a * t
        position = r + (v + 0.5 * a * t) * t
        slope = np.sum(position * rate, axis=-1)
        curvature = np.sum(rate * rate + position * a, axis=-1)
        step = np.divide(
            -slope, curvature, out=np.full(shape, np.nan), where=curvature > 0.0
        )
        moved = time + step
        # NaN stays NaN from here on: no later iterate can bring a cell back.
        time = np.where(np.abs(moved) <= limit, moved, np.nan)
        if not np.any(np.abs(step) > _NEWTON_TOLERANCE):
            break
    converged: FloatArray = np.where(np.abs(step) <= _NEWTON_TOLERANCE, time, np.nan)
    return converged
