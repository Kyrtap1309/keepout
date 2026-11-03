"""The effect of an along-track burn at TCA, as a difference of two orbits.

:func:`burn_effects` integrates the whole grid of burns in one batch, beside a
reference without a burn. The solver's steps are shared, so what it gets wrong
about the orbit is the same with and without the burn, and leaves the
difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from keepout._types import FloatArray, read_only
from keepout._validation import require_positive
from keepout.propagation.cowell import integrate
from keepout.propagation.forces import ForceModel
from keepout.state import StateVector, require_frame

__all__ = ["BurnEffects", "burn_effects", "closest_approach_shift"]

_MAX_NEWTON_ITERATIONS: Final = 8

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
        Change of the position at TCA [m], shape ``(B, D, 3)``.
    dv
        Change of the velocity at TCA [m/s], shape ``(B, D, 3)``.

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


def _axis(values: npt.ArrayLike, name: str) -> FloatArray:
    """Finite numbers of a grid axis, shape ``(N,)``, N >= 1.

    Raises
    ------
    ValueError
        If they are empty, not one-dimensional or not finite.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


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
        The changes at the epoch of ``state``, shape ``(B, D, 3)``.

    Raises
    ------
    GeometryError
        If ``state`` is not in GCRF.
    ValueError
        If ``burn_leads`` or ``delta_v`` is empty, not one-dimensional or not
        finite, or ``burn_leads`` is not descending and >= 0.
    PropagationError
        If the solver fails, or a cell falls below the Earth's surface.
    """
    require_frame(state, "GCRF", "a burn")
    leads = _axis(burn_leads, "burn_leads")
    changes = _axis(delta_v, "delta_v")
    if np.any(np.diff(leads) >= 0.0) or leads[-1] < 0.0:
        raise ValueError(f"burn_leads must be descending and >= 0, got {leads}")
    burning = changes != 0.0
    burns = changes[burning]
    at_tca = np.concatenate((state.r, state.v))[None]
    batch = integrate(force, at_tca, -leads[:1])[0]
    for gap in -np.diff(leads, append=0.0):
        reference = batch[0]
        along_track = reference[3:] / np.linalg.norm(reference[3:])
        cells = np.tile(reference, (len(burns), 1))
        cells[:, 3:] += burns[:, None] * along_track
        batch = integrate(force, np.vstack((batch, cells)), gap[None])[0]
    difference = np.zeros((len(leads), len(changes), 6))
    difference[:, burning] = (batch[1:] - batch[0]).reshape(len(leads), -1, 6)
    return BurnEffects(leads, changes, difference[..., :3], difference[..., 3:])


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
