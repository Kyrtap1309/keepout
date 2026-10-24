"""The peak-overlap expansion centres of Hall's 3D-Nc rate (Hall 2021, sec.
VII.B): both objects' Gaussians at the states where their position densities
overlap most, found by iteration at each time."""

from __future__ import annotations

from typing import Final, NamedTuple

import numpy as np

from keepout._linalg import from_eigen, take
from keepout._types import BoolArray, FloatArray, IndexArray
from keepout.errors import GeometryError
from keepout.pc._orbit import OrbitUncertainty

__all__ = [
    "OVERLAP_MAX_ITERATIONS",
    "OVERLAP_TOLERANCE",
    "RATE_ERRORS",
    "Gaussians",
    "peak_overlap",
]

OVERLAP_TOLERANCE: Final = 1e-6
"""Convergence of the peak-overlap iteration: squared Mahalanobis step of
the peak position (Hall 2021, sec. VII.B step 7)."""

OVERLAP_MAX_ITERATIONS: Final = 100

RATE_ERRORS: Final = (np.linalg.LinAlgError, ArithmeticError, ValueError)
"""Failures of the rate computation itself."""

_ITERATION_ERRORS: Final = (*RATE_ERRORS, GeometryError)
"""Failures of the peak-overlap iteration that fall back to the mean-centred
expansion instead of stopping the computation."""


class Gaussians(NamedTuple):
    """Means and covariances of both objects' states, at one time or a batch.

    Batched: ``x1, x2`` ``(m, 6)`` and ``p1, p2`` ``(m, 6, 6)``.
    """

    x1: FloatArray
    p1: FloatArray
    x2: FloatArray
    p2: FloatArray

    def take(self, index: int | slice | IndexArray | BoolArray) -> Gaussians:
        """The times ``index`` of a batch."""
        return take(self, index)

    @property
    def relative_position(self) -> FloatArray:
        """Mean relative positions ``(m, 3)`` [m]."""
        result: FloatArray = self.x2[:, :3] - self.x1[:, :3]
        return result

    @property
    def relative_velocity(self) -> FloatArray:
        """Mean relative velocities ``(m, 3)`` [m/s]."""
        result: FloatArray = self.x2[:, 3:] - self.x1[:, 3:]
        return result

    @property
    def position_covariance(self) -> FloatArray:
        """Combined position covariances ``(m, 3, 3)`` [m^2]."""
        result: FloatArray = self.p1[:, :3, :3] + self.p2[:, :3, :3]
        return result


def peak_overlap(
    primary: OrbitUncertainty,
    secondary: OrbitUncertainty,
    times: FloatArray,
    floor: float,
    mean_centred: Gaussians,
) -> tuple[Gaussians, BoolArray]:
    """Expansion-centre Gaussians of both objects at ``times`` (Hall 2021,
    sec. VII.B), and where the iteration converged to finite values.

    ``floor`` is the smallest position variance used when inverting;
    ``mean_centred`` (the batched expansion about the mean elements) is the
    first iterate and is returned where the iteration failed. If one time
    makes the batch fail (a diverging iterate can leave the space of elliptic
    orbits), the times are iterated one by one so the others still converge.
    """
    try:
        gaussians, converged = _iterate_overlap(
            primary, secondary, times, floor, mean_centred
        )
    except _ITERATION_ERRORS:
        rows = [
            _single_overlap(primary, secondary, times, floor, mean_centred, k)
            for k in range(times.size)
        ]
        gaussians = Gaussians(
            *(np.stack(parts) for parts in zip(*(row[0] for row in rows), strict=True))
        )
        converged = np.array([row[1] for row in rows], dtype=bool)
    finite = np.all(
        [np.isfinite(a).reshape(times.size, -1).all(axis=1) for a in gaussians], axis=0
    )
    return gaussians, converged & finite


def _single_overlap(
    primary: OrbitUncertainty,
    secondary: OrbitUncertainty,
    times: FloatArray,
    floor: float,
    mean_centred: Gaussians,
    k: int,
) -> tuple[Gaussians, bool]:
    one = mean_centred.take(slice(k, k + 1))
    try:
        gaussians, converged = _iterate_overlap(
            primary, secondary, times[k : k + 1], floor, one
        )
    except _ITERATION_ERRORS:
        return one.take(0), False
    return gaussians.take(0), bool(converged[0])


def _iterate_overlap(
    primary: OrbitUncertainty,
    secondary: OrbitUncertainty,
    times: FloatArray,
    floor: float,
    mean_centred: Gaussians,
) -> tuple[Gaussians, BoolArray]:
    """The peak-overlap iteration on a batch of times; may raise."""
    result = Gaussians(*(a.copy() for a in mean_centred))
    converged = np.zeros(times.size, dtype=bool)
    rows = np.arange(times.size)
    current = mean_centred
    previous: FloatArray | None = None
    for _ in range(OVERLAP_MAX_ITERATIONS):
        x1, p1, x2, p2 = current
        inverse1 = _floored_inverse(p1[:, :3, :3], floor)
        inverse2 = _floored_inverse(p2[:, :3, :3], floor)
        precision = inverse1 + inverse2
        rhs = _apply(inverse1, x1[:, :3]) + _apply(inverse2, x2[:, :3])
        # numpy < 2.5 types the solution as any floating dtype; it is float64.
        peak = np.linalg.solve(precision, rhs[..., None])[..., 0].astype(
            np.float64, copy=False
        )
        if previous is not None:
            step = peak - previous
            distance = np.einsum("mi,mi->m", _apply(precision, step), step)
            done = distance <= OVERLAP_TOLERANCE
            for target, source in zip(result, current, strict=True):
                target[rows[done]] = source[done]
            converged[rows[done]] = True
            keep = ~done
            rows, peak, current = rows[keep], peak[keep], current.take(keep)
            inverse1, inverse2 = inverse1[keep], inverse2[keep]
            if rows.size == 0:
                break
        previous = peak
        at = times[rows]
        centre1 = primary.centre_at(
            _conditioned_state(current.x1, current.p1, inverse1, peak), at
        )
        centre2 = secondary.centre_at(
            _conditioned_state(current.x2, current.p2, inverse2, peak), at
        )
        current = Gaussians(
            *primary.expansions(centre1, at), *secondary.expansions(centre2, at)
        )
    return result, converged


def _floored_inverse(matrix: FloatArray, floor: float) -> FloatArray:
    """Inverse of symmetric matrices ``(..., n, n)`` with eigenvalues raised to
    ``floor``.

    Built from the eigen-decomposition, so a tiny floored eigenvalue next to
    a huge one is not lost to rounding as it would be by inverting the
    reconstructed matrix.
    """
    values, vectors = np.linalg.eigh(matrix)
    return from_eigen(1.0 / np.maximum(values, floor), vectors)


def _apply(matrix: FloatArray, vector: FloatArray) -> FloatArray:
    """Batched matrix-vector products ``(m, i, j) x (m, j) -> (m, i)``."""
    result: FloatArray = (matrix @ vector[..., None])[..., 0]
    return result


def _conditioned_state(
    x: FloatArray, p: FloatArray, inverse: FloatArray, peak: FloatArray
) -> FloatArray:
    """States ``(m, 6)`` at the peak position, velocity conditioned on it.

    ``[peak, v + P_vr A^-1 (peak - r)]`` with the object's mean ``(r, v)``,
    covariance ``P`` and floored inverse position covariance ``A^-1``.
    """
    velocity = x[:, 3:] + _apply(p[:, 3:, :3], _apply(inverse, peak - x[:, :3]))
    return np.concatenate([peak, velocity], axis=1)
