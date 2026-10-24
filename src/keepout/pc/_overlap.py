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

    Batched: the means ``(m, 6)`` and the covariances ``(m, 6, 6)``.
    """

    primary_mean: FloatArray
    primary_covariance: FloatArray
    secondary_mean: FloatArray
    secondary_covariance: FloatArray

    def take(self, index: int | slice | IndexArray | BoolArray) -> Gaussians:
        """The times ``index`` of a batch."""
        return take(self, index)

    @property
    def relative_position(self) -> FloatArray:
        """Mean relative positions ``(m, 3)`` [m]."""
        result: FloatArray = self.secondary_mean[:, :3] - self.primary_mean[:, :3]
        return result

    @property
    def relative_velocity(self) -> FloatArray:
        """Mean relative velocities ``(m, 3)`` [m/s]."""
        result: FloatArray = self.secondary_mean[:, 3:] - self.primary_mean[:, 3:]
        return result

    @property
    def combined_position_covariance(self) -> FloatArray:
        """Combined position covariances ``(m, 3, 3)`` [m^2]."""
        result: FloatArray = (
            self.primary_covariance[:, :3, :3] + self.secondary_covariance[:, :3, :3]
        )
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
        primary_mean, primary_covariance, secondary_mean, secondary_covariance = current
        primary_inverse = _floored_inverse(primary_covariance[:, :3, :3], floor)
        secondary_inverse = _floored_inverse(secondary_covariance[:, :3, :3], floor)
        precision = primary_inverse + secondary_inverse
        rhs = _apply(primary_inverse, primary_mean[:, :3]) + _apply(
            secondary_inverse, secondary_mean[:, :3]
        )
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
            primary_inverse = primary_inverse[keep]
            secondary_inverse = secondary_inverse[keep]
            if rows.size == 0:
                break
        previous = peak
        at = times[rows]
        primary_centre = primary.centre_at(
            _conditioned_state(
                current.primary_mean, current.primary_covariance, primary_inverse, peak
            ),
            at,
        )
        secondary_centre = secondary.centre_at(
            _conditioned_state(
                current.secondary_mean,
                current.secondary_covariance,
                secondary_inverse,
                peak,
            ),
            at,
        )
        current = Gaussians(
            *primary.expansions(primary_centre, at),
            *secondary.expansions(secondary_centre, at),
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
