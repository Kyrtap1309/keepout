"""Hall's 3D-Nc collision rate.

The rate ``Ncdot(t)`` [1/s] is defined in :mod:`keepout.pc.hall3d`; it is
shared by :func:`~keepout.pc.hall3d.hall3d`, which integrates it, and
:func:`~keepout.pc.montecarlo.montecarlo`, which samples trials only where
it is not negligible (:mod:`keepout.pc._encounters` locates them).
"""

from __future__ import annotations

import math
from typing import Final, NamedTuple

import numpy as np
from scipy import special

from keepout._linalg import from_eigen
from keepout._types import FloatArray
from keepout.errors import PcConvergenceError
from keepout.pc._orbit import OrbitUncertainty
from keepout.pc._overlap import RATE_ERRORS, Gaussians, peak_overlap
from keepout.pc._plane import SIGMA_FLOOR, SQRT_2, SQRT_2PI
from keepout.pc._screen import negligible
from keepout.pc._sphere import SphereGrid, sphere_grids

__all__ = ["CollisionRate"]

_SQRT_PI: Final = math.sqrt(math.pi)

_TINY_VARIANCE: Final = 1e-300
"""Floor [m^2/s^2] of the conditional radial-velocity variance, which divides
the mean radial velocity in the flux."""


class CollisionRate:
    """Hall's 3D-Nc collision rate ``Ncdot(t)`` [1/s] of a conjunction.

    Called with an array of times [s from TCA], it returns the rate at each.

    Parameters
    ----------
    primary, secondary
        The two orbits with their uncertainty.
    radius
        Combined hard-body radius [m].
    """

    def __init__(
        self, primary: OrbitUncertainty, secondary: OrbitUncertainty, radius: float
    ) -> None:
        self.primary, self.secondary, self.radius = primary, secondary, radius
        self.floor_variance = (SIGMA_FLOOR * radius) ** 2
        """Smallest position variance [m^2] used when inverting a covariance:
        :data:`~keepout.pc._plane.SIGMA_FLOOR` of the radius, squared (as in
        2-D Pc and CARA's ``Fclip``), so a perfectly known object has a
        finite, tiny extent."""
        self.evaluations = 0
        self.smallest_floored_variance = math.inf
        """Smallest combined position variance [m^2] raised to
        :attr:`floor_variance` by :meth:`_rates` (infinite if the floor never
        engaged)."""
        self.fallbacks: list[tuple[float, float]] = []
        """Times [s] and rates where the peak-overlap iteration failed."""

    @property
    def position_covariance(self) -> FloatArray:
        """Combined Cartesian position covariance at TCA [m^2]."""
        result: FloatArray = (
            self.primary.position_covariance + self.secondary.position_covariance
        )
        return result

    def mean_centred(self, times: FloatArray) -> Gaussians:
        """Both objects' expansions about their mean orbits at ``times``."""
        return Gaussians(
            *self.primary.mean_expansions(times), *self.secondary.mean_expansions(times)
        )

    def __call__(self, t: FloatArray) -> FloatArray:
        times = np.atleast_1d(np.asarray(t, dtype=np.float64))
        self.evaluations += times.size
        gaussians = self.mean_centred(times)
        values = np.zeros(times.size)
        active = np.flatnonzero(~negligible(gaussians, self.radius))
        if active.size == 0:
            return values
        mean_centred = gaussians.take(active)
        converged, ok = peak_overlap(
            self.primary,
            self.secondary,
            times[active],
            self.floor_variance,
            mean_centred,
        )
        source = Gaussians(
            *(
                np.where(ok.reshape(-1, *(1,) * (c.ndim - 1)), c, m)
                for c, m in zip(converged, mean_centred, strict=True)
            )
        )
        values[active] = self._checked_rates(times[active], source)
        self.fallbacks.extend(
            (float(t_k), float(value))
            for t_k, value in zip(times[active[~ok]], values[active[~ok]], strict=True)
        )
        return values

    def _checked_rates(self, times: FloatArray, gaussians: Gaussians) -> FloatArray:
        """:meth:`_rates` at ``times``, with a failure raised as
        :class:`PcConvergenceError`.

        Raises
        ------
        PcConvergenceError
            If the rate at one of ``times`` cannot be computed or is not
            finite (the first such time is named).
        """
        try:
            values = self._rates(gaussians)
        except RATE_ERRORS as error:
            at, cause = float(times[0]), error
            # Name the first time that fails on its own.
            for k, t in enumerate(times.tolist()):
                try:
                    self._rates(gaussians.take(slice(k, k + 1)))
                except RATE_ERRORS as single:
                    at, cause = t, single
                    break
            raise PcConvergenceError(
                f"Hall 3D-Nc: the collision rate at t = {at:.6g} s from TCA "
                f"cannot be computed: {cause}"
            ) from cause
        bad = np.flatnonzero(~np.isfinite(values))
        if bad.size:
            k = int(bad[0])
            raise PcConvergenceError(
                f"Hall 3D-Nc: the collision rate at t = {float(times[k]):.6g} s "
                f"from TCA is not finite ({float(values[k])})"
            )
        return values

    def _rates(self, gaussians: Gaussians) -> FloatArray:
        """The rate ``(m,)`` [1/s] of a batch of expansion-centre Gaussians.

        The linear algebra is batched over the times; the sphere quadrature,
        whose nodes differ between times, is done time by time
        (:func:`_sphere_integral`).

        The arithmetic is that of one time, in Cartesian axes, operation for
        operation: the conditional covariance ``C - P_vr A^-1 P_rv`` can
        cancel most of its digits (by ~6 in some of CARA's Alfano cases), so
        reordering it, e.g. in the principal axes of ``A``, moves the rate
        by up to ~1e-6 relative.
        """
        mean = gaussians.x2 - gaussians.x1
        covariance = gaussians.p1 + gaussians.p2
        values, vectors = np.linalg.eigh(covariance[:, :3, :3])
        smallest = values[:, 0]
        floored_here = smallest < self.floor_variance
        if np.any(floored_here):
            self.smallest_floored_variance = min(
                self.smallest_floored_variance, float(np.min(smallest[floored_here]))
            )
        # The floored position covariance A, used through its eigen-decomposition.
        floored = np.maximum(values, self.floor_variance)
        a_inverse = from_eigen(1.0 / floored, vectors)
        b, c = covariance[:, 3:, :3], covariance[:, 3:, 3:]
        gain = b @ a_inverse
        relative = _Relative(
            mean,
            vectors,
            a_inverse,
            np.sum(np.log(floored), axis=1),
            gain,
            c - gain @ np.swapaxes(b, 1, 2),
        )
        along = np.einsum("mi,mi->m", vectors[:, :, 0], mean[:, :3])
        grids = sphere_grids(floored, along, self.radius)
        rates = [
            _sphere_integral(grid, relative.row(k), self.radius)
            for k, grid in enumerate(grids)
        ]
        return np.array(rates, dtype=np.float64)


class _Relative(NamedTuple):
    """The relative Gaussian at a batch of times, with its floored position
    covariance ``A``.

    Batched: ``mean`` ``(m, 6)``, ``log_det`` ``(m,)``, the others
    ``(m, 3, 3)``; :meth:`row` gives one time.
    """

    mean: FloatArray
    """Mean relative state [m, m/s]."""
    axes: FloatArray
    """Principal axes of ``A`` (columns, ascending variance)."""
    a_inverse: FloatArray
    """``A^-1`` [1/m^2]."""
    log_det: FloatArray
    """``log det A``."""
    gain: FloatArray
    """``P_vr A^-1`` [1/s]: the velocity's regression on the position."""
    conditional: FloatArray
    """Velocity covariance conditioned on the position [m^2/s^2]."""

    def row(self, k: int) -> _Relative:
        """The time ``k`` of the batch."""
        return _Relative(*(a[k] for a in self))


def _sphere_integral(grid: SphereGrid, gaussian: _Relative, radius: float) -> float:
    """The rate [1/s] at one time: the flux through the hard-body sphere,
    integrated by the product rule ``grid``.

    At each node ``n`` of the unit sphere: the density ``N(R n; r, A)``
    times the flux ``E[max(-n . u, 0)]`` of the relative velocity ``u``
    conditioned on the position, Gaussian with mean ``v + G (R n - r)`` and
    variance ``n' C n``.
    """
    r_mean, v_mean = gaussian.mean[:3], gaussian.mean[3:]
    nodes, weights = grid.cartesian(gaussian.axes)
    # The mean is subtracted (and added below) column by column: the same
    # values as broadcasting it, faster for ``(k, 3)`` arrays.
    offset = radius * nodes
    for j in range(3):
        offset[:, j] -= r_mean[j]
    mahalanobis = np.einsum("ij,ij->i", offset @ gaussian.a_inverse, offset)
    density = (
        np.exp(-0.5 * mahalanobis - 0.5 * gaussian.log_det) / (2.0 * math.pi) ** 1.5
    )
    variance = np.einsum("ij,ij->i", nodes @ gaussian.conditional, nodes)
    sigma = np.sqrt(np.maximum(variance, _TINY_VARIANCE))
    velocity = offset @ gaussian.gain.T
    for j in range(3):
        velocity[:, j] += v_mean[j]
    x = np.einsum("ij,ij->i", nodes, velocity) / (SQRT_2 * sigma)
    flux = sigma / SQRT_2PI * (np.exp(-x * x) - _SQRT_PI * x * special.erfc(x))
    return radius**2 * float(np.sum(weights * density * flux))
