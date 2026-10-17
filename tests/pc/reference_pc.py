"""Independent reference Pc: the isotropic (Rice) and principal-axes 2-D
integrals, and a brute-force 3-D Nc."""

import math
import warnings
from typing import NamedTuple

import numpy as np
from builders import LEO_R, LEO_V
from pc_builders import SIDE_BY_SIDE_OFFSET, SIDE_BY_SIDE_VARIANCE
from scipy import integrate, special, stats

from keepout._types import FloatArray
from keepout.orbits import EARTH_MU


def isotropic_pc(miss: float, sigma: float, radius: float) -> float:
    """Pc for plane covariance ``sigma^2 I``: the radial (Rice) integral.

    ``integral_0^R (r / s^2) exp(-(r^2 + m^2) / (2 s^2)) I0(r m / s^2) dr``,
    written with ``i0e`` so it neither overflows nor underflows early.
    """

    def integrand(r: float) -> float:
        scaled = r * miss / sigma**2
        exponent = -((r - miss) ** 2) / (2.0 * sigma**2)
        return float(r / sigma**2 * math.exp(exponent) * special.i0e(scaled))

    value, _ = integrate.quad(integrand, 0.0, radius, epsabs=0.0, epsrel=1e-12)
    return float(value)


def principal_axes_pc(
    miss_u: float, miss_v: float, sigma_u: float, sigma_v: float, radius: float
) -> float:
    """Pc for a plane Gaussian given in its principal axes, by brute force.

    Integrates ``N(v; miss_v, sigma_v) * P(|u - miss_u| < h(v))`` over the **narrow**
    axis ``v`` (keepout integrates over the wide axis ``u``), with ``v = R sin(phi)`` so
    the chord ``h = R cos(phi)`` has no square-root end point, breakpoints across the
    narrow density, and ``scipy.stats.norm`` rather than keepout's erf/erfc helpers.
    """

    def integrand(phi: float) -> float:
        v, half_chord = radius * math.sin(phi), radius * math.cos(phi)
        mass = stats.norm.cdf(half_chord, miss_u, sigma_u) - stats.norm.cdf(
            -half_chord, miss_u, sigma_u
        )
        return float(stats.norm.pdf(v, miss_v, sigma_v) * mass * half_chord)

    steps = [
        math.asin((miss_v + k * sigma_v) / radius)
        for k in range(-12, 13, 2)
        if -radius < miss_v + k * sigma_v < radius
    ]
    # Round-off can stop QUADPACK short of epsrel for tiny Pc; the test's own
    # comparison (rtol 1e-7) is the judge, so its warning is not shown.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", integrate.IntegrationWarning)
        value, _ = integrate.quad(
            integrand,
            -0.5 * math.pi,
            0.5 * math.pi,
            points=sorted(steps) or None,
            epsabs=0.0,
            epsrel=1e-10,
            limit=500,
        )
    return float(value)


SIDE_BY_SIDE_WINDOW = (-60.0, 60.0)
"""Window [s from TCA] of :data:`SIDE_BY_SIDE_NC`."""

SIDE_BY_SIDE_NC = 1.507e-4
"""Nc of :func:`pc_builders.side_by_side` over :data:`SIDE_BY_SIDE_WINDOW`:
the entries into the 5 m hard body per trial, counted by brute force.

``side_by_side_entries(10**8, seed=SIDE_BY_SIDE_SEED)`` counts 15071 entries
in 1e8 trials: 1.507e-4 with a standard error of :data:`SIDE_BY_SIDE_NC_ERROR`.
Recomputed by ``test_side_by_side_nc_matches_a_brute_force_count`` in
``test_hall3d.py`` (``-m validation``, ~15 s).

Every entry is the only one of its trial: a trial drifts towards its TCA
position and back. About half are by trials inside the sphere at the window
start, which drift out and back in, so Nc is about twice the chance of
entering from outside.
"""

SIDE_BY_SIDE_NC_ERROR = 1.23e-6
"""Standard error of :data:`SIDE_BY_SIDE_NC`: sqrt(15071) / 1e8 (Poisson)."""

SIDE_BY_SIDE_SEED = 20211


class EntryCount(NamedTuple):
    """Brute-force entries into the hard body."""

    entries: int
    trials: int
    largest_drift: float
    """Largest relative drift ``|d(t) - d(0)| / |d(0)|`` of a propagated
    trial, which :func:`side_by_side_entries` assumes below its shell."""


SHELL = 0.01
"""Relative shell about the hard body in which trials are propagated."""


def _norm(r: FloatArray) -> FloatArray:
    norm: FloatArray = np.sqrt(np.einsum("...i,...i->...", r, r))
    return norm


def _gravity(r: FloatArray) -> FloatArray:
    acceleration: FloatArray = r * (-EARTH_MU / _norm(r) ** 3)[..., None]
    return acceleration


def _rk4(r: FloatArray, v: FloatArray, dt: float) -> tuple[FloatArray, FloatArray]:
    """One classical Runge-Kutta step of two-body motion."""
    k1r, k1v = v, _gravity(r)
    k2r, k2v = v + 0.5 * dt * k1v, _gravity(r + 0.5 * dt * k1r)
    k3r, k3v = v + 0.5 * dt * k2v, _gravity(r + 0.5 * dt * k2r)
    k4r, k4v = v + dt * k3v, _gravity(r + dt * k3r)
    return (
        r + dt / 6.0 * (k1r + 2.0 * k2r + 2.0 * k3r + k4r),
        v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
    )


def _distances(primary: FloatArray, secondary: FloatArray, step: float) -> FloatArray:
    """Distances ``(m, k)`` of trials starting at rest relative to each other
    (both at :data:`builders.LEO_V`), every ``step`` s over the window."""
    start, end = SIDE_BY_SIDE_WINDOW
    sides: list[list[FloatArray]] = []
    for span in (start, end):
        steps = round(abs(span) / step)
        dt = span / steps
        # Both objects of every trial, propagated as one batch (2, m, 3).
        r = np.stack([primary, secondary])
        v = np.broadcast_to(LEO_V, r.shape)
        side = [_norm(r[1] - r[0])]
        for _ in range(steps):
            r, v = _rk4(r, v, dt)
            side.append(_norm(r[1] - r[0]))
        sides.append(side)
    before, after = sides
    return np.stack([*before[:0:-1], *after], axis=1)


def side_by_side_entries(
    trials: int, *, seed: int, radius: float = 5.0, step: float = 0.5
) -> EntryCount:
    """Entries into the hard body of :func:`pc_builders.side_by_side`, by brute
    force: Monte Carlo of the Cartesian Gaussians at TCA and two-body motion.

    Each trial draws both TCA positions (the velocities are exact) and moves
    both objects by Runge-Kutta steps of ``step`` s over
    :data:`SIDE_BY_SIDE_WINDOW`; an entry is a step from outside the sphere to
    inside it. Hall's Gaussian lives in equinoctial elements, but with 5 m
    sigmas it differs from the Cartesian one by ~(5 m)^2 / 7000 km ~ 4 um, far
    below the centimetres the trials drift. At zero relative velocity a trial
    only drifts by the gravity gradient, ``|d(t) - d(0)| <~ n^2 t^2 |d(0)|``,
    0.42% of its distance at 60 s; so only trials within a 1% shell of the
    sphere are propagated (the drift is returned to check that bound). With
    millimetre-per-second drifts, a 0.5 s step resolves every entry.
    """
    rng = np.random.default_rng(seed)
    variance = SIDE_BY_SIDE_VARIANCE
    kept: list[FloatArray] = []
    done = 0
    while done < trials:
        count = min(4_000_000, trials - done)
        offset = math.sqrt(2.0 * variance) * rng.standard_normal((count, 3))
        distance = np.linalg.norm(SIDE_BY_SIDE_OFFSET + offset, axis=1)
        kept.append(offset[np.abs(distance - radius) < SHELL * distance])
        done += count
    relative = np.concatenate(kept)
    if relative.size == 0:
        return EntryCount(0, done, 0.0)
    # The primary's own deviation given the relative one: both objects'
    # deviations are N(0, variance I), so it is N(-relative / 2, variance / 2 I).
    primary = LEO_R - 0.5 * relative
    primary = primary + math.sqrt(0.5 * variance) * rng.standard_normal(relative.shape)
    secondary = primary + SIDE_BY_SIDE_OFFSET + relative
    # Propagated in blocks that stay in the CPU cache.
    distances = np.concatenate(
        [
            _distances(primary[k : k + 16384], secondary[k : k + 16384], step)
            for k in range(0, relative.shape[0], 16384)
        ]
    )
    inside = distances <= radius
    entries = int(np.count_nonzero(~inside[:, :-1] & inside[:, 1:]))
    at_tca = np.linalg.norm(SIDE_BY_SIDE_OFFSET + relative, axis=1)[:, None]
    drift = float(np.max(np.abs(distances - at_tca) / at_tca, initial=0.0))
    return EntryCount(entries, done, drift)
