"""Monte Carlo Pc with a Clopper-Pearson confidence interval.

Each trial draws both objects' TCA states from their Gaussians in equinoctial
elements (see :mod:`keepout.pc._orbit`), moves them by two-body motion and
counts a collision when the relative distance first drops below the combined
hard-body radius inside the window. Trials already overlapping at the window
start are not collisions in the window (only entries count, as in Hall 2021
and CARA's two-body Monte Carlo); such a trial is not counted even if it
leaves and enters again later. Their fraction is reported.

Time is only searched inside the encounter segments of the 3D-Nc rate scan:
outside them the collision rate is below 1e-16 of its peak. Within a segment
the distance is checked on the scan grid, and wherever a lower bound on a
trial's distance between two grid points is below the radius, its closest
approach is refined by Newton's method on ``r.v = 0`` (slope
``d(r.v)/dt = v.v + r.a``, ``a`` the two-body relative acceleration).
"""

from __future__ import annotations

from typing import Final

import numpy as np

from keepout._types import (
    BoolArray,
    FloatArray,
    IndexArray,
    require_at_least,
    require_positive,
)
from keepout.conjunction import Conjunction
from keepout.equinoctial import Orbits
from keepout.orbits import EARTH_MU
from keepout.pc._plane import require_known_covariance
from keepout.pc._rate import check_fallbacks
from keepout.pc._timeline import RateScan, negligible_note
from keepout.pc.hall3d import Encounters
from keepout.pc.result import PcResult

__all__ = [
    "MC_BATCH",
    "MC_CONFIDENCE",
    "MC_MAX_SAMPLES",
    "MC_RTOL",
    "NEWTON_STEPS",
    "SPEED_MARGIN",
    "montecarlo",
    "sample",
]

MC_RTOL: Final = 0.1
"""Default target: half-width of the confidence interval relative to Pc."""

MC_MAX_SAMPLES: Final = 10_000_000
"""Default budget of trials when stopping on ``rtol``."""

MC_CONFIDENCE: Final = 0.99

MC_BATCH: Final = 50_000
"""Trials per batch: memory stays O(batch) whatever the sample count."""

NEWTON_STEPS: Final = 12

SPEED_MARGIN: Final = 1.5
"""Factor on the larger grid-point relative speed bounding the speed in between."""


def _converged(hits: int, trials: int, low: float, high: float, rtol: float) -> bool:
    """Whether the interval's half-width is at most ``rtol`` of the estimate."""
    return hits > 0 and 0.5 * (high - low) <= rtol * hits / trials


def _gravity(states: FloatArray) -> FloatArray:
    """Two-body accelerations ``-mu r / |r|^3`` of states ``(m, 6)``."""
    r = states[:, :3]
    acceleration: FloatArray = -EARTH_MU * r / np.linalg.norm(r, axis=1)[:, None] ** 3
    return acceleration


class _Trials:
    """A batch of sampled orbit pairs and their relative motion."""

    def __init__(self, first: Orbits, second: Orbits) -> None:
        self.first, self.second = first, second

    def _states(
        self, t: float | FloatArray, rows: slice | IndexArray
    ) -> tuple[FloatArray, FloatArray]:
        """Both objects' Cartesian states ``(m, 6)`` at ``t``."""
        return self.first.cartesian(t, rows), self.second.cartesian(t, rows)

    def motion(
        self, t: float | FloatArray, rows: slice | IndexArray = slice(None)
    ) -> tuple[FloatArray, FloatArray]:
        """Relative position and velocity at ``t``."""
        one, two = self._states(t, rows)
        relative = two - one
        return relative[:, :3], relative[:, 3:]

    def relative(
        self, t: float | FloatArray, rows: slice | IndexArray = slice(None)
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Relative position, velocity and acceleration at ``t``."""
        one, two = self._states(t, rows)
        relative = two - one
        return relative[:, :3], relative[:, 3:], _gravity(two) - _gravity(one)

    def closest_below(
        self, rows: IndexArray, start: float, end: float, radius: float
    ) -> BoolArray:
        """Whether each trial in ``rows`` gets within ``radius`` in the interval.

        The distance is tested at every Newton iterate, not only the last, so
        an iteration that has not converged cannot lose a hit it passed.
        """
        t = np.full(rows.size, 0.5 * (start + end))
        below: BoolArray = np.zeros(rows.size, dtype=bool)
        for _ in range(NEWTON_STEPS):
            r, v, a = self.relative(t, rows)
            below |= np.linalg.norm(r, axis=1) < radius
            f = np.einsum("ij,ij->i", r, v)
            slope = np.einsum("ij,ij->i", v, v) + np.einsum("ij,ij->i", r, a)
            step = np.where(
                slope > 0.0,
                -f / np.where(slope > 0.0, slope, 1.0),
                np.where(f > 0.0, start - t, end - t) * 0.5,
            )
            t = np.clip(t + step, start, end)
        r, _, _ = self.relative(t, rows)
        below |= np.linalg.norm(r, axis=1) < radius
        return below


def _collisions(trials: _Trials, found: RateScan, radius: float) -> tuple[int, int]:
    """Collisions (entries) and trials already inside at the window start."""
    r, _ = trials.motion(found.times[0])
    inside = np.linalg.norm(r, axis=1) < radius
    hit = np.zeros_like(inside)
    for lo, hi in found.segments:
        previous: tuple[FloatArray, FloatArray] | None = None
        for k in range(lo, hi + 1):
            t = float(found.times[k])
            r, v = trials.motion(t)
            distance, speed = np.linalg.norm(r, axis=1), np.linalg.norm(v, axis=1)
            hit |= distance < radius
            if previous is not None:
                gap = t - float(found.times[k - 1])
                bound = 0.5 * (
                    previous[0]
                    + distance
                    - SPEED_MARGIN * np.maximum(previous[1], speed) * gap
                )
                rows = np.flatnonzero((bound < radius) & ~hit)
                if rows.size:
                    below = trials.closest_below(
                        rows, float(found.times[k - 1]), t, radius
                    )
                    hit[rows[below]] = True
            previous = (distance, speed)
    return int(np.count_nonzero(hit & ~inside)), int(np.count_nonzero(inside))


def _percent(fraction: float) -> str:
    """``fraction`` as a percentage with its significant digits (99.5%)."""
    return f"{100.0 * fraction:.10g}%"


def _details(
    found: RateScan,
    confidence: float,
    trials: int,
    hits: int,
    inside: int,
) -> dict[str, float]:
    """The ``details`` of a Monte Carlo result."""
    return {
        "confidence": confidence,
        "samples": float(trials),
        "hits": float(hits),
        "inside_at_start": inside / trials if trials else 0.0,
        "window_start_s": found.window[0],
        "window_end_s": found.window[1],
    }


def _no_encounter(found: RateScan, confidence: float) -> PcResult:
    """The result when the rate scan finds no encounter: no trial is run."""
    note = (
        f"no encounter: {negligible_note(found)}, so no trial can collide there; "
        "no trials were run"
    )
    details = _details(found, confidence, 0, 0, 0)
    return PcResult(0.0, "montecarlo", (note,), details, interval=(0.0, 0.0))


def montecarlo(
    conjunction: Conjunction,
    *,
    window: tuple[float, float] | None = None,
    samples: int | None = None,
    rtol: float = MC_RTOL,
    max_samples: int = MC_MAX_SAMPLES,
    confidence: float = MC_CONFIDENCE,
    seed: int | None = None,
) -> PcResult:
    """Monte Carlo Pc with two-body motion and equinoctial sampling.

    Parameters
    ----------
    conjunction
        The conjunction; both covariances must be known and both orbits
        elliptic.
    window
        Window ``(start, end)`` [s from TCA]; ``None`` sizes it as
        :func:`~keepout.pc.hall3d.hall3d` does.
    samples
        Exact number of trials. ``None`` (default) runs batches until the
        confidence interval's half-width is at most ``rtol`` times the
        estimate, or ``max_samples`` is reached.
    rtol
        Target relative half-width of the confidence interval.
    max_samples
        Budget of trials when stopping on ``rtol``.
    confidence
        Confidence level of the Clopper-Pearson interval.
    seed
        Seed of the random generator; results are reproducible with a seed.

    Returns
    -------
    PcResult
        ``value`` = collisions / trials, with the Clopper-Pearson
        ``interval``. ``details``: ``"confidence"`` (the interval's level),
        ``"samples"``, ``"hits"``, ``"inside_at_start"`` (fraction of trials
        overlapping at the window start, not counted), ``"window_start_s"``,
        ``"window_end_s"``. If the rate scan finds no encounter in the
        window, no trial is run: ``value``, ``"samples"`` and the interval
        are 0, with a diagnostic.

    Raises
    ------
    CovarianceError
        If a covariance is unknown.
    GeometryError
        If an orbit is not elliptic.
    PcConvergenceError
        If the 3D-Nc rate that locates the encounters cannot be computed or
        scanned (see :func:`~keepout.pc._rate.encounter_scan`), or its
        peak-overlap iteration failed inside an encounter segment (see
        :func:`~keepout.pc._rate.check_fallbacks`).
    ValueError
        If an argument is out of range.
    """
    if samples is not None:
        require_at_least(samples, 1, "samples")
    require_positive(rtol, "rtol")
    if not (max_samples >= 1 and 0.0 < confidence < 1.0):
        raise ValueError("need max_samples >= 1 and 0 < confidence < 1")
    require_known_covariance(conjunction.unknown_covariance, "montecarlo")
    encounters = Encounters.scan(conjunction, window)
    return sample(
        conjunction,
        encounters,
        samples=samples,
        rtol=rtol,
        max_samples=max_samples,
        confidence=confidence,
        seed=seed,
    )


def sample(
    conjunction: Conjunction,
    encounters: Encounters,
    *,
    samples: int | None = None,
    rtol: float = MC_RTOL,
    max_samples: int = MC_MAX_SAMPLES,
    confidence: float = MC_CONFIDENCE,
    seed: int | None = None,
) -> PcResult:
    """Monte Carlo Pc over an encounter scan.

    The core of :func:`montecarlo`, without its argument checks: for
    :func:`~keepout.pc.compute_pc`, whose fallback reuses the scan of the
    failed :func:`~keepout.pc.hall3d.hall3d`.
    """
    rate, found = encounters.rate, encounters.found
    # A peak-overlap failure inside a segment could have narrowed the
    # segments searched; the same rule as Hall's applies, checked before
    # sampling since no rate is evaluated after the scan (only the scan's
    # failures count: a reused rate also holds those at Hall's nodes).
    check_fallbacks(encounters.scan_fallbacks, found)
    if not found.segments:
        return _no_encounter(found, confidence)
    # Imported here: scipy.stats would add ~0.2 s to every `import keepout`.
    from scipy import stats

    rng = np.random.default_rng(seed)
    budget = samples if samples is not None else max_samples
    hits = inside = done = 0
    low, high = 0.0, 1.0
    converged = False
    while done < budget:
        count = min(MC_BATCH, budget - done)
        trials = _Trials(
            Orbits(rate.primary.sample(rng, count), rate.primary.fr),
            Orbits(rate.secondary.sample(rng, count), rate.secondary.fr),
        )
        new_hits, new_inside = _collisions(trials, found, conjunction.hard_body_radius)
        hits, inside, done = hits + new_hits, inside + new_inside, done + count
        interval = stats.binomtest(hits, done).proportion_ci(confidence, "exact")
        low, high = float(interval.low), float(interval.high)
        converged = _converged(hits, done, low, high, rtol)
        if samples is None and converged:
            break
    diagnostics: list[str] = []
    if samples is None and not converged:
        diagnostics.append(
            f"Monte Carlo stopped at {done} trials with {hits} collisions: the "
            f"{_percent(confidence)} interval [{low:.3g}, {high:.3g}] is wider than "
            f"rtol = {rtol:g}"
        )
    if inside:
        diagnostics.append(
            f"{inside / done:.3g} of the trials already overlap at the window "
            "start; only entries into the hard body count"
        )
    details = _details(found, confidence, done, hits, inside)
    return PcResult(
        hits / done, "montecarlo", tuple(diagnostics), details, interval=(low, high)
    )
