"""Monte Carlo Pc with a Clopper-Pearson confidence interval.

Each sample draws both objects' TCA states from their Gaussians in equinoctial
elements, moves them by two-body motion and
counts a collision when the relative distance first drops below the combined
hard-body radius inside the window. Samples already overlapping at the window
start are not collisions in the window (only entries count, as in Hall 2021
and CARA's two-body Monte Carlo); such a sample is not counted even if it
leaves and enters again later. Their fraction is reported.

Time is only searched inside the encounter segments of the 3D-Nc rate scan:
outside them the collision rate is below 1e-16 of its peak. Within a segment
the distance is checked on the scan grid, and wherever a lower bound on a
sample's distance between two grid points is below the hard-body radius, its closest
approach is refined by Newton's method on ``r.v = 0`` (slope
``d(r.v)/dt = v.v + r.a``, ``a`` the two-body relative acceleration).
"""

from __future__ import annotations

import contextlib
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, NamedTuple

import numpy as np

from keepout._validation import (
    require_at_least,
    require_open_probability,
    require_positive,
)
from keepout._workers import process_pool, resolve_workers
from keepout.equinoctial import Orbits
from keepout.orbits import EARTH_MU
from keepout.pc._encounters import Encounters, check_fallbacks
from keepout.pc._errors import require_known_covariance
from keepout.pc._timeline import RateScan, negligible_note
from keepout.pc.result import PcResult

if TYPE_CHECKING:
    from collections.abc import Iterator
    from concurrent.futures import Executor

    from keepout._types import BoolArray, FloatArray, IndexArray
    from keepout.conjunction import Conjunction
    from keepout.pc._orbit import OrbitUncertainty

__all__ = [
    "MC_CONFIDENCE",
    "MC_MAX_SAMPLES",
    "MC_RTOL",
    "montecarlo",
    "montecarlo_over_scan",
]

MC_RTOL: Final = 0.1
"""Default target: half-width of the confidence interval relative to Pc."""

MC_MAX_SAMPLES: Final = 10_000_000
"""Default budget of samples when stopping on ``rtol``."""

MC_CONFIDENCE: Final = 0.99
"""Default confidence level of the Clopper-Pearson interval."""

_MC_BATCH: Final = 50_000
"""Samples per batch: memory stays O(batch) whatever the sample count. Each
batch draws from its own stream of the seed (:class:`numpy.random.SeedSequence`
spawned), so a seed gives the same result in any number of processes."""

_NEWTON_STEPS: Final = 12
"""Newton iterations refining a closest approach between two grid points.

The distance is tested at every iterate, so the count bounds the work rather
than setting a convergence tolerance."""

_SPEED_BOUND_FACTOR: Final = 1.5
"""Factor on the larger grid-point relative speed bounding the speed in between."""


def _converged(hits: int, samples: int, low: float, high: float, rtol: float) -> bool:
    """Whether the interval's half-width is at most ``rtol`` of the estimate."""
    return hits > 0 and 0.5 * (high - low) <= rtol * hits / samples


def _gravity(states: FloatArray) -> FloatArray:
    """Two-body accelerations ``-mu r / |r|^3`` of states ``(m, 6)``."""
    r = states[:, :3]
    acceleration: FloatArray = -EARTH_MU * r / np.linalg.norm(r, axis=1)[:, None] ** 3
    return acceleration


class _SamplePairs(NamedTuple):
    """A batch of sampled orbit pairs and their relative motion."""

    primary: Orbits
    """The primary's sampled orbits."""
    secondary: Orbits
    """The secondary's, pair by pair."""

    def subset(self, rows: IndexArray) -> _SamplePairs:
        """The sample pairs ``rows``."""
        return _SamplePairs(self.primary.subset(rows), self.secondary.subset(rows))

    def _states(
        self, t: float | FloatArray, rows: slice | IndexArray
    ) -> tuple[FloatArray, FloatArray]:
        """Both objects' Cartesian states ``(m, 6)`` at ``t``."""
        return self.primary.cartesian(t, rows), self.secondary.cartesian(t, rows)

    def motion(
        self, t: float | FloatArray, rows: slice | IndexArray = slice(None)
    ) -> tuple[FloatArray, FloatArray]:
        """Relative position and velocity at ``t``."""
        primary, secondary = self._states(t, rows)
        relative = secondary - primary
        return relative[:, :3], relative[:, 3:]

    def relative(
        self, t: float | FloatArray, rows: slice | IndexArray = slice(None)
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Relative position, velocity and acceleration at ``t``."""
        primary, secondary = self._states(t, rows)
        relative = secondary - primary
        return (
            relative[:, :3],
            relative[:, 3:],
            _gravity(secondary) - _gravity(primary),
        )

    def closest_below(
        self, rows: IndexArray, start: float, end: float, hard_body_radius: float
    ) -> BoolArray:
        """Whether each sample in ``rows`` gets within ``hard_body_radius`` in
        the interval.

        The distance is tested at every Newton iterate, not only the last, so
        an iteration that has not converged cannot lose a hit it passed.
        """
        # Taken once: every Newton step evaluates the same rows.
        pairs = self.subset(rows)
        t = np.full(rows.size, 0.5 * (start + end))
        below: BoolArray = np.zeros(rows.size, dtype=bool)
        for _ in range(_NEWTON_STEPS):
            r, v, a = pairs.relative(t)
            below |= np.linalg.norm(r, axis=1) < hard_body_radius
            f = np.vecdot(r, v)
            slope = np.vecdot(v, v) + np.vecdot(r, a)
            step = np.where(
                slope > 0.0,
                -f / np.where(slope > 0.0, slope, 1.0),
                np.where(f > 0.0, start - t, end - t) * 0.5,
            )
            t = np.clip(t + step, start, end)
        r, _, _ = pairs.relative(t)
        below |= np.linalg.norm(r, axis=1) < hard_body_radius
        return below


@dataclass(frozen=True, eq=False, slots=True)
class _GridPoint:
    """Every sample's relative distance and speed at one time of the scan grid."""

    t: float
    distance: FloatArray
    speed: FloatArray


def _grid(batch: _SamplePairs, times: FloatArray) -> Iterator[_GridPoint]:
    """The batch at each of ``times``, one grid point at a time."""
    for t in times:
        r, v = batch.motion(float(t))
        yield _GridPoint(float(t), np.linalg.norm(r, axis=1), np.linalg.norm(v, axis=1))


def _entries_between(
    batch: _SamplePairs,
    earlier: _GridPoint,
    later: _GridPoint,
    hard_body_radius: float,
    candidates: BoolArray,
) -> IndexArray:
    """The ``candidates`` that get within ``hard_body_radius`` between two grid
    points.

    Only samples whose distance may drop below it in between are
    refined. The distance in between is at least half of the two distances'
    sum less the distance covered over the gap at the bounding speed,
    ``_SPEED_BOUND_FACTOR`` times the larger of the two speeds.
    """
    gap = later.t - earlier.t
    bound = 0.5 * (
        earlier.distance
        + later.distance
        - _SPEED_BOUND_FACTOR * np.maximum(earlier.speed, later.speed) * gap
    )
    rows = np.flatnonzero((bound < hard_body_radius) & candidates)
    if not rows.size:
        return rows
    return rows[batch.closest_below(rows, earlier.t, later.t, hard_body_radius)]


def _percent(fraction: float) -> str:
    """``fraction`` as a percentage with its significant digits (99.5%)."""
    return f"{100.0 * fraction:.10g}%"


class _Counts(NamedTuple):
    """Sample counts of a batch, or of a run."""

    samples: int
    hits: int
    """Collisions: entries into the hard body in the window."""
    inside: int
    """Samples already inside the hard body at the window start."""
    non_elliptic: int
    """Samples off the ellipse (counted as misses)."""

    def plus(self, other: _Counts) -> _Counts:
        """Both counts summed."""
        return _Counts(*(a + b for a, b in zip(self, other, strict=True)))


def _details(scan: RateScan, confidence: float, counts: _Counts) -> dict[str, float]:
    """The ``details`` of a Monte Carlo result."""
    samples = counts.samples
    return {
        "confidence": confidence,
        "samples": float(samples),
        "hits": float(counts.hits),
        "inside_at_start": counts.inside / samples if samples else 0.0,
        "non_elliptic": float(counts.non_elliptic),
        "window_start_s": scan.window[0],
        "window_end_s": scan.window[1],
    }


def _no_encounter(scan: RateScan, confidence: float) -> PcResult:
    """The result when the rate scan finds no encounter: no sample is drawn."""
    note = (
        f"no encounter: {negligible_note(scan)}, so no sample can collide there; "
        "no samples were drawn"
    )
    details = _details(scan, confidence, _Counts(0, 0, 0, 0))
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
    workers: int | None = 1,
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
        Exact number of samples. ``None`` (default) runs batches until the
        confidence interval's half-width is at most ``rtol`` times the
        estimate, or ``max_samples`` is reached.
    rtol
        Target relative half-width of the confidence interval.
    max_samples
        Budget of samples when stopping on ``rtol``.
    confidence
        Confidence level of the Clopper-Pearson interval.
    seed
        Seed of the random generator; results are reproducible with a seed,
        whatever ``workers``.
    workers
        Number of processes to sample in, a batch of 50,000 samples
        each at a time; 1 samples in this process, ``None`` uses one per
        CPU, at most 4. Processes pay off from two batches (a batch takes
        seconds); a script calling with more than 1 needs an
        ``if __name__ == "__main__":`` guard.

    Returns
    -------
    PcResult
        ``value`` = collisions / samples, with the Clopper-Pearson
        ``interval``. ``details``: ``"confidence"`` (the interval's level),
        ``"samples"``, ``"hits"``, ``"inside_at_start"`` (fraction of samples
        overlapping at the window start, not counted), ``"non_elliptic"``
        (samples whose elements are no ellipse, counted as misses, with a
        diagnostic), ``"window_start_s"``, ``"window_end_s"``. If the rate
        scan finds no encounter in the window, no sample is drawn:
        ``value``, ``"samples"`` and the interval are 0, with a diagnostic.

    Raises
    ------
    CovarianceError
        If a covariance is unknown.
    GeometryError
        If an orbit is not elliptic.
    PcConvergenceError
        If the 3D-Nc rate that locates the encounters cannot be computed or
        scanned (its encounters need more detection points than allowed, the
        rate is significant where the screen ruled it negligible, or the rate
        cannot be computed on the scan grid), or its peak-overlap iteration
        failed inside an encounter segment.
    ValueError
        If an argument is out of range.
    """
    processes = resolve_workers(workers)
    if samples is not None:
        require_at_least(samples, 1, "samples")
    require_positive(rtol, "rtol")
    require_at_least(max_samples, 1, "max_samples")
    require_open_probability(confidence, "confidence")
    require_known_covariance(conjunction.unknown_covariance, "montecarlo")
    encounters = Encounters.locate(conjunction, window)
    return montecarlo_over_scan(
        conjunction,
        encounters,
        samples=samples,
        rtol=rtol,
        max_samples=max_samples,
        confidence=confidence,
        seed=seed,
        workers=processes,
    )


def _elliptic(elements: FloatArray) -> BoolArray:
    """Whether each element sample ``(m, 6)`` is an ellipse: ``n > 0`` and
    ``|(af, ag)| < 1``."""
    af, ag = elements[:, 1], elements[:, 2]
    elliptic: BoolArray = (elements[:, 0] > 0.0) & (1.0 - af * af - ag * ag > 0.0)
    return elliptic


@dataclass(frozen=True, eq=False, slots=True)
class _Sampler:
    """What a batch of samples needs, and all a pool's process is sent: both
    orbits, the scan grid with its encounter segments and the hard-body
    radius."""

    primary: OrbitUncertainty
    secondary: OrbitUncertainty
    times: FloatArray
    index_ranges: tuple[tuple[int, int], ...]
    hard_body_radius: float

    def batch(self, stream: np.random.SeedSequence, count: int) -> _Counts:
        """The counts of ``count`` samples drawn from ``stream``.

        A sample pair off the ellipse (the Gaussian in the elements has tails
        there) cannot be propagated: it is a miss.
        """
        rng = np.random.default_rng(stream)
        primary = self.primary.sample(rng, count)
        secondary = self.secondary.sample(rng, count)
        elliptic = _elliptic(primary) & _elliptic(secondary)
        pairs = _SamplePairs(
            Orbits.from_elements(primary[elliptic], self.primary.fr),
            Orbits.from_elements(secondary[elliptic], self.secondary.fr),
        )
        hits, inside = self._collisions(pairs)
        return _Counts(count, hits, inside, count - int(np.count_nonzero(elliptic)))

    def _collisions(self, pairs: _SamplePairs) -> tuple[int, int]:
        """Collisions (entries) and samples already inside at the window start."""
        radius = self.hard_body_radius
        r, _ = pairs.motion(self.times[0])
        inside = np.linalg.norm(r, axis=1) < radius
        hit = np.zeros_like(inside)
        for first, last in self.index_ranges:
            # A segment has at least two grid points (it is padded by one step).
            points = _grid(pairs, self.times[first : last + 1])
            for earlier, later in itertools.pairwise(points):
                # Marking the earlier point again is harmless and covers the
                # segment's first point.
                hit |= (earlier.distance < radius) | (later.distance < radius)
                hit[_entries_between(pairs, earlier, later, radius, ~hit)] = True
        return int(np.count_nonzero(hit & ~inside)), int(np.count_nonzero(inside))


def _batch_counts(
    sampler: _Sampler, stream: np.random.SeedSequence, count: int
) -> _Counts:
    """:meth:`_Sampler.batch`, at module level for the processes of a pool."""
    return sampler.batch(stream, count)


def _batch_results(
    sampler: _Sampler,
    counts: list[int],
    seed: int | None,
    pool: Executor | None,
    wave: int,
) -> Iterator[_Counts]:
    """Each batch's counts, in order; with a pool, ``wave`` batches are
    sampled at a time, so a caller that stops early wastes fewer than
    ``wave``."""
    streams = np.random.SeedSequence(seed).spawn(len(counts))
    batches = list(zip(counts, streams, strict=True))
    for first in range(0, len(batches), wave):
        jobs = batches[first : first + wave]
        if pool is None:
            for count, stream in jobs:
                yield sampler.batch(stream, count)
            continue
        futures = [
            pool.submit(_batch_counts, sampler, stream, count) for count, stream in jobs
        ]
        for future in futures:
            yield future.result()


def montecarlo_over_scan(
    conjunction: Conjunction,
    encounters: Encounters,
    *,
    samples: int | None = None,
    rtol: float = MC_RTOL,
    max_samples: int = MC_MAX_SAMPLES,
    confidence: float = MC_CONFIDENCE,
    seed: int | None = None,
    workers: int = 1,
) -> PcResult:
    """Monte Carlo Pc over an encounter scan.

    The core of :func:`montecarlo`, without its argument checks: for
    :func:`~keepout.pc.compute_pc`, whose fallback reuses the scan of the
    failed :func:`~keepout.pc.hall3d.hall3d`.

    Parameters
    ----------
    conjunction
        The conjunction, both covariances known and both orbits elliptic.
    encounters
        The scan of its collision rate, whose encounter segments are sampled.
    samples, rtol, max_samples, confidence, seed
        As for :func:`montecarlo`.
    workers
        Number of processes, at least 1 (``None`` already resolved).

    Returns
    -------
    PcResult
        The Monte Carlo Pc and its confidence interval.
    """
    scan = encounters.scan
    # A peak-overlap failure inside a segment could have narrowed the
    # segments searched; the same rule as Hall's applies, checked before
    # sampling since no rate is evaluated after the scan (only the scan's
    # failures count: a reused rate also holds those at Hall's nodes).
    check_fallbacks(encounters.scan_fallbacks, scan)
    if not scan.index_ranges:
        return _no_encounter(scan, confidence)
    # Imported here: scipy.stats would add ~0.2 s to every `import keepout`.
    from scipy import stats

    budget = samples if samples is not None else max_samples
    full, rest = divmod(budget, _MC_BATCH)
    counts = [_MC_BATCH] * full + ([rest] if rest else [])
    processes = min(workers, len(counts))
    rate = encounters.rate
    sampler = _Sampler(
        rate.primary,
        rate.secondary,
        scan.times,
        scan.index_ranges,
        conjunction.combined_hard_body_radius,
    )
    total = _Counts(0, 0, 0, 0)
    low, high = 0.0, 1.0
    converged = False
    with process_pool(processes) if processes > 1 else contextlib.nullcontext() as pool:
        for batch in _batch_results(sampler, counts, seed, pool, processes):
            total = total.plus(batch)
            hits, done = total.hits, total.samples
            interval = stats.binomtest(hits, done).proportion_ci(confidence, "exact")
            low, high = float(interval.low), float(interval.high)
            converged = _converged(hits, done, low, high, rtol)
            if samples is None and converged:
                break
    hits, done = total.hits, total.samples
    diagnostics: list[str] = []
    if samples is None and not converged:
        diagnostics.append(
            f"Monte Carlo stopped at {done} samples with {hits} collisions: the "
            f"{_percent(confidence)} interval [{low:.3g}, {high:.3g}] is wider than "
            f"rtol = {rtol:g}"
        )
    if total.inside:
        diagnostics.append(
            f"{total.inside / done:.3g} of the samples already overlap at the window "
            "start; only entries into the hard body count"
        )
    if total.non_elliptic:
        diagnostics.append(
            f"{total.non_elliptic} of the {done} samples "
            f"({_percent(total.non_elliptic / done)}) are not elliptic orbits "
            "(eccentricity of 1 or more, or a mean motion not above 0): counted "
            "as misses"
        )
    details = _details(scan, confidence, total)
    return PcResult(
        hits / done, "montecarlo", tuple(diagnostics), details, interval=(low, high)
    )
