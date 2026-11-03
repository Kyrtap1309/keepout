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
from collections.abc import Iterator
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Final, NamedTuple

import numpy as np

from keepout._types import BoolArray, FloatArray, IndexArray
from keepout._validation import require_at_least, require_positive
from keepout._workers import process_pool, resolve_workers
from keepout.conjunction import Conjunction
from keepout.equinoctial import Orbits
from keepout.orbits import EARTH_MU
from keepout.pc._encounters import Encounters, check_fallbacks
from keepout.pc._plane import require_known_covariance
from keepout.pc._timeline import RateScan, negligible_note
from keepout.pc.result import PcResult

__all__ = [
    "MC_BATCH",
    "MC_CONFIDENCE",
    "MC_MAX_SAMPLES",
    "MC_RTOL",
    "NEWTON_STEPS",
    "SPEED_BOUND_FACTOR",
    "montecarlo",
    "montecarlo_over_scan",
]

MC_RTOL: Final = 0.1
"""Default target: half-width of the confidence interval relative to Pc."""

MC_MAX_SAMPLES: Final = 10_000_000
"""Default budget of samples when stopping on ``rtol``."""

MC_CONFIDENCE: Final = 0.99
"""Default confidence level of the Clopper-Pearson interval."""

MC_BATCH: Final = 50_000
"""Samples per batch: memory stays O(batch) whatever the sample count. Each
batch draws from its own stream of the seed (:class:`numpy.random.SeedSequence`
spawned), so a seed gives the same result in any number of processes."""

NEWTON_STEPS: Final = 12
"""Newton iterations refining a closest approach between two grid points.

The distance is tested at every iterate, so the count bounds the work rather
than setting a convergence tolerance."""

SPEED_BOUND_FACTOR: Final = 1.5
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
        for _ in range(NEWTON_STEPS):
            r, v, a = pairs.relative(t)
            below |= np.linalg.norm(r, axis=1) < hard_body_radius
            f = np.einsum("ij,ij->i", r, v)
            slope = np.einsum("ij,ij->i", v, v) + np.einsum("ij,ij->i", r, a)
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
    :data:`SPEED_BOUND_FACTOR` times the larger of the two speeds.
    """
    gap = later.t - earlier.t
    bound = 0.5 * (
        earlier.distance
        + later.distance
        - SPEED_BOUND_FACTOR * np.maximum(earlier.speed, later.speed) * gap
    )
    rows = np.flatnonzero((bound < hard_body_radius) & candidates)
    if not rows.size:
        return rows
    return rows[batch.closest_below(rows, earlier.t, later.t, hard_body_radius)]


def _collisions(
    batch: _SamplePairs, scan: RateScan, hard_body_radius: float
) -> tuple[int, int]:
    """Collisions (entries) and samples already inside at the window start."""
    r, _ = batch.motion(scan.times[0])
    inside = np.linalg.norm(r, axis=1) < hard_body_radius
    hit = np.zeros_like(inside)
    for first, last in scan.index_ranges:
        # A segment has at least two grid points (it is padded by one step).
        points = _grid(batch, scan.times[first : last + 1])
        for earlier, later in itertools.pairwise(points):
            # Marking the earlier point again is harmless and covers the
            # segment's first point.
            hit |= (earlier.distance < hard_body_radius) | (
                later.distance < hard_body_radius
            )
            hit[_entries_between(batch, earlier, later, hard_body_radius, ~hit)] = True
    return int(np.count_nonzero(hit & ~inside)), int(np.count_nonzero(inside))


def _percent(fraction: float) -> str:
    """``fraction`` as a percentage with its significant digits (99.5%)."""
    return f"{100.0 * fraction:.10g}%"


def _details(
    scan: RateScan,
    confidence: float,
    samples: int,
    hits: int,
    inside: int,
) -> dict[str, float]:
    """The ``details`` of a Monte Carlo result."""
    return {
        "confidence": confidence,
        "samples": float(samples),
        "hits": float(hits),
        "inside_at_start": inside / samples if samples else 0.0,
        "window_start_s": scan.window[0],
        "window_end_s": scan.window[1],
    }


def _no_encounter(scan: RateScan, confidence: float) -> PcResult:
    """The result when the rate scan finds no encounter: no sample is drawn."""
    note = (
        f"no encounter: {negligible_note(scan)}, so no sample can collide there; "
        "no samples were drawn"
    )
    details = _details(scan, confidence, 0, 0, 0)
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
        Number of processes to sample in, a batch of :data:`MC_BATCH` samples
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
        overlapping at the window start, not counted), ``"window_start_s"``,
        ``"window_end_s"``. If the rate scan finds no encounter in the
        window, no sample is drawn: ``value``, ``"samples"`` and the interval
        are 0, with a diagnostic.

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
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
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


def _batch_collisions(
    conjunction: Conjunction,
    encounters: Encounters,
    stream: np.random.SeedSequence,
    count: int,
) -> tuple[int, int]:
    """Collisions and samples inside at the start, of ``count`` samples drawn
    from ``stream``; at module level, for the processes of a pool."""
    rng = np.random.default_rng(stream)
    rate = encounters.rate
    batch = _SamplePairs(
        Orbits.from_elements(rate.primary.sample(rng, count), rate.primary.fr),
        Orbits.from_elements(rate.secondary.sample(rng, count), rate.secondary.fr),
    )
    return _collisions(batch, encounters.scan, conjunction.combined_hard_body_radius)


def _batch_results(
    conjunction: Conjunction,
    encounters: Encounters,
    counts: list[int],
    seed: int | None,
    pool: Executor | None,
    wave: int,
) -> Iterator[tuple[int, int, int]]:
    """Each batch's size, collisions and samples inside at the start, in
    order; with a pool, ``wave`` batches are sampled at a time, so a caller
    that stops early wastes fewer than ``wave``."""
    streams = np.random.SeedSequence(seed).spawn(len(counts))
    batches = list(zip(counts, streams, strict=True))
    for first in range(0, len(batches), wave):
        jobs = batches[first : first + wave]
        if pool is None:
            for count, stream in jobs:
                yield count, *_batch_collisions(conjunction, encounters, stream, count)
            continue
        futures = [
            pool.submit(_batch_collisions, conjunction, encounters, stream, count)
            for count, stream in jobs
        ]
        for (count, _), future in zip(jobs, futures, strict=True):
            yield count, *future.result()


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
    full, rest = divmod(budget, MC_BATCH)
    counts = [MC_BATCH] * full + ([rest] if rest else [])
    processes = min(workers, len(counts))
    hits = inside = done = 0
    low, high = 0.0, 1.0
    converged = False
    with process_pool(processes) if processes > 1 else contextlib.nullcontext() as pool:
        for count, new_hits, new_inside in _batch_results(
            conjunction, encounters, counts, seed, pool, processes
        ):
            hits, inside, done = hits + new_hits, inside + new_inside, done + count
            interval = stats.binomtest(hits, done).proportion_ci(confidence, "exact")
            low, high = float(interval.low), float(interval.high)
            converged = _converged(hits, done, low, high, rtol)
            if samples is None and converged:
                break
    diagnostics: list[str] = []
    if samples is None and not converged:
        diagnostics.append(
            f"Monte Carlo stopped at {done} samples with {hits} collisions: the "
            f"{_percent(confidence)} interval [{low:.3g}, {high:.3g}] is wider than "
            f"rtol = {rtol:g}"
        )
    if inside:
        diagnostics.append(
            f"{inside / done:.3g} of the samples already overlap at the window "
            "start; only entries into the hard body count"
        )
    details = _details(scan, confidence, done, hits, inside)
    return PcResult(
        hits / done, "montecarlo", tuple(diagnostics), details, interval=(low, high)
    )
