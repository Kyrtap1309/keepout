"""Brute-force reference for the screening tests: every pair, densely sampled.

:func:`brute_force` propagates every object on a uniform time grid with
sgp4's own ``SatrecArray`` at astropy's UTC dates (not through
``keepout.propagation``), computes the distance of every primary-catalog
pair at every sample and reports each local minimum of the sampled distance
within a reach. It uses positions only: SGP4's velocity is not exactly the
derivative of its position. :func:`refine` moves one minimum to the minimum
of the distance between its neighbouring samples (bounded Brent on the
positions of two single ``Satrec``), for TCA and miss-distance references.

What the oracle finds, screening must find: a sampled distance within the
reach means the pair came that close. (The converse does not hold: a
minimum between two samples can be missed when the pair passes the reach
for less than a step.) An object with an SGP4 error at any sample is left
out and listed in :attr:`Oracle.failed`, as screening skips it; a primary
is not paired with its own catalog number.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.time import Time
from scipy.optimize import minimize_scalar
from sgp4.api import SatrecArray

from keepout._types import FloatArray
from keepout.omm import Omm

ORACLE_STEP: Final = 10.0
"""Default time between samples [s]."""

BLOCK_SAMPLES: Final = 4_000_000
"""Most pair-samples whose distances are held at once."""


def positions(
    omms: Sequence[Omm], start: Time, offsets: npt.ArrayLike
) -> tuple[FloatArray, npt.NDArray[np.uint8]]:
    """TEME positions [m] and SGP4 error codes of objects on a time grid.

    Parameters
    ----------
    omms
        The objects.
    start
        Reference time of the grid.
    offsets
        Seconds since ``start``, shape ``(T,)``.

    Returns
    -------
    tuple[FloatArray, npt.NDArray[np.uint8]]
        Positions ``(N, T, 3)`` (NaN where SGP4 fails) and error codes
        ``(N, T)``.
    """
    times = (start + np.asarray(offsets, dtype=np.float64) * u.s).utc
    if not omms:
        return np.empty((0, len(times), 3)), np.empty((0, len(times)), np.uint8)
    errors, r, _ = SatrecArray([omm.satrec() for omm in omms]).sgp4(
        np.atleast_1d(times.jd1), np.atleast_1d(times.jd2)
    )
    return r * 1e3, errors


def local_minima(distances: FloatArray) -> npt.NDArray[np.bool_]:
    """Where sampled distances have a local minimum, along the last axis.

    A sample is a minimum when it is below the previous sample and not above
    the next (the first sample of a flat run counts); the first and last
    samples compare with their one neighbour only.
    """
    below_previous = np.ones(distances.shape, dtype=bool)
    below_previous[..., 1:] = distances[..., 1:] < distances[..., :-1]
    not_above_next = np.ones(distances.shape, dtype=bool)
    not_above_next[..., :-1] = distances[..., :-1] <= distances[..., 1:]
    minima: npt.NDArray[np.bool_] = below_previous & not_above_next
    return minima


@dataclass(frozen=True)
class OracleMinimum:
    """A local minimum of the sampled distance of a pair.

    Parameters
    ----------
    primary, secondary
        Catalog numbers.
    sample
        Index of the sample on the grid.
    offset
        Time of the minimum [s since the grid's start].
    distance
        Distance at the minimum [m].
    at_edge
        Whether the sample is the first or the last of the grid (the
        distance may still be falling outside the window).
    """

    primary: int
    secondary: int
    sample: int
    offset: float
    distance: float
    at_edge: bool


@dataclass(frozen=True)
class Oracle:
    """What :func:`brute_force` finds.

    Parameters
    ----------
    start
        Start of the window.
    offsets
        The grid [s since ``start``].
    minima
        Every local minimum within the reach, by primary, secondary and time.
    failed
        Catalog numbers of the objects left out (an SGP4 error on the grid).
    """

    start: Time
    offsets: FloatArray
    minima: tuple[OracleMinimum, ...]
    failed: frozenset[int]

    @property
    def pairs(self) -> frozenset[tuple[int, int]]:
        """``(primary, secondary)`` of every pair with a minimum."""
        return frozenset((m.primary, m.secondary) for m in self.minima)

    def within(self, reach: float) -> tuple[OracleMinimum, ...]:
        """The minima at ``reach`` [m] or closer."""
        return tuple(m for m in self.minima if m.distance <= reach)


def brute_force(
    primaries: Sequence[Omm],
    catalog: Sequence[Omm],
    start: Time,
    duration: float,
    reach: float | Sequence[float],
    *,
    step: float = ORACLE_STEP,
) -> Oracle:
    """Every local minimum of every pair's sampled distance within a reach.

    Parameters
    ----------
    primaries, catalog
        The objects; each primary is paired with every catalog object of
        another catalog number.
    start
        Start of the window.
    duration
        Length of the window [s], >= 0.
    reach
        Largest distance reported [m], for every primary or per primary.
    step
        Longest time between samples [s]; the grid spans the window evenly,
        both ends included.
    """
    count = max(1, math.ceil(duration / step)) + 1 if duration > 0 else 1
    offsets = np.linspace(0.0, duration, count)
    reaches = np.broadcast_to(np.asarray(reach, dtype=np.float64), (len(primaries),))
    r_primary, e_primary = positions(primaries, start, offsets)
    r_catalog, e_catalog = positions(catalog, start, offsets)
    good_primary = ~np.any(e_primary, axis=1)
    good_catalog = ~np.any(e_catalog, axis=1)
    failed = frozenset(
        [
            o.catalog_number
            for o, ok in zip(primaries, good_primary, strict=True)
            if not ok
        ]
        + [
            o.catalog_number
            for o, ok in zip(catalog, good_catalog, strict=True)
            if not ok
        ]
    )
    numbers_p = np.array([o.catalog_number for o in primaries], dtype=np.int64)
    numbers_c = np.array([o.catalog_number for o in catalog], dtype=np.int64)
    block = max(1, BLOCK_SAMPLES // max(1, len(primaries) * count))
    found: list[OracleMinimum] = []
    for first in range(0, len(catalog), block):
        chosen = slice(first, first + block)
        distances = np.linalg.norm(
            r_catalog[None, chosen] - r_primary[:, None], axis=-1
        )
        usable = (
            good_primary[:, None]
            & good_catalog[None, chosen]
            & (numbers_p[:, None] != numbers_c[None, chosen])
        )
        hits = (
            local_minima(distances)
            & (distances <= reaches[:, None, None])
            & usable[:, :, None]
        )
        p, s, k = np.nonzero(hits)
        found.extend(
            OracleMinimum(
                primary=int(numbers_p[i]),
                secondary=int(numbers_c[first + j]),
                sample=int(sample),
                offset=float(offsets[sample]),
                distance=float(distances[i, j, sample]),
                at_edge=bool(sample in (0, count - 1)),
            )
            for i, j, sample in zip(p.tolist(), s.tolist(), k.tolist(), strict=True)
        )
    found.sort(key=lambda m: (m.primary, m.secondary, m.sample))
    return Oracle(start, offsets, tuple(found), failed)


def refine(
    primary: Omm, secondary: Omm, oracle: Oracle, minimum: OracleMinimum
) -> OracleMinimum:
    """The minimum of the distance between a sampled minimum's neighbours.

    Bounded Brent (to 1 us) on the distance of the two objects' positions
    from single ``Satrec`` at astropy's UTC dates. It searches the time since
    the earlier neighbour: its tolerance grows with the size of the variable
    (``sqrt(eps) |x|``), which would be milliseconds days into a window.
    """
    satrecs = (primary.satrec(), secondary.satrec())
    offsets = oracle.offsets
    lowest = offsets[max(0, minimum.sample - 1)]
    highest = offsets[min(len(offsets) - 1, minimum.sample + 1)]

    def distance(elapsed: float) -> float:
        time = (oracle.start + (lowest + elapsed) * u.s).utc
        a, b = (satrec.sgp4(time.jd1, time.jd2)[1] for satrec in satrecs)
        return float(np.linalg.norm(np.subtract(b, a))) * 1e3

    if highest == lowest:
        return minimum
    result = minimize_scalar(
        distance,
        bounds=(0.0, highest - lowest),
        method="bounded",
        options={"xatol": 1e-6},
    )
    best = min(
        (lowest + float(result.x), float(result.fun)),
        (minimum.offset, minimum.distance),
        key=lambda pair: pair[1],
    )
    return dataclasses.replace(minimum, offset=best[0], distance=best[1])
