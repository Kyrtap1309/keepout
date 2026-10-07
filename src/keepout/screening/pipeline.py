"""Screening: every close approach of some primaries with a catalog over a window.

:func:`screen` takes the primaries' and the catalog's element sets (OMM, SGP4
theory) and a window, and returns a
:class:`~keepout.screening.result.ScreeningResult`:

1. **Element sets.** Of several element sets of one object, the newest is
   screened (:func:`keepout.omm.deduplicate`). An element set is left out
   when the window ends more than ``max_age`` after its epoch (stale) or
   starts more than ``max_lead`` before it, when SGP4 cannot be initialised
   with it, or (a primary) when the volume rule refuses it (raises a
   :class:`~keepout.errors.KeepoutError`).
2. **Pairs.** Every primary is paired with every catalog object, except
   itself (the same catalog number); the apogee/perigee filter
   (:func:`~keepout.screening.filters.apogee_perigee_filter`) drops the pairs
   whose distances from the Earth's centre keep them out of the primary's
   volume.
3. **Search.** :class:`~keepout.screening.tca.ApproachSearch` finds, on a
   time grid, every local minimum of a pair's range inside the volume, with
   its TCA. An object for which SGP4 reports an error inside the window is
   left out, with every minimum it has: its states before the error are not
   trusted either. (An object the filter rules out of every pair is never
   propagated, so an error it would have is not seen.)
4. **Approaches.** The states at each TCA are converted from the
   propagator's frame (TEME for SGP4) to GCRF
   (:func:`keepout.frames.frame_to_gcrf`) into a
   :class:`~keepout.screening.result.CloseApproach`.

**Element age.** The search's guarantee that no minimum is missed assumes
every acceleration beyond the point mass and J2 stays below
:data:`~keepout.screening.tca.PERTURBATION` (1 m/s^2). On the public catalog
of October 2026, SGP4 stays below 0.2 m/s^2 from 3 days before to 38 days
after each element epoch, with one exception (below); further from the
epoch some objects leave the bound (from 39 days after, by orders of
magnitude, and from 18 days before). Hence the defaults :data:`MAX_AGE` (30
days) and :data:`MAX_LEAD` (3 days). The exception is an object re-entering
the atmosphere: with its perigee 113 km up, SGP4's drag on it exceeds 1
m/s^2 from 8 days after its epoch, days before SGP4 fails for it. The
guarantee does not cover such objects. GP data loses accuracy much sooner
than this (kilometres within days in low orbits), so a smaller ``max_age``
may suit an application better.

**Workers.** The catalog is dealt into ``workers`` parts (object ``i`` to
part ``i mod workers``); each part is initialised, filtered and searched in
its own process (``spawn``, the same on every platform), and the results are
merged. Each minimum depends on its own pair's states only, so the result is
the same, bit for bit, for any number of workers and any ``max_states``.
The processes import the main module again: with ``workers`` above 1, a
script must call :func:`screen` under ``if __name__ == "__main__":``, and a
script read from standard input (``python -``) cannot.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Final

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.time import Time

from keepout._types import (
    FloatArray,
    IndexArray,
    require_non_negative,
    require_positive,
)
from keepout._units import DAY
from keepout.errors import KeepoutError
from keepout.frames import frame_to_gcrf
from keepout.omm import Omm, deduplicate
from keepout.propagation.base import (
    CHUNK_STATES,
    Propagator,
    RadiusBounds,
    describe_error,
)
from keepout.propagation.sgp4_prop import Sgp4Propagator
from keepout.screening.filters import apogee_perigee_filter
from keepout.screening.result import (
    CloseApproach,
    ScreeningResult,
    ScreeningRole,
    SkippedObject,
    SkipReason,
)
from keepout.screening.tca import (
    DEFAULT_STEP,
    ApproachSearch,
    ObjectFailure,
    RangeMinimum,
    SearchResult,
)
from keepout.screening.volume import (
    DEFAULT_VOLUME,
    ScreeningVolume,
    VolumeRule,
    volume_for,
)
from keepout.state import RefFrame, StateVector

__all__ = ["DEFAULT_DURATION", "MAX_AGE", "MAX_LEAD", "screen"]

DEFAULT_DURATION: Final = 7 * DAY
"""Length of the window unless told otherwise [s]: 7 days, as SOCRATES."""

MAX_AGE: Final = 30 * DAY
"""Longest time from an element epoch to the end of the window [s] unless
told otherwise; older element sets are stale and left out."""

MAX_LEAD: Final = 3 * DAY
"""Longest time from the start of the window to a later element epoch [s]
unless told otherwise. CelesTrak's data hold epochs up to about a day and a
half after the start of the day they are downloaded on."""


def screen(
    primaries: Iterable[Omm],
    catalog: Iterable[Omm],
    start: Time,
    duration: float = DEFAULT_DURATION,
    volume: ScreeningVolume | VolumeRule = DEFAULT_VOLUME,
    *,
    workers: int = 1,
    max_age: float = MAX_AGE,
    max_lead: float = MAX_LEAD,
    step: float = DEFAULT_STEP,
    max_states: int = CHUNK_STATES,
) -> ScreeningResult:
    """Find every close approach of the primaries with the catalog in a window.

    See the module's description for the method.

    Parameters
    ----------
    primaries
        Element sets of the protected objects.
    catalog
        Element sets of the objects to screen them against. A primary may
        be in it too: it is not paired with itself.
    start
        Start of the window; one time.
    duration
        Length of the window [s], finite and >= 0. Approaches are reported
        when their TCA lies in ``(start, start + duration]``.
    volume
        The screening volume of every primary, or a rule giving each
        primary's own (e.g. ``ScreeningVolume.sfs_handbook``); called once
        per primary, in this process. A primary the rule raises a
        :class:`~keepout.errors.KeepoutError` for is left out; any other
        exception the rule raises propagates.
    workers
        Number of processes to screen in; 1 screens in this process.
    max_age
        Longest time [s] from an element epoch to the end of the window;
        element sets older than that are left out as stale.
    max_lead
        Longest time [s] from the start of the window to a later element
        epoch; element sets further ahead are left out.
    step
        Longest time between the search's grid samples [s]
        (:func:`~keepout.screening.tca.time_grid`); the result does not
        depend on it, the time it takes does.
    max_states
        Most states (objects x times) a process propagates at once; bounds
        the memory (about 49 bytes per state).

    Returns
    -------
    ScreeningResult
        The close approaches in chronological order, the element sets left
        out with their reasons, the window and each primary's volume.

    Raises
    ------
    ValueError
        If ``start`` is not one time, or ``duration``, ``max_age``,
        ``max_lead``, ``step``, ``workers`` or ``max_states`` is out of range.
    TypeError
        If the volume rule returns something other than a
        :class:`~keepout.screening.volume.ScreeningVolume`.
    """
    if not start.isscalar:
        raise ValueError(f"start must be a single time, got shape {start.shape}")
    require_non_negative(duration, "duration")
    require_non_negative(max_age, "max_age")
    require_non_negative(max_lead, "max_lead")
    require_positive(step, "step")
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    if max_states < 1:
        raise ValueError(f"max_states must be at least 1, got {max_states}")
    window = _Window(start, float(duration), float(step), max_states)
    skipped: list[SkippedObject] = []
    own = _current(primaries, ScreeningRole.PRIMARY, window, max_age, max_lead, skipped)
    others = _current(
        catalog, ScreeningRole.CATALOG, window, max_age, max_lead, skipped
    )
    propagator, volumes = _primaries(own, volume, skipped)
    bounds = propagator.radius_bounds(start, window.duration)
    numbers = np.array([omm.catalog_number for omm in propagator.omms], dtype=np.int64)
    count = min(workers, len(others))
    parts = [
        _Part(
            window,
            propagator,
            numbers,
            bounds,
            tuple(volumes),
            others[part::count],
            np.arange(part, len(others), count),
        )
        for part in range(count)
    ]
    results = _run(parts, workers)
    lost = [
        (others[index], message)
        for result in results
        for index, message in result.unsupported
    ]
    _skip_unsupported(lost, ScreeningRole.CATALOG, skipped)
    found = SearchResult.merge(result.search for result in results)
    primary_omms = propagator.omms
    gone_primaries = _skip_failed(
        found.primary_failures, primary_omms, ScreeningRole.PRIMARY, window, skipped
    )
    gone = _skip_failed(
        found.catalog_failures, others, ScreeningRole.CATALOG, window, skipped
    )
    minima = [
        minimum
        for minimum in found.minima
        if minimum.primary_index not in gone_primaries
        and minimum.secondary_index not in gone
    ]
    screened_volumes = {
        omm.catalog_number: volumes[index]
        for index, omm in enumerate(primary_omms)
        if index not in gone_primaries
    }
    return ScreeningResult(
        approaches=tuple(
            _approaches(minima, primary_omms, others, start, propagator.frame)
        ),
        skipped=tuple(skipped),
        start=start,
        duration=window.duration,
        volumes=screened_volumes,
        catalog_count=len(others) - len(lost) - len(gone),
    )


@dataclass(frozen=True)
class _Window:
    """The window and the search's settings, shared by every part."""

    start: Time
    duration: float
    step: float
    max_states: int


@dataclass(frozen=True, eq=False)
class _Part:
    """The primaries and one part of the catalog, to screen in one process."""

    window: _Window
    primaries: Propagator
    primary_numbers: npt.NDArray[np.int64]
    """Catalog number of each primary."""
    primary_bounds: RadiusBounds
    volumes: tuple[ScreeningVolume, ...]
    catalog: tuple[Omm, ...]
    columns: IndexArray
    """Position of each of ``catalog`` in the whole catalog."""


@dataclass(frozen=True, eq=False)
class _PartResult:
    """What one part found; catalog positions are in the whole catalog."""

    search: SearchResult
    unsupported: tuple[tuple[int, str], ...]


def _current(
    omms: Iterable[Omm],
    role: ScreeningRole,
    window: _Window,
    max_age: float,
    max_lead: float,
    skipped: list[SkippedObject],
) -> tuple[Omm, ...]:
    """The newest element set of each object, unless too far from the window."""
    unique = deduplicate(omms)
    newest = {omm.catalog_number: omm for omm in unique.kept}
    for omm in unique.dropped:
        used = newest[omm.catalog_number].epoch.isot
        detail = f"epoch {omm.epoch.isot}; the element set of epoch {used} is used"
        skipped.append(SkippedObject(omm, role, SkipReason.DUPLICATE, detail))
    if not unique.kept:
        return ()
    epochs = Time(
        [omm.epoch.jd1 for omm in unique.kept],
        [omm.epoch.jd2 for omm in unique.kept],
        format="jd",
        scale="utc",
    )
    since = (window.start - epochs).to_value(u.s)
    current: list[Omm] = []
    for omm, before in zip(unique.kept, since.tolist(), strict=True):
        age, lead = before + window.duration, -before
        if age > max_age:
            detail = (
                f"the window ends {age / DAY:.2f} days after the epoch "
                f"{omm.epoch.isot} (max_age {max_age / DAY:g} days)"
            )
            skipped.append(SkippedObject(omm, role, SkipReason.STALE, detail))
        elif lead > max_lead:
            detail = (
                f"the window starts {lead / DAY:.2f} days before the epoch "
                f"{omm.epoch.isot} (max_lead {max_lead / DAY:g} days)"
            )
            skipped.append(SkippedObject(omm, role, SkipReason.FUTURE_EPOCH, detail))
        else:
            current.append(omm)
    return tuple(current)


def _primaries(
    omms: Sequence[Omm],
    volume: ScreeningVolume | VolumeRule,
    skipped: list[SkippedObject],
) -> tuple[Sgp4Propagator, list[ScreeningVolume]]:
    """SGP4 for the primaries it can be initialised with and that have a
    volume, and their volumes."""
    initialised, _, rejected = Sgp4Propagator.from_omms(omms)
    _skip_unsupported(
        [(omms[index], str(error)) for index, error in rejected],
        ScreeningRole.PRIMARY,
        skipped,
    )
    chosen: list[int] = []
    volumes: list[ScreeningVolume] = []
    for index, omm in enumerate(initialised.omms):
        try:
            volumes.append(volume_for(volume, omm))
        except KeepoutError as error:
            reason = SkipReason.NO_VOLUME
            skipped.append(
                SkippedObject(omm, ScreeningRole.PRIMARY, reason, str(error))
            )
            continue
        chosen.append(index)
    return initialised.subset(np.array(chosen, dtype=np.intp)), volumes


def _skip_unsupported(
    unsupported: Iterable[tuple[Omm, str]],
    role: ScreeningRole,
    skipped: list[SkippedObject],
) -> None:
    for omm, message in unsupported:
        skipped.append(SkippedObject(omm, role, SkipReason.UNSUPPORTED, message))


def _skip_failed(
    failures: Mapping[int, ObjectFailure],
    omms: Sequence[Omm],
    role: ScreeningRole,
    window: _Window,
    skipped: list[SkippedObject],
) -> set[int]:
    """List the objects SGP4 failed for; returns their positions."""
    for index, failure in failures.items():
        when = (window.start + failure.offset * u.s).utc.isot
        detail = f"{describe_error(failure.code)}, at {when}"
        reason = SkipReason.PROPAGATION_FAILED
        skipped.append(SkippedObject(omms[index], role, reason, detail))
    return set(failures)


def _run(parts: Sequence[_Part], workers: int) -> list[_PartResult]:
    """Screen the parts, in this process or in a pool of ``workers``."""
    if workers == 1 or len(parts) < 2:
        return [_screen_part(part) for part in parts]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(parts), mp_context=context) as pool:
        return list(pool.map(_screen_part, parts))


def _screen_part(part: _Part) -> _PartResult:
    """Initialise, filter and search one part of the catalog."""
    catalog, kept, rejected = Sgp4Propagator.from_omms(part.catalog)
    columns = part.columns[kept]
    lost = tuple((int(part.columns[index]), str(error)) for index, error in rejected)
    window = part.window
    if len(part.primaries) == 0 or len(catalog) == 0:
        return _PartResult(SearchResult((), {}, {}), lost)
    bounds = catalog.radius_bounds(window.start, window.duration)
    pairs = apogee_perigee_filter(part.primary_bounds, bounds, part.volumes)
    numbers = np.array([omm.catalog_number for omm in catalog.omms])
    pairs &= part.primary_numbers[:, None] != numbers[None, :]
    search = ApproachSearch(
        part.primaries,
        catalog,
        window.start,
        window.duration,
        part.volumes,
        pairs,
        step=window.step,
        primary_bounds=part.primary_bounds,
        catalog_bounds=bounds,
    )
    found = search.run(max_states=window.max_states)
    minima = [
        replace(minimum, secondary_index=int(columns[minimum.secondary_index]))
        for minimum in found.minima
    ]
    failures = {
        int(columns[index]): failure
        for index, failure in found.catalog_failures.items()
    }
    return _PartResult(
        SearchResult(tuple(minima), found.primary_failures, failures), lost
    )


def _approaches(
    minima: Sequence[RangeMinimum],
    primaries: Sequence[Omm],
    catalog: Sequence[Omm],
    start: Time,
    frame: RefFrame,
) -> list[CloseApproach]:
    """The minima as close approaches, their states (in ``frame``) in GCRF."""
    if not minima:
        return []
    offsets = np.array([minimum.offset for minimum in minima])
    times = (start + offsets * u.s).utc
    r: FloatArray = np.array([[m.r_primary, m.r_secondary] for m in minima])
    v: FloatArray = np.array([[m.v_primary, m.v_secondary] for m in minima])
    r_gcrf, v_gcrf = frame_to_gcrf(frame, times.reshape(-1, 1), r, v)
    approaches: list[CloseApproach] = []
    for i, minimum in enumerate(minima):
        tca = times[i]
        approaches.append(
            CloseApproach(
                primaries[minimum.primary_index].catalog_number,
                catalog[minimum.secondary_index].catalog_number,
                StateVector(tca, r_gcrf[i, 0], v_gcrf[i, 0], "GCRF"),
                StateVector(tca, r_gcrf[i, 1], v_gcrf[i, 1], "GCRF"),
            )
        )
    return approaches
