"""Screening: every close approach of some primaries with a catalog over a window.

:func:`screen` takes the primaries' and the catalog's element sets (OMM, SGP4
theory) and a window, and returns a
:class:`~keepout.screening.result.ScreeningResult`:

1. **Element sets.** An element set is left out when the window ends more
   than ``max_age`` after its epoch (stale) or starts more than ``max_lead``
   before it. Of the other element sets of one object, the newest is
   screened (exact copies are one, as in :func:`keepout.omm.deduplicate`):
   an object whose newest set is too far ahead is screened with an older
   one. An element set is also left out when SGP4 cannot be initialised
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

**After a burn.** ``adjust_primaries`` replaces the primaries' propagator
after step 1, Element sets: step 2, Pairs, and step 3, Search, see a
:class:`~keepout.propagation.base.Propagator` only, and an element set still
names the primary.

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

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

import astropy.units as u
import numpy as np
from astropy.time import Time

from keepout._defaults import SCREENING_DURATION
from keepout._validation import (
    require_at_least,
    require_non_negative,
    require_single_time,
)
from keepout._workers import process_pool, resolve_workers
from keepout.errors import KeepoutError
from keepout.frames import frame_to_gcrf
from keepout.omm import Omm, deduplicate
from keepout.propagation import (
    CHUNK_STATES,
    Propagator,
    RadiusBounds,
    Sgp4Propagator,
    describe_error,
)
from keepout.screening.filters import apogee_perigee_filter
from keepout.screening.result import (
    CloseApproach,
    ScreeningResult,
    ScreeningRole,
    SkippedObject,
    SkipReason,
)
from keepout.screening.search_result import (
    ObjectFailure,
    RangeMinimum,
    SearchResult,
)
from keepout.screening.tca import SEARCH_REACH, ApproachSearch
from keepout.screening.volume import (
    DEFAULT_VOLUME,
    ScreeningVolume,
    VolumeRule,
    volume_for,
)
from keepout.state import RefFrame, StateVector
from keepout.units import DAY

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    import numpy.typing as npt

    from keepout._types import FloatArray, IndexArray

__all__ = ["DEFAULT_DURATION", "MAX_AGE", "MAX_LEAD", "screen"]

DEFAULT_DURATION: Final = SCREENING_DURATION
"""Length of the window unless told otherwise [s]: 7 days, as in SOCRATES."""

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
    workers: int | None = 1,
    max_age: float = MAX_AGE,
    max_lead: float = MAX_LEAD,
    max_states: int = CHUNK_STATES,
    adjust_primaries: Callable[[Propagator], Propagator] | None = None,
) -> ScreeningResult:
    """Every close approach of the primaries with the catalog in a window.

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
        Number of processes to screen in; 1 screens in this process,
        ``None`` uses one per CPU, at most 4.
    max_age
        Longest time [s] from an element epoch to the end of the window;
        element sets older than that are left out as stale.
    max_lead
        Longest time [s] from the start of the window to a later element
        epoch; element sets further ahead are left out.
    max_states
        Most states (objects x times) a process propagates at once; bounds
        the memory (about 49 bytes per state).
    adjust_primaries
        Called once, in this process, with the SGP4 propagator of the primaries
        that are screened, when there are any; what it returns (the same
        objects in order, in the catalog's frame, TEME) is screened in its
        place, as a :class:`~keepout.propagation.ManeuveredPropagator` for a
        burn. It must give states to
        :data:`~keepout.screening.tca.SEARCH_REACH` past the
        end of the window, where the search reads them, which is checked
        once. ``None`` screens SGP4.

    Returns
    -------
    ScreeningResult
        The close approaches in chronological order, the element sets left
        out with their reasons, the window and each primary's volume.

    Raises
    ------
    ValueError
        If ``start`` is not one time, or ``duration``, ``max_age``,
        ``max_lead``, ``workers`` or ``max_states`` is out of range,
        or the propagator of ``adjust_primaries`` changes the number of
        primaries or their frame or stops short of the states the search
        reads.
    TypeError
        If the volume rule returns something other than a
        :class:`~keepout.screening.volume.ScreeningVolume`.
    PropagationError
        If ``adjust_primaries`` raises it, as a
        :class:`~keepout.propagation.ManeuveredPropagator` does for an orbit
        it cannot integrate.

    Examples
    --------
    ```python
    from astropy.time import Time

    from keepout import omm, screen

    # A sample catalog in the repository.
    catalog = omm.load("tests/screening/data/celestrak-public.csv.gz")
    iss = [item for item in catalog if item.catalog_number == 25544]
    start = Time("2026-10-06T00:00:00", scale="utc")
    result = screen(iss, catalog, start, duration=86400.0)  # one day, 5 km sphere
    for approach in result.approaches:
        distance = round(approach.range_at_tca)  # [m]
        print(approach.secondary_number, approach.tca.isot, distance)
    # 58180 2026-10-06T05:12:33.772 4633
    # 58180 2026-10-06T12:10:37.724 4902
    ```
    """
    prepared = prepare_catalog(
        catalog,
        start,
        duration,
        workers=workers,
        max_age=max_age,
        max_lead=max_lead,
        max_states=max_states,
    )
    return prepared.screen(primaries, volume, adjust_primaries=adjust_primaries)


@dataclass(frozen=True, slots=True)
class _SearchSettings:
    """The window and the search's settings, shared by every part."""

    start: Time
    duration: float
    max_states: int


@dataclass(frozen=True, eq=False, slots=True)
class _CatalogPart:
    """Element sets of a catalog initialised for SGP4, with their radius
    bounds (``None`` when nothing is screened against them)."""

    catalog: Sgp4Propagator
    columns: IndexArray
    """Position of each object of ``catalog`` in the whole catalog."""
    unsupported: tuple[tuple[int, str], ...]
    """Position in the whole catalog of each element set SGP4 cannot be
    initialised with, and why."""
    bounds: RadiusBounds | None
    numbers: npt.NDArray[np.int64]
    """Catalog number of each object of ``catalog``."""


@dataclass(frozen=True, eq=False, slots=True, kw_only=True)
class _Part:
    """The primaries and one part of the catalog, to screen in one process."""

    settings: _SearchSettings
    primaries: Propagator
    primary_omms: tuple[Omm, ...]
    """Element set of each of ``primaries``."""
    primary_bounds: RadiusBounds
    volumes: tuple[ScreeningVolume, ...]
    catalog: tuple[Omm, ...] | _CatalogPart
    """The part's element sets, initialised in its process, or already
    initialised."""
    columns: IndexArray
    """Position of each of the element sets in the whole catalog."""


@dataclass(frozen=True, eq=False, slots=True)
class _PartResult:
    """What one part found; catalog positions are in the whole catalog."""

    search: SearchResult
    unsupported: tuple[tuple[int, str], ...]


@dataclass(frozen=True, eq=False, slots=True, kw_only=True)
class PreparedCatalog:
    """A catalog made ready to screen against over one window, by
    :func:`prepare_catalog`: the element set of each object chosen and, to
    screen in this process, initialised for SGP4 with its radius bounds.

    Screening several sets of primaries against it (the primary with and
    without a burn, in :func:`keepout.maneuver.rescreen`) does that work
    once; each result is the one :func:`screen` gives.
    """

    settings: _SearchSettings
    max_age: float
    max_lead: float
    processes: int
    elements: tuple[Omm, ...]
    """The element set screened of each object."""
    skipped: tuple[SkippedObject, ...]
    """The element sets left out before anything is initialised."""
    initialised: _CatalogPart | None
    """All of ``elements``, initialised, when screened in this process."""

    @property
    def by_number(self) -> dict[int, Omm]:
        """The element set screened of each object, by catalog number."""
        return {omm.catalog_number: omm for omm in self.elements}

    def screen(
        self,
        primaries: Iterable[Omm],
        volume: ScreeningVolume | VolumeRule = DEFAULT_VOLUME,
        *,
        adjust_primaries: Callable[[Propagator], Propagator] | None = None,
    ) -> ScreeningResult:
        """:func:`screen` of ``primaries`` against the catalog.

        Raises
        ------
        ValueError
            As :func:`screen`, for ``adjust_primaries``.
        TypeError
            As :func:`screen`, for the volume rule.
        PropagationError
            If ``adjust_primaries`` raises it.
        """
        settings = self.settings
        current_primaries, duplicate_primaries, out_of_window_primaries = (
            _screened_sets(
                primaries, ScreeningRole.PRIMARY, settings, self.max_age, self.max_lead
            )
        )
        sgp4_primaries, volumes, skipped_unusable_primaries = _primaries(
            current_primaries, volume
        )
        primary_omms = sgp4_primaries.omms
        primary_propagator = _adjusted(sgp4_primaries, adjust_primaries, settings)
        skipped_before_search = [
            *duplicate_primaries,
            *out_of_window_primaries,
            *self.skipped,
            *skipped_unusable_primaries,
        ]
        return _merge_part_results(
            _run(_parts(primary_propagator, primary_omms, volumes, self)),
            primary_propagator,
            primary_omms,
            volumes,
            self.elements,
            settings,
            skipped_before_search,
        )


def prepare_catalog(
    catalog: Iterable[Omm],
    start: Time,
    duration: float = DEFAULT_DURATION,
    *,
    workers: int | None = 1,
    max_age: float = MAX_AGE,
    max_lead: float = MAX_LEAD,
    max_states: int = CHUNK_STATES,
) -> PreparedCatalog:
    """A catalog made ready to screen against, as :func:`screen` takes it.

    Raises
    ------
    ValueError
        As :func:`screen`, for the window and the settings.
    """
    require_single_time(start, "start")
    require_non_negative(duration, "duration")
    require_non_negative(max_age, "max_age")
    require_non_negative(max_lead, "max_lead")
    processes = resolve_workers(workers)
    require_at_least(max_states, 1, "max_states")
    settings = _SearchSettings(start, float(duration), max_states)
    current, duplicates, out_of_window = _screened_sets(
        catalog, ScreeningRole.CATALOG, settings, max_age, max_lead
    )
    initialised = None
    if processes == 1 and current:
        initialised = _initialised(
            current, np.arange(len(current)), settings, bounded=True
        )
    return PreparedCatalog(
        settings=settings,
        max_age=float(max_age),
        max_lead=float(max_lead),
        processes=processes,
        elements=current,
        skipped=(*duplicates, *out_of_window),
        initialised=initialised,
    )


def _screened_sets(
    omms: Iterable[Omm],
    role: ScreeningRole,
    settings: _SearchSettings,
    max_age: float,
    max_lead: float,
) -> tuple[tuple[Omm, ...], list[SkippedObject], list[SkippedObject]]:
    """The element set screened of each object, the newest of those whose
    epochs are close enough to the window (the first given of a tie); the
    other sets of the objects screened and the older sets of the others,
    left out as duplicates; and the newest set of each of the others, left
    out by its epoch.

    Exact copies are one set (:func:`keepout.omm.deduplicate`).
    """
    elements = tuple(omms)
    unique = deduplicate(elements)
    position: dict[int, int] = {}
    for index, omm in enumerate(elements):
        position.setdefault(id(omm), index)
    distinct = sorted((*unique.kept, *unique.dropped), key=lambda o: position[id(o)])
    outside = _outside_window(distinct, role, settings, max_age, max_lead)
    chosen: dict[int, Omm] = {}
    for omm in distinct:
        best = chosen.get(omm.catalog_number)
        if id(omm) not in outside and (best is None or omm.epoch > best.epoch):
            chosen[omm.catalog_number] = omm
    newest = unique.by_number
    duplicates: list[SkippedObject] = []
    for omm in distinct:
        number = omm.catalog_number
        used = chosen.get(number)
        if used is not None and omm is not used:
            detail = f"the element set of epoch {used.epoch.isot} is used"
        elif used is None and omm is not newest[number]:
            detail = (
                f"the newest element set, of epoch {newest[number].epoch.isot}, "
                "is left out too"
            )
        else:
            continue
        duplicates.append(
            SkippedObject(
                omm, role, SkipReason.DUPLICATE, f"epoch {omm.epoch.isot}; {detail}"
            )
        )
    current = tuple(
        chosen[omm.catalog_number]
        for omm in unique.kept
        if omm.catalog_number in chosen
    )
    out_of_window = [
        outside[id(omm)] for omm in unique.kept if omm.catalog_number not in chosen
    ]
    return current, duplicates, out_of_window


def _outside_window(
    omms: Sequence[Omm],
    role: ScreeningRole,
    settings: _SearchSettings,
    max_age: float,
    max_lead: float,
) -> dict[int, SkippedObject]:
    """The element sets whose epochs are too far from the window (stale, or
    too far ahead), left out, by ``id``."""
    if not omms:
        return {}
    epochs = Time(
        [omm.epoch.jd1 for omm in omms],
        [omm.epoch.jd2 for omm in omms],
        format="jd",
        scale="utc",
    )
    elapsed_at_start = (settings.start - epochs).to_value(u.s)
    outside: dict[int, SkippedObject] = {}
    for omm, elapsed in zip(omms, elapsed_at_start.tolist(), strict=True):
        age, lead = elapsed + settings.duration, -elapsed
        if age > max_age:
            detail = (
                f"the window ends {age / DAY:.2f} days after the epoch "
                f"{omm.epoch.isot} (max_age {max_age / DAY:g} days)"
            )
            outside[id(omm)] = SkippedObject(omm, role, SkipReason.STALE, detail)
        elif lead > max_lead:
            detail = (
                f"the window starts {lead / DAY:.2f} days before the epoch "
                f"{omm.epoch.isot} (max_lead {max_lead / DAY:g} days)"
            )
            outside[id(omm)] = SkippedObject(omm, role, SkipReason.FUTURE_EPOCH, detail)
    return outside


def _primaries(
    omms: Sequence[Omm],
    volume: ScreeningVolume | VolumeRule,
) -> tuple[Sgp4Propagator, list[ScreeningVolume], list[SkippedObject]]:
    """SGP4 for the primaries it can be initialised with and that have a
    volume, their volumes, and the primaries left out."""
    initialised, _, rejected = Sgp4Propagator.from_omms(omms)
    skipped = _unsupported(
        [(omms[index], str(error)) for index, error in rejected],
        ScreeningRole.PRIMARY,
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
    subset = initialised.subset(np.array(chosen, dtype=np.intp))
    return subset, volumes, skipped


def _adjusted(
    primaries: Sgp4Propagator,
    adjust: Callable[[Propagator], Propagator] | None,
    settings: _SearchSettings,
) -> Propagator:
    """The propagator to screen ``primaries`` with.

    The search would reach past the window only when a minimum lies near its
    end, so a propagator that cannot is refused here, on any data.

    Raises
    ------
    ValueError
        If ``adjust`` changes the number of primaries or their frame, or
        gives no states as far past the window as the search reads them.
    """
    if adjust is None or len(primaries) == 0:
        return primaries
    adjusted = adjust(primaries)
    if len(adjusted) != len(primaries):
        raise ValueError(
            f"adjust_primaries must keep the {len(primaries)} primaries, "
            f"got {len(adjusted)}"
        )
    if adjusted.frame != primaries.frame:
        raise ValueError(
            f"adjust_primaries must keep the frame {primaries.frame}, "
            f"got {adjusted.frame}"
        )
    try:
        adjusted.propagate(settings.start, settings.duration + SEARCH_REACH)
    except ValueError as error:
        raise ValueError(
            f"adjust_primaries must give states to {SEARCH_REACH} s past the end of "
            f"the window: {error}"
        ) from error
    return adjusted


def _unsupported(
    unsupported: Iterable[tuple[Omm, str]], role: ScreeningRole
) -> list[SkippedObject]:
    """The element sets SGP4 cannot be initialised with, left out."""
    return [
        SkippedObject(omm, role, SkipReason.UNSUPPORTED, message)
        for omm, message in unsupported
    ]


def _parts(
    primaries: Propagator,
    primary_omms: tuple[Omm, ...],
    volumes: Sequence[ScreeningVolume],
    prepared: PreparedCatalog,
) -> list[_Part]:
    """The primaries with the catalog dealt into at most ``processes`` parts:
    object ``i`` to part ``i mod count``, none for an empty catalog; one part,
    already initialised, in this process."""
    settings = prepared.settings
    bounds = primaries.radius_bounds(settings.start, settings.duration)
    catalog = prepared.elements

    def part_of(items: tuple[Omm, ...] | _CatalogPart, columns: IndexArray) -> _Part:
        return _Part(
            settings=settings,
            primaries=primaries,
            primary_omms=primary_omms,
            primary_bounds=bounds,
            volumes=tuple(volumes),
            catalog=items,
            columns=columns,
        )

    if prepared.initialised is not None:
        return [part_of(prepared.initialised, np.arange(len(catalog)))]
    count = min(prepared.processes, len(catalog))
    return [
        part_of(catalog[part::count], np.arange(part, len(catalog), count))
        for part in range(count)
    ]


def _run(parts: Sequence[_Part]) -> list[_PartResult]:
    """The results of the parts, screened in this process or in a process
    each."""
    if len(parts) < 2:
        return [_screen_part(part) for part in parts]
    with process_pool(len(parts)) as pool:
        return list(pool.map(_screen_part, parts))


def _initialised(
    omms: Sequence[Omm],
    columns: IndexArray,
    settings: _SearchSettings,
    *,
    bounded: bool,
) -> _CatalogPart:
    """Element sets (at ``columns`` of the catalog) initialised for SGP4, with
    their radius bounds over the window if ``bounded``."""
    catalog, kept, rejected = Sgp4Propagator.from_omms(omms)
    unsupported = tuple((int(columns[index]), str(error)) for index, error in rejected)
    bounds = (
        catalog.radius_bounds(settings.start, settings.duration)
        if bounded and len(catalog) > 0
        else None
    )
    numbers = np.array([omm.catalog_number for omm in catalog.omms], dtype=np.int64)
    return _CatalogPart(catalog, columns[kept], unsupported, bounds, numbers)


def _screen_part(part: _Part) -> _PartResult:
    """What one part finds: its catalog initialised, filtered and searched."""
    settings = part.settings
    prepared = part.catalog
    if not isinstance(prepared, _CatalogPart):
        bounded = len(part.primaries) > 0
        prepared = _initialised(prepared, part.columns, settings, bounded=bounded)
    catalog, columns, bounds = prepared.catalog, prepared.columns, prepared.bounds
    if len(part.primaries) == 0 or bounds is None:
        return _PartResult(SearchResult.empty(), prepared.unsupported)
    pairs = apogee_perigee_filter(part.primary_bounds, bounds, part.volumes)
    primary_numbers = np.array(
        [omm.catalog_number for omm in part.primary_omms], dtype=np.int64
    )
    pairs &= primary_numbers[:, None] != prepared.numbers[None, :]
    search = ApproachSearch(
        part.primaries,
        catalog,
        settings.start,
        settings.duration,
        part.volumes,
        pairs,
        primary_bounds=part.primary_bounds,
        catalog_bounds=bounds,
    )
    search_result = search.run(max_states=settings.max_states)
    minima = [
        replace(minimum, secondary_index=int(columns[minimum.secondary_index]))
        for minimum in search_result.minima
    ]
    failures = {
        int(columns[index]): failure
        for index, failure in search_result.catalog_failures.items()
    }
    return _PartResult(
        SearchResult(tuple(minima), search_result.primary_failures, failures),
        prepared.unsupported,
    )


def _merge_part_results(
    results: Sequence[_PartResult],
    primaries: Propagator,
    primary_omms: Sequence[Omm],
    volumes: Sequence[ScreeningVolume],
    catalog: Sequence[Omm],
    settings: _SearchSettings,
    skipped_before_search: Sequence[SkippedObject],
) -> ScreeningResult:
    """The parts' results merged into the screening's result.

    ``skipped_before_search`` are the element sets left out before the
    search; to them are added the catalog's element sets SGP4 cannot be
    initialised with and the objects SGP4 failed for, whose minima are
    dropped.
    """
    unsupported_catalog = [
        (catalog[index], message)
        for result in results
        for index, message in result.unsupported
    ]
    search = SearchResult.merge(result.search for result in results)
    skipped = [
        *skipped_before_search,
        *_unsupported(unsupported_catalog, ScreeningRole.CATALOG),
        *_failed(
            search.primary_failures, primary_omms, ScreeningRole.PRIMARY, settings
        ),
        *_failed(search.catalog_failures, catalog, ScreeningRole.CATALOG, settings),
    ]
    failed_primaries = set(search.primary_failures)
    failed_catalog = set(search.catalog_failures)
    minima = [
        minimum
        for minimum in search.minima
        if minimum.primary_index not in failed_primaries
        and minimum.secondary_index not in failed_catalog
    ]
    screened_volumes = {
        omm.catalog_number: volumes[index]
        for index, omm in enumerate(primary_omms)
        if index not in failed_primaries
    }
    approaches = _approaches(
        minima, primary_omms, catalog, settings.start, primaries.frame
    )
    return ScreeningResult(
        approaches=tuple(approaches),
        skipped=tuple(skipped),
        start=settings.start,
        duration=settings.duration,
        volumes=screened_volumes,
        catalog_count=len(catalog) - len(unsupported_catalog) - len(failed_catalog),
    )


def _failed(
    failures: Mapping[int, ObjectFailure],
    omms: Sequence[Omm],
    role: ScreeningRole,
    settings: _SearchSettings,
) -> list[SkippedObject]:
    """The objects SGP4 failed for (by position in ``omms``), left out."""
    skipped: list[SkippedObject] = []
    for index, failure in failures.items():
        when = (settings.start + failure.offset * u.s).utc.isot
        detail = f"{describe_error(failure.code)}, at {when}"
        reason = SkipReason.PROPAGATION_FAILED
        skipped.append(SkippedObject(omms[index], role, reason, detail))
    return skipped


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
