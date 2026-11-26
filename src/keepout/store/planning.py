"""Manoeuvre plans of stored events, and the computations of queued jobs.

:func:`plan_stored` plans a stored event as it was at one of its CDMs. Its
burns are re-screened against the store's copy of that CDM's catalog, if
the CDM was made from one, or against a catalog given.
:func:`cached_or_planned` gives the plan the store caches: the cached one,
else one planned and cached. :func:`run_maneuver_job` is the same plan as the
computation of a queued job (:mod:`keepout.store.worker`), and
:func:`run_refresh_job` the computation of the Pc a stored event lacks
(:func:`compute_missing`), in batches its worker writes as they come.
:func:`run_job` runs either, by the job's kind.

The numerical libraries are imported when a plan is made, not with this
module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, assert_never

from sqlalchemy import select

from keepout.errors import StoreError
from keepout.settings import plan_arguments
from keepout.store import models
from keepout.store.codec import observation_to_json
from keepout.store.computing import (
    batches,
    lacking_hard_body_radius,
    observation_of,
    parsed_cdm,
    refuse_lacking_hard_body_radius,
)
from keepout.store.db import open_store
from keepout.store.models import JobKind
from keepout.store.queries import cdms_of, oldest_first, pinned_in, require_events

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from keepout._types import JsonValue
    from keepout.maneuver import BurnGrid, ManeuverOutcome
    from keepout.omm import Omm
    from keepout.risk import PcObservation
    from keepout.settings import Settings
    from keepout.store.jobs import Job
    from keepout.store.repo import Store

__all__ = [
    "CachedPlan",
    "cached_or_planned",
    "compute_missing",
    "plan_stored",
    "run_job",
    "run_maneuver_job",
    "run_refresh_job",
]


def plan_stored(
    store: Store,
    event_id: str,
    settings: Settings,
    *,
    message_id: str | None = None,
    workers: int | None,
    grid: BurnGrid | None = None,
    burn: tuple[float, float] | None = None,
    catalog: Sequence[Omm] | None = None,
    catalog_label: str | None = None,
) -> ManeuverOutcome:
    """The outcome of a stored event's burns, by
    :func:`~keepout.maneuver.plan_event`.

    The event and its catalog are read at one CDM, in one read transaction.
    A CDM ingested meanwhile is not planned.

    Parameters
    ----------
    store
        The open store that holds the event.
    event_id
        The store's id of the event.
    settings
        The operator's settings of the plan.
    message_id
        The ``MESSAGE_ID`` of the CDM to plan at: the event without the CDMs
        after it. ``None``: the event's latest CDM.
    workers
        Processes of each re-screen, as for
        :func:`~keepout.maneuver.plan_event`.
    grid
        The burns tried; ``None``: the default grid of
        :func:`~keepout.maneuver.plan_maneuver`.
    burn
        One burn to evaluate instead of the grid, as for
        :func:`~keepout.maneuver.plan_event`.
    catalog
        The element sets to re-screen each burn against instead of the
        stored copy, which is then not read; ``None``: the stored copy, if
        any.
    catalog_label
        What ``catalog`` is, for people; given with ``catalog`` and only then.

    Returns
    -------
    ManeuverOutcome
        The outcome. Re-screened against the stored copy, its catalog label
        is ``stored copy of`` the catalog's.

    Raises
    ------
    StoreError
        If there is no such event, ``message_id`` is not a CDM of it, or the
        database cannot be used.
    SourceUnavailable
        If the copy of the catalog is gone; the message names it.
    ChecksumMismatch
        If the copy of the catalog changed; the message names it.
    KeepoutError
        As :func:`~keepout.maneuver.plan_event`.
    ValueError
        As :func:`~keepout.maneuver.plan_event`.
    """
    from keepout.maneuver import DEFAULT_BURN_GRID, outcome
    from keepout.sources.catalog import load_reference

    with store.read() as session:
        event, reference = pinned_in(session, store.path.parent, event_id, message_id)
    if catalog is None and reference is not None:
        catalog = load_reference(reference)
        catalog_label = f"stored copy of {reference.label}"
    return outcome.plan_event(
        event,
        **plan_arguments(settings),
        burn=burn,
        grid=DEFAULT_BURN_GRID if grid is None else grid,
        catalog=catalog,
        catalog_label=catalog_label,
        workers=workers,
    )


class CachedPlan(NamedTuple):
    """The plan :func:`cached_or_planned` gives."""

    outcome: ManeuverOutcome
    unsaved: StoreError | None
    """Why a plan made now could not be cached; ``None`` if it was, or was
    read from the cache."""


def cached_or_planned(
    store: Store, event_id: str, settings: Settings, *, workers: int | None
) -> CachedPlan:
    """The plan of a stored event at its latest CDM, against its stored
    catalog: the cached one, else :func:`plan_stored`'s, then cached.

    A plan that cannot be cached (e.g. the database locked) is given all
    the same, with the reason.

    Parameters
    ----------
    store
        The open store that holds the event.
    event_id
        The store's id of the event.
    settings
        The operator's settings of the plan.
    workers
        Processes of each re-screen, as for
        :func:`~keepout.maneuver.plan_event`.

    Returns
    -------
    CachedPlan
        The plan, and why it was not cached.

    Raises
    ------
    StoreError
        If there is no such event, or the cache cannot be read.
    SourceUnavailable
        If the copy of the catalog is gone; the message names it.
    ChecksumMismatch
        If the copy of the catalog changed; the message names it.
    KeepoutError
        As :func:`~keepout.maneuver.plan_event`.
    ValueError
        As :func:`~keepout.maneuver.plan_event`, or if a settings file
        cannot hold ``settings`` (:func:`~keepout.store.codec.maneuver_key`).
    """
    cached = store.maneuver(event_id, settings)
    if cached is not None:
        return CachedPlan(cached, None)
    planned = plan_stored(store, event_id, settings, workers=workers)
    try:
        # Planned at the event's latest CDM then: cached on it.
        store.save_maneuver(planned.event.latest.message_id, settings, planned)
    except StoreError as exc:
        return CachedPlan(planned, exc)
    return CachedPlan(planned, None)


def run_maneuver_job(
    path: str, job: Job, workers: int | None, *, grid: BurnGrid | None = None
) -> dict[str, JsonValue]:
    """The manoeuvre plan of a job, as JSON: :func:`plan_stored` at the
    job's CDM.

    Parameters
    ----------
    path
        The store's database file.
    job
        The job, claimed from the store's queue; its event, CDM and settings are
        planned.
    workers
        Processes of each re-screen, as for
        :func:`~keepout.maneuver.plan_event`.
    grid
        The burns tried; ``None``: the default grid.

    Returns
    -------
    dict
        The outcome's :func:`~keepout.store.maneuver_codec.outcome_to_json`.

    Raises
    ------
    StoreError
        If the store cannot be opened or no longer holds the job's CDM.
    SourceUnavailable
        If the copy of the catalog is gone; the message names it.
    ChecksumMismatch
        If the copy of the catalog changed; the message names it.
    KeepoutError
        As :func:`~keepout.maneuver.plan_event`.
    ValueError
        As :func:`~keepout.maneuver.plan_event`.
    """
    from keepout.store.maneuver_codec import outcome_to_json

    with open_store(path, create=False) as store:
        planned = plan_stored(
            store,
            job.event_id,
            job.settings,
            message_id=job.message_id,
            workers=workers,
            grid=grid,
        )
    return outcome_to_json(planned)


def compute_missing(
    store: Store, event_id: str, settings: Settings, *, key: str, until: str
) -> Iterator[list[PcObservation]]:
    """The Pc observations a stored event lacks under ``key``, computed with
    ``settings`` as they are iterated, in batches
    (:func:`~keepout.store.computing.batches`); none is written.

    The texts of the CDMs up to ``until`` (its ``MESSAGE_ID``) without a
    result under ``key`` are read in one read transaction, and checked for
    ``HBR`` before any Pc is computed, outside it.

    Parameters
    ----------
    store
        The open store.
    event_id
        The store's id of the event.
    settings
        How Pc is computed.
    key
        The key the observations are cached under: the
        :func:`~keepout.store.codec.result_key` of ``settings``.
    until
        The ``MESSAGE_ID`` of the last CDM, in (``CREATION_DATE``,
        ``MESSAGE_ID``) order.

    Yields
    ------
    list of PcObservation
        The observations of the CDMs without one, oldest first.

    Raises
    ------
    StoreError
        If there is no such event, or the database cannot be used.
    HardBodyRadiusError
        If the settings give no hard-body radius and a CDM lacks ``HBR``
        for an object (as :meth:`~keepout.store.Store.refresh` of given
        events).
    CdmParseError
        If a stored CDM no longer parses (a store edited by hand).
    """
    with store.read() as session:
        require_events(session, [event_id])
        query = oldest_first(
            select(models.Cdm.content), cdms_of(event_id, until=until), lacking=key
        )
        texts = session.scalars(query).all()
    cdms = [parsed_cdm(text) for text in texts]
    refuse_lacking_hard_body_radius(lacking_hard_body_radius(settings, cdms))
    yield from batches(observation_of(event_id, cdm, settings) for cdm in cdms)


def run_refresh_job(
    path: str,
    job: Job,
    workers: int | None,  # noqa: ARG001 (one process computes a Pc)
) -> Iterator[dict[str, JsonValue]]:
    """The Pc observations a job's event lacks, as JSON, in batches: those of
    its CDMs up to the job's, computed with the job's settings
    (:func:`compute_missing`), not written.

    The worker that runs it writes each batch as it comes
    (:meth:`~keepout.store.jobs.JobQueue.save_partial`), but the Pc cached
    meanwhile: a refresh stopped halfway keeps what it computed.

    Parameters
    ----------
    path
        The store's database file.
    job
        The job, claimed from the store's queue: a refresh.
    workers
        Not used: the Pc of a CDM is computed in one process.

    Yields
    ------
    dict
        ``{"observations": {MESSAGE_ID: observation}}``, each observation's
        :func:`~keepout.store.codec.observation_to_json`, oldest CDM first;
        one batch at least, empty if no Pc was missing.

    Raises
    ------
    StoreError
        If the store cannot be opened or no longer holds the job's event.
    HardBodyRadiusError
        If the settings give no hard-body radius and a CDM lacks ``HBR``
        (as :meth:`~keepout.store.Store.refresh` of given events).
    CdmParseError
        If a stored CDM no longer parses (a store edited by hand).
    """
    with open_store(path, create=False) as store:
        computed = compute_missing(
            store, job.event_id, job.settings, key=job.key, until=job.message_id
        )
        empty = True
        for batch in computed:
            observations: dict[str, JsonValue] = {
                observation.message_id: observation_to_json(observation)
                for observation in batch
                if observation.message_id is not None
            }
            empty = False
            yield {"observations": observations}
        if empty:  # nothing was missing: a result all the same
            yield {"observations": {}}


def run_job(
    path: str, job: Job, workers: int | None
) -> dict[str, JsonValue] | Iterator[dict[str, JsonValue]]:
    """The computation of a job of either kind, as JSON: a plan
    (:func:`run_maneuver_job`) or the missing Pc, in batches
    (:func:`run_refresh_job`).

    Parameters
    ----------
    path
        The store's database file.
    job
        The job, claimed from the store's queue.
    workers
        Processes of each re-screen of a plan.

    Returns
    -------
    dict or iterator of dict
        The result of the job's computation, or its batches.

    Raises
    ------
    StoreError
        As :func:`run_maneuver_job` and :func:`run_refresh_job`.
    KeepoutError
        As :func:`run_maneuver_job` and :func:`run_refresh_job`.
    ValueError
        As :func:`run_maneuver_job`.
    """
    match job.kind:
        case JobKind.MANEUVER:
            return run_maneuver_job(path, job, workers)
        case JobKind.REFRESH:
            return run_refresh_job(path, job, workers)
        case _:
            assert_never(job.kind)
