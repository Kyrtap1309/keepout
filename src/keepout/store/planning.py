"""Manoeuvre plans of stored events, and the computations of queued jobs.

:func:`plan_stored` plans a stored event as it was at one of its CDMs. Its
burns are re-screened against the store's copy of that CDM's catalog, if
the CDM was made from one. :func:`run_maneuver_job` is the same plan as the
computation of a queued job (:mod:`keepout.store.worker`), and
:func:`run_refresh_job` the computation of the Pc a stored event lacks.
:func:`run_job` runs either, by the job's kind.

The numerical libraries are imported when a plan is made, not with this
module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from keepout.settings import plan_arguments
from keepout.store.codec import observation_to_json
from keepout.store.db import open_store
from keepout.store.models import JobKind
from keepout.store.repo import compute_missing, pinned_in

if TYPE_CHECKING:
    from keepout._types import JsonValue
    from keepout.maneuver import BurnGrid, ManeuverOutcome
    from keepout.settings import Settings
    from keepout.store.jobs import Job
    from keepout.store.repo import Store

__all__ = ["plan_stored", "run_job", "run_maneuver_job", "run_refresh_job"]


def plan_stored(
    store: Store,
    event_id: str,
    settings: Settings,
    *,
    message_id: str | None = None,
    workers: int | None,
    grid: BurnGrid | None = None,
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

    Returns
    -------
    ManeuverOutcome
        The outcome. Its catalog label is ``stored copy of`` the catalog's.

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
    if reference is None:
        catalog, label = None, None
    else:
        catalog, label = load_reference(reference), f"stored copy of {reference.label}"
    return outcome.plan_event(
        event,
        **plan_arguments(settings),
        grid=DEFAULT_BURN_GRID if grid is None else grid,
        catalog=catalog,
        catalog_label=label,
        workers=workers,
    )


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


def run_refresh_job(
    path: str,
    job: Job,
    workers: int | None,  # noqa: ARG001 (one process computes a Pc)
) -> dict[str, JsonValue]:
    """The Pc observations a job's event lacks, as JSON: those of its CDMs up
    to the job's, computed with the job's settings, not written.

    The worker that runs it writes them
    (:meth:`~keepout.store.jobs.JobQueue.finish`), but those cached
    meanwhile.

    Parameters
    ----------
    path
        The store's database file.
    job
        The job, claimed from the store's queue: a refresh.
    workers
        Not used: the Pc of a CDM is computed in one process.

    Returns
    -------
    dict
        ``{"observations": {MESSAGE_ID: observation}}``, each observation's
        :func:`~keepout.store.codec.observation_to_json`, oldest CDM first.

    Raises
    ------
    StoreError
        If the store cannot be opened or no longer holds the job's event or
        CDM.
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
    observations: dict[str, JsonValue] = {
        message_id: observation_to_json(observation)
        for message_id, observation in computed.items()
    }
    return {"observations": observations}


def run_job(path: str, job: Job, workers: int | None) -> dict[str, JsonValue]:
    """The computation of a job of either kind, as JSON: a plan
    (:func:`run_maneuver_job`) or the missing Pc (:func:`run_refresh_job`).

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
    dict
        The result of the job's computation.

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
