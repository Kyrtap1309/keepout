"""``keepout jobs``: the background jobs of the store's queue, listed and
cancelled.

``keepout.store`` is imported in the functions, never here: ``keepout
--help`` does not load it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, TypeAlias

import typer

from keepout.cli.options import JsonFlag
from keepout.cli.output import JsonOut, isot, minute, show, table
from keepout.cli.store import StorePath, importing_store, open_cli_store

if TYPE_CHECKING:
    from astropy.time import Time

    from keepout.store.jobs import Job

__all__ = ["jobs_cancel", "jobs_list"]

JobId: TypeAlias = Annotated[
    int,
    typer.Argument(metavar="ID", show_default=False, help="The job's id."),
]
"""The id of the job to cancel."""


def _time(time: Time | None) -> str | None:
    """``time`` for JSON, ``None`` kept."""
    return None if time is None else isot(time)


def _job_json(job: Job) -> dict[str, JsonOut]:
    """A job as JSON values."""
    return {
        "job": job.id,
        "event": job.event_id,
        "cdm": job.message_id,
        "status": job.status.value,
        "attempt": job.attempt,
        "worker": job.worker,
        "created": isot(job.created_at),
        "started": _time(job.started_at),
        "finished": _time(job.finished_at),
        "error": job.error,
    }


def jobs_list(*, store: StorePath, as_json: JsonFlag = False) -> None:
    """List the store's jobs, the newest first.

    A job is queued, running, succeeded, failed or cancelled; a failed job
    gives its error.
    """
    with importing_store():
        from keepout.store import JobQueue

    with open_cli_store(store, create=False) as opened:
        jobs = JobQueue(opened).jobs()
    payload: dict[str, JsonOut] = {
        "path": str(store),
        "jobs": [_job_json(job) for job in jobs],
    }
    if not jobs:
        show(payload, f"No jobs in {store}.", as_json=as_json)
        return
    rows = table(f"Jobs in {store}")
    for column in ("Job", "Event", "CDM", "Status", "Attempt", "Created (UTC)"):
        rows.add_column(column, no_wrap=column != "CDM")
    rows.add_column("Error")
    for job in jobs:
        rows.add_row(
            str(job.id),
            job.event_id,
            job.message_id,
            job.status.value,
            str(job.attempt),
            minute(job.created_at),
            job.error or "",
        )
    show(payload, rows, as_json=as_json)


def jobs_cancel(job_id: JobId, *, store: StorePath, as_json: JsonFlag = False) -> None:
    """Cancel a queued or running job.

    A running job's worker stops computing it when it next renews its
    lease.
    """
    with importing_store():
        from keepout.store import JobQueue

    with open_cli_store(store, create=False) as opened:
        job = JobQueue(opened).cancel(job_id)
    show(
        {"job": job.id, "event": job.event_id, "status": job.status.value},
        f"Cancelled job {job.id} (event {job.event_id}).",
        as_json=as_json,
    )
