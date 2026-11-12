"""The conjunction store: events, CDMs and cached results in a SQLite
database (the ``store`` extra: SQLAlchemy and Alembic).

:func:`open_store` opens one, creating a new file at the current schema;
:func:`upgrade` brings a file made by an older keepout up to it. A
:class:`Store` ingests CDMs (:class:`CdmInput`) into events and caches
their Pc (:class:`RefreshReport`). A :class:`JobQueue` holds the manoeuvre
plans queued for a worker (:class:`Job`, :class:`Submission`,
:class:`SubmissionOutcome`).
"""

from keepout.store.db import open_store, upgrade
from keepout.store.jobs import Job, JobQueue, JobStatus, Submission, SubmissionOutcome
from keepout.store.models import EventId, MessageId
from keepout.store.repo import CdmInput, IngestReport, RefreshReport, Store

__all__ = [
    "CdmInput",
    "EventId",
    "IngestReport",
    "Job",
    "JobQueue",
    "JobStatus",
    "MessageId",
    "RefreshReport",
    "Store",
    "Submission",
    "SubmissionOutcome",
    "open_store",
    "upgrade",
]
