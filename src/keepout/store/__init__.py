"""The conjunction store: events, CDMs and cached results in a SQLite
database (the ``store`` extra: SQLAlchemy and Alembic).

:func:`open_store` opens one, creating a new file at the current schema;
:func:`upgrade` brings a file made by an older keepout up to it. A
:class:`Store` ingests CDMs (:class:`CdmInput`) into events and caches
their Pc (:class:`RefreshReport`); for the dashboard, it sums up a page of
events (:class:`SummaryPage` of :class:`EventSummary`) and reads one event
for its page (:class:`EventDetail`). A :class:`JobQueue` holds the
manoeuvre plans and Pc refreshes queued for a worker (:class:`Job` of a
:class:`JobKind`, :class:`Submission`, :class:`SubmissionOutcome`), and an
event's manoeuvre jobs for the planner (:class:`PlanJobs`).
"""

from keepout.store.db import open_store, upgrade
from keepout.store.jobs import (
    Job,
    JobKind,
    JobQueue,
    JobStatus,
    PlanJobs,
    Submission,
    SubmissionOutcome,
)
from keepout.store.models import EventId, MessageId
from keepout.store.repo import CdmInput, IngestReport, RefreshReport, Store
from keepout.store.summaries import EventDetail, EventSummary, SummaryPage

__all__ = [
    "CdmInput",
    "EventDetail",
    "EventId",
    "EventSummary",
    "IngestReport",
    "Job",
    "JobKind",
    "JobQueue",
    "JobStatus",
    "MessageId",
    "PlanJobs",
    "RefreshReport",
    "Store",
    "Submission",
    "SubmissionOutcome",
    "SummaryPage",
    "open_store",
    "upgrade",
]
