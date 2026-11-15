"""Tables of the conjunction store.

Times are UTC ISO-8601 text with nanoseconds
(``YYYY-MM-DDTHH:MM:SS.fffffffff``, :class:`UtcTime`), so text order is time
order and a TCA reads back as the CDM gave it. A foreign key deletes with its
parent (``ON DELETE CASCADE``), but for the catalog of a CDM (``RESTRICT``).
Relationships never load on access (``lazy="raise"``): a query says what it
loads. The migrations in ``keepout/store/migrations`` build the same tables: a
test compares the two.
"""

from __future__ import annotations

import enum
from typing import Any, Final, NewType, cast, get_args

from astropy.time import Time
from sqlalchemy import (
    JSON,
    CheckConstraint,
    Dialect,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator, TypeEngine

from keepout._types import override
from keepout.cdm import CdmFormat
from keepout.units import elapsed_seconds

__all__ = [
    "NAMING_CONVENTION",
    "Base",
    "Catalog",
    "CatalogFile",
    "Cdm",
    "Event",
    "EventId",
    "Job",
    "JobKind",
    "JobKindText",
    "JobStatus",
    "JobStatusText",
    "MessageId",
    "Result",
    "UtcTime",
]

EventId = NewType("EventId", str)
"""A stored event's id (:func:`keepout.events.stable_event_id`)."""

MessageId = NewType("MessageId", str)
"""A stored CDM's ``MESSAGE_ID``."""


class JobStatus(enum.StrEnum):
    """Where a job is: waiting, held by a worker, done, failed or cancelled."""

    QUEUED = "queued"
    """Waiting for a worker."""
    RUNNING = "running"
    """Claimed by a worker, under a lease it renews."""
    SUCCEEDED = "succeeded"
    """Done: its plan is cached in the store."""
    FAILED = "failed"
    """Ended with an error (:attr:`~keepout.store.Job.error`)."""
    CANCELLED = "cancelled"
    """Cancelled while queued or running (:meth:`~keepout.store.JobQueue.cancel`)."""

    @property
    def is_active(self) -> bool:
        """Whether a job of this status has not ended: queued or running."""
        return self in {JobStatus.QUEUED, JobStatus.RUNNING}


class JobKind(enum.StrEnum):
    """What a job computes: a manoeuvre plan, or the Pc an event lacks."""

    MANEUVER = "maneuver"
    """A manoeuvre plan of the event at its pinned CDM, cached there under
    :func:`~keepout.store.codec.maneuver_key`."""
    REFRESH = "refresh"
    """The Pc observations the event's CDMs up to the pinned one lack,
    cached under :func:`~keepout.store.codec.result_key`."""


NAMING_CONVENTION: Final = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
"""Names of indexes and constraints: SQLite alters a table by copying it
(Alembic's batch mode), which needs every constraint named."""

_CDM_FORMATS: Final[tuple[CdmFormat, ...]] = get_args(CdmFormat)


class UtcTime(TypeDecorator[Time]):
    """An :class:`~astropy.time.Time` as UTC ISO-8601 text with nanoseconds."""

    impl: TypeEngine[Any] | type[TypeEngine[Any]] = String
    cache_ok: bool | None = True

    @override
    def process_bind_param(
        self,
        value: Time | None,
        dialect: Dialect,  # noqa: ARG002 (SQLite only)
    ) -> str | None:
        """The text of a time, rounded to the nanosecond.

        Parameters
        ----------
        value
            The time to store, or None.
        dialect
            The database dialect (not used).

        Returns
        -------
        str or None
            The time as UTC ISO text, or None for None.
        """
        if value is None:
            return None
        # A scalar time's isot is a str (astropy types it array-or-scalar).
        return cast("str", Time(value.utc, precision=9).isot)

    @override
    def process_result_value(
        self,
        value: str | None,
        dialect: Dialect,  # noqa: ARG002 (SQLite only)
    ) -> Time | None:
        """The time of a text.

        Parameters
        ----------
        value
            The stored text, or None.
        dialect
            The database dialect (not used).

        Returns
        -------
        astropy.time.Time or None
            The time, in the UTC scale, or None for None.
        """
        if value is None:
            return None
        return Time(value, format="isot", scale="utc")

    @override
    def compare_values(self, x: Time | None, y: Time | None) -> bool:
        """Whether two times are the same instant (a changed one is written).

        Parameters
        ----------
        x, y
            The times to compare, or None.

        Returns
        -------
        bool
            ``True`` if the times are equal.
        """
        if x is None or y is None:
            return x is y
        return elapsed_seconds(x, y) == 0.0


class JobStatusText(TypeDecorator[JobStatus]):
    """A :class:`JobStatus` as its text."""

    impl: TypeEngine[Any] | type[TypeEngine[Any]] = String
    cache_ok: bool | None = True

    @override
    def process_bind_param(
        self,
        value: JobStatus | None,
        dialect: Dialect,  # noqa: ARG002 (SQLite only)
    ) -> str | None:
        """The text of a status.

        Parameters
        ----------
        value
            The status to store, or None.
        dialect
            The database dialect (not used).

        Returns
        -------
        str or None
            The status's text, or None for None.
        """
        return None if value is None else JobStatus(value).value

    @override
    def process_result_value(
        self,
        value: str | None,
        dialect: Dialect,  # noqa: ARG002 (SQLite only)
    ) -> JobStatus | None:
        """The status of a text.

        Parameters
        ----------
        value
            The stored text, or None.
        dialect
            The database dialect (not used).

        Returns
        -------
        JobStatus or None
            The status, or None for None.
        """
        return None if value is None else JobStatus(value)


class JobKindText(TypeDecorator[JobKind]):
    """A :class:`JobKind` as its text."""

    impl: TypeEngine[Any] | type[TypeEngine[Any]] = String
    cache_ok: bool | None = True

    @override
    def process_bind_param(
        self,
        value: JobKind | None,
        dialect: Dialect,  # noqa: ARG002 (SQLite only)
    ) -> str | None:
        """The text of a kind.

        Parameters
        ----------
        value
            The kind to store, or None.
        dialect
            The database dialect (not used).

        Returns
        -------
        str or None
            The kind's text, or None for None.
        """
        return None if value is None else JobKind(value).value

    @override
    def process_result_value(
        self,
        value: str | None,
        dialect: Dialect,  # noqa: ARG002 (SQLite only)
    ) -> JobKind | None:
        """The kind of a text.

        Parameters
        ----------
        value
            The stored text, or None.
        dialect
            The database dialect (not used).

        Returns
        -------
        JobKind or None
            The kind, or None for None.
        """
        return None if value is None else JobKind(value)


class Base(DeclarativeBase):
    """Base class of the store's tables."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Event(Base):
    """A conjunction event: the CDMs of one encounter of two objects."""

    __tablename__ = "events"
    __table_args__: tuple[object, ...] = (
        Index(None, "object1", "object2", "tca_min"),
        Index(None, "tca_max"),
    )

    id: Mapped[EventId] = mapped_column(String(12), primary_key=True)
    object1: Mapped[str] = mapped_column(String)
    object2: Mapped[str] = mapped_column(String)
    conjunction_id: Mapped[str | None] = mapped_column(String)
    tca_min: Mapped[Time] = mapped_column(UtcTime)
    """Earliest TCA of the event's CDMs."""
    tca_max: Mapped[Time] = mapped_column(UtcTime)
    """Latest TCA of the event's CDMs."""
    created_at: Mapped[Time] = mapped_column(UtcTime)

    cdms: Mapped[list[Cdm]] = relationship(
        back_populates="event", lazy="raise", passive_deletes=True
    )


class Cdm(Base):
    """A stored CDM: its original text and where it came from."""

    __tablename__ = "cdms"
    __table_args__: tuple[object, ...] = (
        Index(None, "event_id", "creation_date"),
        CheckConstraint(
            f"format IN ({', '.join(repr(f) for f in _CDM_FORMATS)})",
            name="format",
        ),
    )

    message_id: Mapped[MessageId] = mapped_column(String, primary_key=True)
    event_id: Mapped[EventId] = mapped_column(
        String(12), ForeignKey("events.id", ondelete="CASCADE")
    )
    creation_date: Mapped[Time] = mapped_column(UtcTime)
    tca: Mapped[Time] = mapped_column(UtcTime)
    digest: Mapped[str] = mapped_column(String(64))
    """SHA-256 of the message written as KVN, in hexadecimal."""
    content: Mapped[str] = mapped_column(Text)
    """The message as read, KVN or XML."""
    format: Mapped[CdmFormat] = mapped_column(String(3))
    origin: Mapped[str] = mapped_column(String)
    """The file the message was read from, or ``"screen"``."""
    catalog_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("catalogs.id", ondelete="RESTRICT"), index=True
    )
    """The catalog a screening made the message from. A catalog of stored
    messages cannot be deleted (``ON DELETE RESTRICT``): its provenance
    never deletes them."""
    ingested_at: Mapped[Time] = mapped_column(UtcTime)

    event: Mapped[Event] = relationship(back_populates="cdms", lazy="raise")
    catalog: Mapped[Catalog | None] = relationship(lazy="raise")
    results: Mapped[list[Result]] = relationship(
        back_populates="cdm", lazy="raise", passive_deletes=True
    )


class Catalog(Base):
    """A catalog a screening ran against: its files, as they were."""

    __tablename__ = "catalogs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label: Mapped[str] = mapped_column(String)

    files: Mapped[list[CatalogFile]] = relationship(
        lazy="raise", passive_deletes=True, order_by="CatalogFile.id"
    )


class CatalogFile(Base):
    """A file of a catalog, with its digest at the time of the screening."""

    __tablename__ = "catalog_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    catalog_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("catalogs.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String)
    path: Mapped[str] = mapped_column(String)
    url: Mapped[str | None] = mapped_column(String)
    retrieved: Mapped[Time | None] = mapped_column(UtcTime)
    sha256: Mapped[str] = mapped_column(String(64))


class Result(Base):
    """A cached computation on a CDM, e.g. its Pc observation."""

    __tablename__ = "results"
    __table_args__: tuple[object, ...] = (UniqueConstraint("message_id", "key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[MessageId] = mapped_column(
        String, ForeignKey("cdms.message_id", ondelete="CASCADE")
    )
    key: Mapped[str] = mapped_column(String(64))
    """What was computed, and with which options."""
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[Time] = mapped_column(UtcTime)

    cdm: Mapped[Cdm] = relationship(back_populates="results", lazy="raise")


class Job(Base):
    """A computation queued on an event's CDM, and how far it got."""

    __tablename__ = "jobs"
    __table_args__: tuple[object, ...] = (
        Index(None, "status", "created_at"),
        Index(None, "event_id", "status"),
        CheckConstraint(
            f"status IN ({', '.join(repr(s.value) for s in JobStatus)})",
            name="status",
        ),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    """Never reused (SQLite ``AUTOINCREMENT``): a worker's stale write to a
    deleted job cannot reach a new one."""
    event_id: Mapped[EventId] = mapped_column(
        String(12), ForeignKey("events.id", ondelete="CASCADE")
    )
    """Indexed with :attr:`status` (``ix_jobs_event_id_status``)."""
    message_id: Mapped[MessageId] = mapped_column(
        String, ForeignKey("cdms.message_id", ondelete="CASCADE"), index=True
    )
    """The CDM the computation is pinned to: the event's latest when queued."""
    key: Mapped[str] = mapped_column(String(64))
    """The key its result is cached under."""
    params: Mapped[dict[str, Any]] = mapped_column(JSON)
    """What the computation needs, e.g. the settings."""
    status: Mapped[JobStatus] = mapped_column(JobStatusText)
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    """How many attempts it cost: a claim adds one, a release on a stop takes
    it back."""
    claims: Mapped[int] = mapped_column(Integer, default=0)
    """How many times it was claimed, never lowered: a worker's writes name
    the claim they belong to."""
    worker: Mapped[str | None] = mapped_column(String)
    """The worker that holds it, or that finished, failed or held it when
    cancelled; cleared when it goes back to the queue."""
    lease_until: Mapped[Time | None] = mapped_column(UtcTime)
    """When the worker's hold on it lapses."""
    created_at: Mapped[Time] = mapped_column(UtcTime)
    started_at: Mapped[Time | None] = mapped_column(UtcTime)
    finished_at: Mapped[Time | None] = mapped_column(UtcTime)
    error: Mapped[str | None] = mapped_column(String)
    """Why it failed."""
    # Last: migration 0003 added it to the end of the table.
    kind: Mapped[JobKind] = mapped_column(
        JobKindText, server_default=JobKind.MANEUVER.value
    )
    """What it computes; the jobs of a store from before kinds are
    manoeuvre plans."""
