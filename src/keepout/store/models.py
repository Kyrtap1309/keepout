"""Tables of the conjunction store.

Times are UTC ISO-8601 text with microseconds (``YYYY-MM-DDTHH:MM:SS.ffffff``),
so text order is time order. A foreign key deletes with its parent
(``ON DELETE CASCADE``), but for the catalog of a CDM (``RESTRICT``). The
migrations in ``keepout/store/migrations`` build the same tables: a test
compares the two.
"""

from typing import Any, Final, Literal

from sqlalchemy import (
    JSON,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    false,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

__all__ = [
    "NAMING_CONVENTION",
    "Base",
    "Catalog",
    "CatalogFile",
    "Cdm",
    "CdmFormat",
    "Event",
    "Result",
]

NAMING_CONVENTION: Final = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
"""Names of indexes and constraints: SQLite alters a table by copying it
(Alembic's batch mode), which needs every constraint named."""

CdmFormat = Literal["kvn", "xml"]
"""The format a CDM was read in."""


class Base(DeclarativeBase):
    """Base class of the store's tables."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Event(Base):
    """A conjunction event: the CDMs of one encounter of two objects."""

    __tablename__ = "events"
    __table_args__ = (Index(None, "object1", "object2", "tca_min"),)

    id: Mapped[str] = mapped_column(String(8), primary_key=True)
    """Eight lowercase hexadecimal characters."""
    object1: Mapped[str] = mapped_column(String)
    object2: Mapped[str] = mapped_column(String)
    conjunction_id: Mapped[str | None] = mapped_column(String)
    tca_min: Mapped[str] = mapped_column(String)
    """Earliest TCA of the event's CDMs."""
    tca_max: Mapped[str] = mapped_column(String)
    """Latest TCA of the event's CDMs."""
    archived: Mapped[bool] = mapped_column(default=False, server_default=false())
    created_at: Mapped[str] = mapped_column(String)


class Cdm(Base):
    """A stored CDM: its original text and where it came from."""

    __tablename__ = "cdms"
    __table_args__ = (
        Index(None, "event_id", "creation_date"),
        CheckConstraint("format IN ('kvn', 'xml')", name="format"),
    )

    message_id: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(
        String(8), ForeignKey("events.id", ondelete="CASCADE")
    )
    creation_date: Mapped[str] = mapped_column(String)
    tca: Mapped[str] = mapped_column(String)
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
    ingested_at: Mapped[str] = mapped_column(String)


class Catalog(Base):
    """A catalog a screening ran against: its files, as they were."""

    __tablename__ = "catalogs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label: Mapped[str] = mapped_column(String)


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
    retrieved: Mapped[str | None] = mapped_column(String)
    sha256: Mapped[str] = mapped_column(String(64))


class Result(Base):
    """A cached computation on a CDM, e.g. its Pc observation."""

    __tablename__ = "results"
    __table_args__ = (UniqueConstraint("message_id", "key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[str] = mapped_column(
        String, ForeignKey("cdms.message_id", ondelete="CASCADE")
    )
    key: Mapped[str] = mapped_column(String(64))
    """What was computed, and with which options."""
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(String)
