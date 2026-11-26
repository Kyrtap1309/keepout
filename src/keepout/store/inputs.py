"""What a store is given and what it reports: a CDM to ingest
(:class:`CdmInput`), and the reports of an ingest (:class:`IngestReport`)
and a refresh (:class:`RefreshReport`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import keepout.cdm

if TYPE_CHECKING:
    from pathlib import Path

    from keepout.cdm import Cdm, CdmFormat
    from keepout.store.models import EventId, MessageId

__all__ = ["CdmInput", "IngestReport", "RefreshReport"]


@dataclass(frozen=True, slots=True)
class CdmInput:
    """A CDM to ingest, with its text as read and where it came from.

    Parameters
    ----------
    cdm
        The message.
    text
        Its text, KVN or XML: what the store keeps.
    format
        The text's format.
    origin
        Where it came from: the file it was read from, or ``"screen"``.
    """

    cdm: Cdm
    text: str
    format: CdmFormat
    origin: str

    @classmethod
    def read(cls, path: Path) -> CdmInput:
        """The CDM of a KVN or XML file (UTF-8, gzip-compressed or not),
        its origin the path.

        Parameters
        ----------
        path
            The CDM file, KVN or XML; its path is the origin.

        Returns
        -------
        CdmInput
            The parsed CDM, its text, format and origin.

        Raises
        ------
        CdmParseError
            If the file is broken gzip data, not UTF-8 text or not a CDM.
        OSError
            If it cannot be read.
        """
        return cls.parse(keepout.cdm.read_text(path), str(path))

    @classmethod
    def parse(cls, text: str, origin: str) -> CdmInput:
        """The CDM of a text, KVN or XML.

        Parameters
        ----------
        text
            The message's text, as read (decoded, without a byte-order mark).
        origin
            Where it came from, e.g. the file it was read from.

        Returns
        -------
        CdmInput
            The parsed CDM, its text, format and origin.

        Raises
        ------
        CdmParseError
            If the text is not a CDM.
        """
        return cls(keepout.cdm.loads(text), text, keepout.cdm.format_of(text), origin)

    @classmethod
    def of(cls, cdm: Cdm, origin: str) -> CdmInput:
        """A CDM made in memory, e.g. by a screening, kept as its KVN.

        Parameters
        ----------
        cdm
            The message, e.g. from :func:`~keepout.cdm.make_cdm`.
        origin
            Where it came from, e.g. ``"screen"``.

        Returns
        -------
        CdmInput
            The CDM, its KVN text and its origin.
        """
        return cls(cdm, keepout.cdm.kvn.dumps(cdm), "kvn", origin)


@dataclass(frozen=True, slots=True)
class IngestReport:
    """What an ingest did, in the order the messages were given.

    Parameters
    ----------
    added
        The ``MESSAGE_ID`` of each message stored.
    already_stored
        The ``MESSAGE_ID`` of each message given that was stored before
        (or earlier in the same ingest), with the same content.
    events
        The id of each event the messages given belong to.
    new_events
        The id of each of those events that the ingest created.
    """

    added: tuple[MessageId, ...]
    already_stored: tuple[MessageId, ...]
    events: tuple[EventId, ...]
    new_events: tuple[EventId, ...]


@dataclass(frozen=True, slots=True)
class RefreshReport:
    """What a refresh did.

    Parameters
    ----------
    computed
        How many Pc observations it computed and cached.
    skipped
        The ``MESSAGE_ID`` of each CDM left without a Pc because it lacks
        ``HBR`` and the settings give no hard-body radius (only when every
        event is refreshed), oldest event first.
    """

    computed: int
    skipped: tuple[MessageId, ...] = ()
