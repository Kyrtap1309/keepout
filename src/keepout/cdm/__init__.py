"""CCSDS Conjunction Data Messages: model, KVN and XML.

Use :func:`load` / :func:`loads` to read either format; :mod:`keepout.cdm.kvn`
and :mod:`keepout.cdm.xml` read and write a specific one.
:func:`~keepout.cdm.builder.make_cdm` builds the CDM of a conjunction.
"""

from pathlib import Path

from keepout.ccsds.files import read_file, strip_bom
from keepout.cdm import kvn, xml
from keepout.cdm._io import CDM
from keepout.cdm.builder import (
    ORIGINATOR,
    Maneuverable,
    ObjectMetadata,
    ObjectType,
    make_cdm,
    make_message_id,
)
from keepout.cdm.model import Cdm, CdmObject, CovarianceMethod, Field, Section

__all__ = [
    "ORIGINATOR",
    "Cdm",
    "CdmObject",
    "CovarianceMethod",
    "Field",
    "Maneuverable",
    "ObjectMetadata",
    "ObjectType",
    "Section",
    "kvn",
    "load",
    "loads",
    "make_cdm",
    "make_message_id",
    "read_text",
    "xml",
]


def loads(text: str) -> Cdm:
    """The CDM in KVN or XML ``text``, told apart by the first non-blank character.

    Line numbers in errors count from the first non-blank line for XML (whose
    declaration must open the document), from the first line for KVN.

    Parameters
    ----------
    text
        The message.
    """
    body = strip_bom(text)
    start = body.lstrip()
    return xml.loads(start) if start.startswith("<") else kvn.loads(body)


def load(path: str | Path) -> Cdm:
    """The CDM in a KVN or XML file.

    Parameters
    ----------
    path
        File to read (UTF-8, gzip-compressed or not).
    """
    return loads(read_text(path))


def read_text(path: str | Path) -> str:
    """The text of a CDM file, as :func:`load` reads it: decompressed if it
    is gzip data, without a byte-order mark.

    Parameters
    ----------
    path
        File to read (UTF-8, gzip-compressed or not).

    Raises
    ------
    CdmParseError
        If the file is broken gzip data or not UTF-8 text.
    OSError
        If it cannot be read.
    """
    return read_file(path, CDM)
