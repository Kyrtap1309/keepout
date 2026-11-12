"""CCSDS Conjunction Data Messages: model, KVN and XML.

Use :func:`load` / :func:`loads` to read either format; :mod:`keepout.cdm.kvn`
and :mod:`keepout.cdm.xml` read and write a specific one.
:func:`~keepout.cdm.builder.make_cdm` builds the CDM of a conjunction.
"""

from pathlib import Path
from typing import Literal, TypeAlias

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
    "CdmFormat",
    "CdmObject",
    "CovarianceMethod",
    "Field",
    "Maneuverable",
    "ObjectMetadata",
    "ObjectType",
    "Section",
    "format_of",
    "kvn",
    "load",
    "loads",
    "make_cdm",
    "make_message_id",
    "read_text",
    "xml",
]


CdmFormat: TypeAlias = Literal["kvn", "xml"]
"""The two encodings of a CDM."""


def format_of(text: str) -> CdmFormat:
    """The format of a CDM's ``text``: XML if its first non-blank character
    is ``<``, else KVN.

    Parameters
    ----------
    text
        A CDM's text, as read from its file.
    """
    return "xml" if strip_bom(text).lstrip().startswith("<") else "kvn"


def loads(text: str) -> Cdm:
    """The CDM in KVN or XML ``text``, told apart by the first non-blank character.

    Line numbers in errors count from the first non-blank line for XML (whose
    declaration must open the document), from the first line for KVN.

    Parameters
    ----------
    text
        The message: one CDM, KVN or XML.

    Returns
    -------
    Cdm
        The message, every field kept as read.

    Raises
    ------
    CdmParseError
        If the message is malformed; the error names the line and keyword
        (KVN) or the element path (XML).

    Examples
    --------
    ```python
    from pathlib import Path

    from keepout import cdm

    message = cdm.loads(Path("tests/cdm/data/tracss_v2.kvn").read_text())
    print(message.message_id, message.tca.isot, message.miss_distance)
    # TRACSS-TEST-0001 2025-06-02T08:15:30.123456 412.5
    ```
    """
    body = strip_bom(text)
    return xml.loads(body.lstrip()) if format_of(body) == "xml" else kvn.loads(body)


def load(path: str | Path) -> Cdm:
    """The CDM in a KVN or XML file.

    Parameters
    ----------
    path
        File to read: UTF-8 text, or gzip-compressed UTF-8 text (recognised
        by its content, whatever the file name).

    Returns
    -------
    Cdm
        The message, every field kept as read.

    Raises
    ------
    CdmParseError
        If the file is broken gzip data, not UTF-8 text, or not a valid CDM.
    OSError
        If the file cannot be read.

    Examples
    --------
    ```python
    from keepout import cdm

    message = cdm.load("tests/cdm/data/tracss_v2.kvn")
    print(message.object1.object_id, message.object2.object_id)
    # 12345 30337
    print(message.collision_probability)  # the originator's Pc
    # 1.2e-05
    ```
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
