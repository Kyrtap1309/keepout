"""CCSDS OMM NDM/XML reader and writer.

An OMM in XML (CCSDS 502.0-B-3 section 8, NDM/XML 505.0) is an ``<omm>``
element of a header and a body with one segment of ``metadata`` and ``data``
(``meanElements``, ``tleParameters``, ...). Several OMMs come wrapped in an
``<ndm>`` element, as CelesTrak's XML does. The reader takes either, with or
without the ``ndm:`` namespace, in versions 2.0 and 3.0; the writer writes
an ``<ndm>`` of version 3.0 OMMs, valid against the NDM/XML 4.0.0 schema.

Only the SGP4 elements of :data:`keepout.omm.keywords.KEYWORDS` are kept; the
reader reads every element of the segment wherever it is grouped and ignores
the header, comments, ``GM``, spacecraft parameters, covariance and
user-defined parameters, though a segment may not repeat a keyword and a
``<USER_DEFINED>`` element must name its ``parameter`` (it is the keyword
``USER_DEFINED_NAME``, as in KVN). ``EPHEMERIS_TYPE`` and
``CLASSIFICATION_TYPE`` may be omitted (:data:`~keepout.omm.keywords.DEFAULTS`).
DOCTYPE declarations are refused, which rules out entity-expansion attacks.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Final
from xml.etree import ElementTree

from astropy.time import Time

from keepout.ccsds.message import COMMENT, UNKNOWN, KeywordValues
from keepout.ccsds.xml import XSI, add_leaf, local, parse, single, tostring, values
from keepout.errors import OmmParseError
from keepout.omm._io import (
    OMM,
    ccsds_values,
    check_version,
    creation_date_text,
    parse_record,
)
from keepout.omm.keywords import DEFAULTS, VERSION, VERSION_KEYWORD
from keepout.omm.model import Omm

__all__ = ["SCHEMA_LOCATION", "dumps", "load", "loads"]

SCHEMA_LOCATION: Final = (
    "https://sanaregistry.org/r/ndmxml_unqualified/ndmxml-4.0.0-master-4.0.xsd"
)
"""Schema of the NDM/XML 4.0.0 suite, which includes OMM 3.0."""

_NDM_PARTS: Final = frozenset({COMMENT, "MESSAGE_ID"})
"""Children of ``<ndm>`` that are not messages."""


def _messages(root: ElementTree.Element) -> list[tuple[ElementTree.Element, str]]:
    """The ``<omm>`` elements of a document and their paths.

    Raises
    ------
    OmmParseError
        If the root element is neither ``<omm>`` nor ``<ndm>``, or an
        ``<ndm>`` holds a message other than an OMM.
    """
    name = local(root.tag)
    if name == "omm":
        return [(root, "omm")]
    if name != "ndm":
        raise OmmParseError(f"root element must be <omm> or <ndm>, got <{name}>")
    messages: list[tuple[ElementTree.Element, str]] = []
    for child in root:
        child_name = local(child.tag)
        if child_name == "omm":
            messages.append((child, f"ndm/omm[{len(messages) + 1}]"))
        elif child_name not in _NDM_PARTS:
            raise OmmParseError(
                f"<ndm> holds a <{child_name}>; only OMMs are read", path="ndm"
            )
    return messages


def _record(element: ElementTree.Element, path: str, record: int) -> Omm:
    version = element.get("version")
    if version is None:
        raise OmmParseError("<omm> has no version attribute", record=record, path=path)
    check_version(version, record=record, path=path)
    body, body_path = single(element, "body", path, OMM, record=record)
    segment, segment_path = single(body, "segment", body_path, OMM, record=record)
    found = KeywordValues(OMM, record=record)
    for part in ("metadata", "data"):
        container, container_path = single(
            segment, part, segment_path, OMM, record=record
        )
        for keyword, value in values(container, container_path, OMM, record=record):
            found.add(keyword, value)
    return parse_record({**DEFAULTS, **found.values}, record=record, path=segment_path)


def loads(text: str) -> tuple[Omm, ...]:
    """The OMMs of an OMM, or of an NDM of OMMs, in XML.

    Parameters
    ----------
    text
        The document: an ``<omm>`` or an ``<ndm>`` holding ``<omm>`` elements.

    Returns
    -------
    tuple[Omm, ...]
        The elements of each message, in document order.

    Raises
    ------
    OmmParseError
        If the document is malformed or has a DOCTYPE, the root is neither
        ``<omm>`` nor ``<ndm>``, an ``<ndm>`` holds another kind of message,
        a message lacks its version or has an unsupported one, its structure
        (``body``, ``segment``, ``metadata``, ``data``) is broken, it repeats
        a keyword, a ``USER_DEFINED`` element has no ``parameter``, or its
        elements are missing or invalid. The error names the
        message (``record``) and the element path.
    """
    root = parse(text, OMM)
    return tuple(
        _record(element, path, record)
        for record, (element, path) in enumerate(_messages(root), start=1)
    )


def load(path: str | Path) -> tuple[Omm, ...]:
    """The OMMs in an XML file (UTF-8, or gzip-compressed UTF-8); see :func:`loads`.

    Parameters
    ----------
    path
        File to read.
    """
    return OMM.load(path, loads)


def dumps(
    omms: Iterable[Omm],
    *,
    originator: str = UNKNOWN,
    creation_date: Time | None = None,
) -> str:
    """OMMs as an NDM/XML ``<ndm>`` of version 3.0 ``<omm>`` elements.

    Values carry their units. The messages are valid against the NDM/XML
    4.0.0 schema (:data:`SCHEMA_LOCATION`), and reading the text gives the
    same elements back.

    Parameters
    ----------
    omms
        The objects, written in order.
    originator
        ``ORIGINATOR`` of every message: the creating agency or operator.
    creation_date
        ``CREATION_DATE`` of every message; the current time (whole seconds)
        if not given.

    Raises
    ------
    OmmWriteError
        If a value contains a character XML 1.0 cannot hold.
    """
    created = creation_date_text(creation_date)
    root = ElementTree.Element(
        "ndm", {"xmlns:xsi": XSI, "xsi:noNamespaceSchemaLocation": SCHEMA_LOCATION}
    )
    for omm in omms:
        message = ElementTree.SubElement(
            root, "omm", {"id": VERSION_KEYWORD, "version": VERSION}
        )
        header = ElementTree.SubElement(message, "header")
        _leaf(header, "CREATION_DATE", created, None)
        _leaf(header, "ORIGINATOR", originator, None)
        segment = ElementTree.SubElement(
            ElementTree.SubElement(message, "body"), "segment"
        )
        for keyword, value, unit, group in ccsds_values(omm):
            parent = segment
            for name in group:
                found = parent.find(name)
                parent = (
                    found if found is not None else ElementTree.SubElement(parent, name)
                )
            _leaf(parent, keyword, value, unit)
    return tostring(root)


def _leaf(
    parent: ElementTree.Element, keyword: str, value: str, unit: str | None
) -> None:
    add_leaf(parent, keyword, value, unit, OMM)
