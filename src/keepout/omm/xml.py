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
user-defined parameters. ``EPHEMERIS_TYPE`` and ``CLASSIFICATION_TYPE`` may be
omitted (:data:`~keepout.omm.keywords.DEFAULTS`). DOCTYPE declarations are
refused, which rules out entity-expansion attacks.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Final
from xml.etree import ElementTree

from astropy.time import Time

from keepout.errors import OmmParseError
from keepout.omm._io import (
    Text,
    ccsds_values,
    check_version,
    creation_date_text,
    load_file,
    parse_record,
)
from keepout.omm.keywords import DEFAULTS, VERSION
from keepout.omm.model import Omm

__all__ = ["SCHEMA_LOCATION", "dumps", "load", "loads"]

SCHEMA_LOCATION: Final = (
    "https://sanaregistry.org/r/ndmxml_unqualified/ndmxml-4.0.0-master-4.0.xsd"
)
"""Schema of the NDM/XML 4.0.0 suite, which includes OMM 3.0."""

_XSI: Final = "http://www.w3.org/2001/XMLSchema-instance"
_USER_DEFINED: Final = "USER_DEFINED"
_NDM_PARTS: Final = frozenset({"COMMENT", "MESSAGE_ID"})
"""Children of ``<ndm>`` that are not messages."""
_NOT_XML: Final = re.compile("[^\t\n\r\x20-퟿-�\U00010000-\U0010ffff]")
"""A character XML 1.0 cannot hold."""


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _single(
    parent: ElementTree.Element, name: str, path: str, record: int
) -> tuple[ElementTree.Element, str]:
    found = [child for child in parent if _local(child.tag) == name]
    if len(found) != 1:
        raise OmmParseError(
            f"expected exactly one <{name}>, found {len(found)}",
            record=record,
            path=path,
        )
    return found[0], f"{path}/{name}"


def _leaves(
    element: ElementTree.Element, path: str
) -> Iterator[tuple[ElementTree.Element, str]]:
    for child in element:
        child_path = f"{path}/{_local(child.tag)}"
        if len(child):
            yield from _leaves(child, child_path)
        else:
            yield child, child_path


def _messages(root: ElementTree.Element) -> list[tuple[ElementTree.Element, str]]:
    """The ``<omm>`` elements of a document and their paths."""
    name = _local(root.tag)
    if name == "omm":
        return [(root, "omm")]
    if name != "ndm":
        raise OmmParseError(f"root element must be <omm> or <ndm>, got <{name}>")
    messages: list[tuple[ElementTree.Element, str]] = []
    for child in root:
        child_name = _local(child.tag)
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
    body, body_path = _single(element, "body", path, record)
    segment, segment_path = _single(body, "segment", body_path, record)
    values: dict[str, Text] = {}
    for part in ("metadata", "data"):
        container, container_path = _single(segment, part, segment_path, record)
        for leaf, leaf_path in _leaves(container, container_path):
            keyword = _local(leaf.tag)
            if keyword == "COMMENT":
                continue
            if keyword == _USER_DEFINED:
                keyword += "_" + leaf.get("parameter", "")
            if keyword in values:
                raise OmmParseError(
                    "duplicate keyword", record=record, path=leaf_path, keyword=keyword
                )
            values[keyword] = Text(leaf.text or "", leaf.get("units"), path=leaf_path)
    return parse_record({**DEFAULTS, **values}, record=record, path=segment_path)


def loads(text: str) -> tuple[Omm, ...]:
    """Parse an OMM, or an NDM of OMMs, in XML.

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
        a keyword, or its elements are missing or invalid. The error names the
        message (``record``) and the element path.
    """
    if "<!DOCTYPE" in text.upper():
        raise OmmParseError("DOCTYPE declarations are not allowed in OMM XML")
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        raise OmmParseError(
            f"malformed XML: {exc}", line=exc.position[0] or None
        ) from exc
    return tuple(
        _record(element, path, record)
        for record, (element, path) in enumerate(_messages(root), start=1)
    )


def load(path: str | Path) -> tuple[Omm, ...]:
    """Read an XML OMM file (UTF-8, or gzip-compressed UTF-8); see :func:`loads`.

    Parameters
    ----------
    path
        File to read.
    """
    return load_file(path, loads)


def dumps(
    omms: Iterable[Omm],
    *,
    originator: str = "UNKNOWN",
    creation_date: Time | None = None,
) -> str:
    """Write OMMs as an NDM/XML ``<ndm>`` of version 3.0 ``<omm>`` elements.

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
    ValueError
        If a value contains a character XML 1.0 cannot hold.
    """
    created = creation_date_text(creation_date)
    root = ElementTree.Element(
        "ndm", {"xmlns:xsi": _XSI, "xsi:noNamespaceSchemaLocation": SCHEMA_LOCATION}
    )
    for omm in omms:
        message = ElementTree.SubElement(
            root, "omm", {"id": "CCSDS_OMM_VERS", "version": VERSION}
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
    ElementTree.indent(root)
    declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
    return declaration + ElementTree.tostring(root, encoding="unicode") + "\n"


def _leaf(
    parent: ElementTree.Element, keyword: str, value: str, unit: str | None
) -> None:
    if _NOT_XML.search(value):
        raise ValueError(f"{keyword}: XML 1.0 cannot hold {value!r}")
    element = ElementTree.SubElement(parent, keyword)
    element.text = value
    if unit is not None:
        element.set("units", unit)
