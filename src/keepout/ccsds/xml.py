"""CCSDS NDM/XML documents: parsing, navigating and writing.

The message readers (:mod:`keepout.cdm.xml`, :mod:`keepout.omm.xml`) parse a
document with :func:`parse` and walk it with :func:`local`, :func:`single`
and :func:`leaves`, which ignore namespaces, so messages with or without the
``ndm:`` namespace read alike. DOCTYPE declarations are refused: CCSDS
messages never need them, and refusing them rules out entity-expansion
attacks. Errors are raised as the parse error of the message family
(:class:`~keepout.errors.CdmParseError`, :class:`~keepout.errors.OmmParseError`).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Iterator
from typing import Final
from xml.etree import ElementTree

from keepout.errors import CcsdsParseError

__all__ = ["XSI", "add_leaf", "leaves", "local", "parse", "single", "tostring"]

XSI: Final = "http://www.w3.org/2001/XMLSchema-instance"
"""Namespace of the ``xsi:`` schema-location attributes."""

_NOT_XML: Final = re.compile("[^\t\n\r\x20-퟿-�\U00010000-\U0010ffff]")
"""A character XML 1.0 cannot hold."""


def parse(text: str, message: str, error: type[CcsdsParseError]) -> ElementTree.Element:
    """The root element of an XML document.

    Parameters
    ----------
    text
        The document.
    message
        Name of the message type, for errors, e.g. ``CDM``.
    error
        Parse error of the message family.

    Raises
    ------
    CcsdsParseError
        ``error`` if the document has a DOCTYPE declaration or is not
        well-formed XML (with the line of the problem).
    """
    if "<!DOCTYPE" in text.upper():
        raise error(f"DOCTYPE declarations are not allowed in {message} XML")
    try:
        return ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        raise error(f"malformed XML: {exc}", line=exc.position[0] or None) from exc


def local(tag: str) -> str:
    """An element's name without its namespace."""
    return tag.rsplit("}", 1)[-1]


def single(
    parent: ElementTree.Element,
    name: str,
    path: str,
    error: type[CcsdsParseError],
    *,
    record: int | None = None,
) -> tuple[ElementTree.Element, str]:
    """The one child of ``parent`` named ``name``, and its path.

    Parameters
    ----------
    parent
        The element.
    name
        Local name of the child.
    path
        Element path of ``parent``, for errors.
    error
        Parse error of the message family.
    record
        Position of the message in its document, for errors.

    Raises
    ------
    CcsdsParseError
        ``error`` if ``parent`` has no such child or more than one.
    """
    found = [child for child in parent if local(child.tag) == name]
    if len(found) != 1:
        raise error(
            f"expected exactly one <{name}>, found {len(found)}",
            record=record,
            path=path,
        )
    return found[0], f"{path}/{name}"


def leaves(
    element: ElementTree.Element, path: str, groups: Collection[str] = ()
) -> Iterator[tuple[ElementTree.Element, str]]:
    """The elements without children inside ``element``, in document order,
    with their paths.

    Parameters
    ----------
    element
        The element.
    path
        Its element path.
    groups
        Names of grouping elements: they are walked into, never yielded,
        even when empty.
    """
    for child in element:
        name = local(child.tag)
        child_path = f"{path}/{name}"
        if len(child) or name in groups:
            yield from leaves(child, child_path, groups)
        else:
            yield child, child_path


def add_leaf(
    parent: ElementTree.Element,
    tag: str,
    text: str,
    unit: str | None,
    *,
    error: Callable[[str], Exception],
    keyword: str | None = None,
    attributes: dict[str, str] | None = None,
) -> ElementTree.Element:
    """Append an element holding ``text``, with a ``units`` attribute if
    ``unit`` is given.

    Parameters
    ----------
    parent
        The element to append to.
    tag
        Name of the new element.
    text
        Its text.
    unit
        Its ``units`` attribute, or None for none.
    error
        Makes the exception raised for text XML cannot hold, from its
        message: the writer error of the message family.
    keyword
        Keyword the element holds, for errors; ``tag`` if not given.
    attributes
        Further attributes of the new element.

    Raises
    ------
    Exception
        ``error(...)`` if ``text`` contains a character XML 1.0 cannot hold.
    """
    if _NOT_XML.search(text):
        raise error(f"{keyword or tag}: XML 1.0 cannot hold {text!r}")
    element = ElementTree.SubElement(parent, tag, attributes or {})
    element.text = text
    if unit is not None:
        element.set("units", unit)
    return element


def tostring(root: ElementTree.Element) -> str:
    """A document of ``root``: an XML declaration and the elements indented.

    ``root`` is indented in place.
    """
    ElementTree.indent(root)
    declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
    return declaration + ElementTree.tostring(root, encoding="unicode") + "\n"
