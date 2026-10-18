"""CCSDS NDM/XML documents: parsing, navigating and writing.

The message readers (:mod:`keepout.cdm.xml`, :mod:`keepout.omm.xml`) parse a
document with :func:`parse` and walk it with :func:`local`, :func:`single`,
:func:`leaves` and :func:`values`, which ignore namespaces, so messages with
or without the ``ndm:`` namespace read alike. ``<USER_DEFINED
parameter="NAME">`` is the keyword ``USER_DEFINED_NAME``, as in KVN.
DOCTYPE declarations are refused: CCSDS messages never need them, and
refusing them rules out entity-expansion attacks. Errors are raised as the
parse error of the message family (:class:`~keepout.ccsds.message.Family`:
:class:`~keepout.errors.CdmParseError`, :class:`~keepout.errors.OmmParseError`).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterator
from typing import Final
from xml.etree import ElementTree

from keepout.ccsds.message import USER_DEFINED, USER_DEFINED_PREFIX, Family, Located

__all__ = [
    "XSI",
    "add_leaf",
    "keyword",
    "leaves",
    "local",
    "parse",
    "single",
    "tostring",
    "values",
]

XSI: Final = "http://www.w3.org/2001/XMLSchema-instance"
"""Namespace of the ``xsi:`` schema-location attributes."""

_NOT_XML: Final = re.compile("[^\t\n\r\x20-퟿-�\U00010000-\U0010ffff]")
"""A character XML 1.0 cannot hold."""


def parse(text: str, family: Family) -> ElementTree.Element:
    """The root element of an XML document.

    Parameters
    ----------
    text
        The document.
    family
        The message family.

    Raises
    ------
    CcsdsParseError
        The family's parse error if the document has a DOCTYPE declaration
        or is not well-formed XML (with the line of the problem).
    """
    error = family.parse_error
    if "<!DOCTYPE" in text.upper():
        raise error(f"DOCTYPE declarations are not allowed in {family.name} XML")
    try:
        # Safe without defusedxml: with no DOCTYPE no entity can be declared,
        # and ElementTree never fetches external resources.
        return ElementTree.fromstring(text)  # noqa: S314
    except ElementTree.ParseError as exc:
        raise error(f"malformed XML: {exc}", line=exc.position[0] or None) from exc


def local(tag: str) -> str:
    """An element's name without its namespace."""
    return tag.rsplit("}", 1)[-1]


def single(
    parent: ElementTree.Element,
    name: str,
    path: str,
    family: Family,
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
    family
        The message family.
    record
        Position of the message in its document, for errors.

    Raises
    ------
    CcsdsParseError
        The family's parse error if ``parent`` has no such child or more
        than one.
    """
    found = [child for child in parent if local(child.tag) == name]
    if len(found) != 1:
        raise family.parse_error(
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


def keyword(
    leaf: ElementTree.Element,
    path: str,
    family: Family,
    *,
    record: int | None = None,
) -> str:
    """The keyword an element holds: its name, ``USER_DEFINED_NAME`` for
    ``<USER_DEFINED parameter="NAME">``.

    Parameters
    ----------
    leaf
        The element.
    path
        Its element path, for errors.
    family
        The message family.
    record
        Position of the message in its document, for errors.

    Raises
    ------
    CcsdsParseError
        The family's parse error for a ``USER_DEFINED`` element without a
        ``parameter``.
    """
    name = local(leaf.tag)
    if name != USER_DEFINED:
        return name
    parameter = leaf.get("parameter")
    if not parameter:
        raise family.parse_error(
            f"{USER_DEFINED} has no parameter attribute", record=record, path=path
        )
    return USER_DEFINED_PREFIX + parameter


def values(
    element: ElementTree.Element,
    path: str,
    family: Family,
    groups: Collection[str] = (),
    *,
    record: int | None = None,
) -> Iterator[tuple[str, Located]]:
    """The keyword and value of each element without children inside
    ``element`` (see :func:`leaves`), in document order.

    Each value is the element's text without surrounding blanks, with its
    ``units`` attribute and element path.

    Parameters
    ----------
    element
        The element.
    path
        Its element path.
    family
        The message family.
    groups
        Names of grouping elements, walked into even when empty.
    record
        Position of the message in its document, for errors.

    Raises
    ------
    CcsdsParseError
        As :func:`keyword`.
    """
    for leaf, leaf_path in leaves(element, path, groups):
        name = keyword(leaf, leaf_path, family, record=record)
        text = (leaf.text or "").strip()
        yield name, Located(text, leaf.get("units"), path=leaf_path)


def add_leaf(
    parent: ElementTree.Element,
    tag: str,
    text: str,
    unit: str | None,
    family: Family,
    *,
    keyword: str | None = None,
    attributes: dict[str, str] | None = None,
) -> ElementTree.Element:
    """A new last child holding ``text``, with a ``units`` attribute if
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
    family
        The message family.
    keyword
        Keyword the element holds, for errors; ``tag`` if not given.
    attributes
        Further attributes of the new element.

    Raises
    ------
    KeepoutError
        The family's write error if ``text`` contains a character XML 1.0
        cannot hold.
    """
    if _NOT_XML.search(text):
        raise family.write_error(f"{keyword or tag}: XML 1.0 cannot hold {text!r}")
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
