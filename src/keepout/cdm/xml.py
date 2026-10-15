"""CDM NDM/XML reader and writer.

The reader flattens grouping elements, so it accepts CCSDS 508.0-B-1 messages
with or without the ``ndm:`` namespace (Technical Corrigendum 2) and CDM 2.0
groups such as ``physicalParameters`` and ``userDefinedParameters``.
``<USER_DEFINED parameter="NAME">`` becomes the keyword ``USER_DEFINED_NAME``,
as in KVN. DOCTYPE declarations are rejected: CDMs never need them, and
refusing them rules out entity-expansion attacks. No XSD validation is
performed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final
from xml.etree import ElementTree

from keepout.ccsds.xml import XSI, add_leaf, leaves, local, parse, single, tostring
from keepout.cdm._io import (
    Location,
    build_cdm,
    build_section,
    dump_file,
    load_file,
)
from keepout.cdm.keywords import KEYWORDS, SectionKind
from keepout.cdm.model import (
    COMMENT,
    USER_DEFINED_PREFIX,
    Cdm,
    Field,
    Section,
    format_field,
    parse_field,
)
from keepout.errors import CdmParseError, CdmWriteError

__all__ = ["NDM_NAMESPACE", "dump", "dumps", "load", "loads"]

NDM_NAMESPACE: Final = "urn:ccsds:schema:ndmxml"
# Schema of CDM 1.0 messages ("1.0" is the CDM version): the master schema of
# the NDM/XML 2.0.0 suite, which includes CDM 1.0.
_CDM_1_0_SCHEMA_LOCATION: Final = (
    "https://sanaregistry.org/r/ndmxml_unqualified/ndmxml-2.0.0-master-2.0.xsd"
)
_USER_DEFINED: Final = "USER_DEFINED"
_USER_GROUP: Final = ("data", "userDefinedParameters")
_GROUPS: Final = frozenset(
    {
        name
        for table in KEYWORDS.values()
        for spec in table.values()
        for name in spec.group
    }
    | {"physicalParameters", *_USER_GROUP}
)
"""Grouping elements; they are structure even when empty."""


def _keyword(leaf: ElementTree.Element, path: str) -> str:
    keyword = local(leaf.tag)
    if keyword != _USER_DEFINED:
        return keyword
    parameter = leaf.get("parameter")
    if not parameter:
        raise CdmParseError("USER_DEFINED has no parameter attribute", path=path)
    return USER_DEFINED_PREFIX + parameter


def _section(
    kind: SectionKind,
    element: ElementTree.Element,
    path: str,
    leading: Sequence[Field] = (),
) -> Section:
    fields = list(leading)
    locations = [Location(path=path)] * len(fields)
    for leaf, leaf_path in leaves(element, path, _GROUPS):
        keyword = _keyword(leaf, leaf_path)
        text = (leaf.text or "").strip()
        try:
            fields.append(
                Field(COMMENT, text)
                if keyword == COMMENT
                else parse_field(kind, keyword, text, leaf.get("units"))
            )
        except ValueError as exc:
            raise CdmParseError(str(exc), path=leaf_path, keyword=keyword) from exc
        locations.append(Location(path=leaf_path))
    return build_section(kind, fields, locations, Location(path=path))


def loads(text: str) -> Cdm:
    """Parse a CDM in NDM/XML.

    Parameters
    ----------
    text
        The message.

    Raises
    ------
    CdmParseError
        If the message is malformed; the error names the element path.
    """
    root = parse(text, "CDM", CdmParseError)
    if local(root.tag) != "cdm":
        raise CdmParseError(f"root element must be <cdm>, got <{local(root.tag)}>")
    version = root.get("version")
    if not version:
        raise CdmParseError("<cdm> has no version attribute")
    header, header_path = single(root, "header", "cdm", CdmParseError)
    body, body_path = single(root, "body", "cdm", CdmParseError)
    relative, relative_path = single(
        body, "relativeMetadataData", body_path, CdmParseError
    )
    segments = [child for child in body if local(child.tag) == "segment"]
    if len(segments) != 2:
        raise CdmParseError(
            f"expected 2 <segment> elements, found {len(segments)}", path=body_path
        )
    return build_cdm(
        _section("header", header, header_path, [Field("CCSDS_CDM_VERS", version)]),
        _section("relative", relative, relative_path),
        [
            _section("segment", segment, f"{body_path}/segment[{index}]")
            for index, segment in enumerate(segments, start=1)
        ],
    )


def _groups(
    fields: Sequence[Field], kind: SectionKind, version: str
) -> list[tuple[str, ...]]:
    """XML group path of each field.

    Known keywords use the table; ``USER_DEFINED_*`` go to
    ``userDefinedParameters``; other unknown keywords stay in the group of the
    field before them; comments take the group of the field after them
    (trailing comments the group of the last field).
    """
    table = KEYWORDS[kind]
    physical = (
        "physicalParameters" if version.startswith("2") else "additionalParameters"
    )
    group: tuple[str, ...] = ("metadata",) if kind == "segment" else ()
    groups: list[tuple[str, ...]] = []
    comments = 0
    for field in fields:
        if field.keyword == COMMENT:
            comments += 1
            continue
        spec = table.get(field.keyword)
        if spec is not None:
            group = tuple(
                physical if name == "additionalParameters" else name
                for name in spec.group
            )
        elif kind == "segment" and field.keyword.startswith(USER_DEFINED_PREFIX):
            group = _USER_GROUP
        groups.extend([group] * (comments + 1))
        comments = 0
    groups.extend([group] * comments)
    return groups


def _insert_comment(element: ElementTree.Element, text: str) -> None:
    """Add a comment after the element's leading comments: NDM/XML allows
    ``COMMENT`` only at the start of an element."""
    position = next(
        (i for i, child in enumerate(element) if child.tag != COMMENT), len(element)
    )
    comment = add_leaf(element, COMMENT, text, None, error=CdmWriteError)
    element.remove(comment)
    element.insert(position, comment)


def _write(
    parent: ElementTree.Element,
    fields: Sequence[Field],
    kind: SectionKind,
    version: str,
) -> None:
    names: list[str] = []
    elements = [parent]
    for field, group in zip(fields, _groups(fields, kind, version), strict=True):
        depth = 0
        while depth < min(len(names), len(group)) and names[depth] == group[depth]:
            depth += 1
        del names[depth:], elements[depth + 1 :]
        for name in group[depth:]:
            names.append(name)
            elements.append(ElementTree.SubElement(elements[-1], name))
        if field.keyword == COMMENT:
            _insert_comment(elements[-1], str(field.value))
            continue
        text, unit = format_field(kind, field)
        if field.keyword.startswith(USER_DEFINED_PREFIX) and group == _USER_GROUP:
            parameter = field.keyword.removeprefix(USER_DEFINED_PREFIX)
            tag, attributes = _USER_DEFINED, {"parameter": parameter}
        else:
            tag, attributes = field.keyword, None
        add_leaf(
            elements[-1],
            tag,
            text,
            unit,
            error=CdmWriteError,
            keyword=field.keyword,
            attributes=attributes,
        )


def dumps(cdm: Cdm) -> str:
    """Write a CDM as NDM/XML (elements not namespace-qualified).

    Version 1.0 messages use the CCSDS 508.0-B-1 groups and schema location;
    version 2.0 messages use ``physicalParameters`` and
    ``userDefinedParameters``. Comments are moved to the start of their
    element, where the schema allows them.

    Parameters
    ----------
    cdm
        The message.

    Raises
    ------
    CdmWriteError
        If a value or comment contains a character XML 1.0 cannot hold.
    """
    root = ElementTree.Element("cdm", {"id": "CCSDS_CDM_VERS", "version": cdm.version})
    if cdm.version == "1.0":
        root.set("xmlns:xsi", XSI)
        root.set("xsi:noNamespaceSchemaLocation", _CDM_1_0_SCHEMA_LOCATION)
    header_fields = [f for f in cdm.header.fields if f.keyword != "CCSDS_CDM_VERS"]
    _write(ElementTree.SubElement(root, "header"), header_fields, "header", cdm.version)
    body = ElementTree.SubElement(root, "body")
    for tag, section in (
        ("relativeMetadataData", cdm.relative),
        *(("segment", segment) for segment in cdm.segments),
    ):
        _write(
            ElementTree.SubElement(body, tag), section.fields, section.kind, cdm.version
        )
    return tostring(root)


def load(path: str | Path) -> Cdm:
    """Read an XML CDM file (UTF-8, gzip-compressed or not; a
    byte-order mark is ignored).

    Parameters
    ----------
    path
        File to read.
    """
    return load_file(path, loads)


def dump(cdm: Cdm, path: str | Path) -> None:
    """Write a CDM to an XML file (UTF-8).

    Parameters
    ----------
    cdm
        The message.
    path
        File to write.
    """
    dump_file(cdm, path, dumps)
