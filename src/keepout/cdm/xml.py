"""CDM NDM/XML reader and writer.

The reader flattens grouping elements, so it accepts CCSDS 508.0-B-1 messages
with or without the ``ndm:`` namespace (Technical Corrigendum 2) and CDM 2.0
groups such as ``physicalParameters`` and ``userDefinedParameters``. DOCTYPE
declarations are rejected: CDMs never need them, and refusing them rules out
entity-expansion attacks. No XSD validation is performed.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from xml.etree import ElementTree

from keepout.cdm.keywords import KEYWORDS, SectionKind, format_value, parse_value
from keepout.cdm.model import COMMENT, USER_DEFINED_PREFIX, Cdm, Field, Section
from keepout.errors import CdmParseError

NDM_NAMESPACE = "urn:ccsds:schema:ndmxml"
_XSI = "http://www.w3.org/2001/XMLSchema-instance"
_SCHEMA_1_0 = (
    "https://sanaregistry.org/r/ndmxml_unqualified/ndmxml-2.0.0-master-2.0.xsd"
)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _single(parent: ElementTree.Element, name: str, path: str) -> ElementTree.Element:
    found = [child for child in parent if _local(child.tag) == name]
    if len(found) != 1:
        raise CdmParseError(
            f"expected exactly one <{name}>, found {len(found)}", path=path
        )
    return found[0]


def _leaves(
    element: ElementTree.Element, path: str
) -> Iterator[tuple[ElementTree.Element, str]]:
    for child in element:
        child_path = f"{path}/{_local(child.tag)}"
        if len(child):
            yield from _leaves(child, child_path)
        else:
            yield child, child_path


def _fields(
    element: ElementTree.Element, kind: SectionKind, path: str
) -> tuple[Field, ...]:
    table = KEYWORDS[kind]
    fields: list[Field] = []
    seen: set[str] = set()
    for leaf, leaf_path in _leaves(element, path):
        keyword = _local(leaf.tag)
        text = (leaf.text or "").strip()
        if keyword == COMMENT:
            fields.append(Field(COMMENT, text))
            continue
        if keyword in seen:
            raise CdmParseError("duplicate keyword", path=leaf_path, keyword=keyword)
        seen.add(keyword)
        unit = leaf.get("units")
        spec = table.get(keyword)
        if spec is None:
            fields.append(Field(keyword, text, unit))
            continue
        try:
            fields.append(Field(keyword, parse_value(spec, text, unit)))
        except ValueError as exc:
            raise CdmParseError(str(exc), path=leaf_path, keyword=keyword) from exc
    return tuple(fields)


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
    if "<!DOCTYPE" in text.upper():
        raise CdmParseError("DOCTYPE declarations are not allowed in CDM XML")
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        raise CdmParseError(f"malformed XML: {exc}") from exc
    if _local(root.tag) != "cdm":
        raise CdmParseError(f"root element must be <cdm>, got <{_local(root.tag)}>")
    version = root.get("version")
    if not version:
        raise CdmParseError("<cdm> has no version attribute")
    header = _single(root, "header", "cdm")
    body = _single(root, "body", "cdm")
    relative = _single(body, "relativeMetadataData", "cdm/body")
    segments = [child for child in body if _local(child.tag) == "segment"]
    if len(segments) != 2:
        raise CdmParseError(
            f"expected 2 <segment> elements, found {len(segments)}", path="cdm/body"
        )
    header_fields = (
        Field("CCSDS_CDM_VERS", version),
        *_fields(header, "header", "cdm/header"),
    )
    try:
        return Cdm(
            Section("header", header_fields),
            Section(
                "relative",
                _fields(relative, "relative", "cdm/body/relativeMetadataData"),
            ),
            (
                Section(
                    "segment", _fields(segments[0], "segment", "cdm/body/segment[1]")
                ),
                Section(
                    "segment", _fields(segments[1], "segment", "cdm/body/segment[2]")
                ),
            ),
        )
    except ValueError as exc:
        raise CdmParseError(str(exc)) from exc


def _groups(
    fields: Sequence[Field], kind: SectionKind, version: str
) -> list[tuple[str, ...]]:
    """XML group path of each field.

    Known keywords use the table; ``USER_DEFINED_*`` go to
    ``userDefinedParameters``; other unknown keywords stay in the group of the
    field before them; comments take the group of the field after them.
    """
    table = KEYWORDS[kind]
    default: tuple[str, ...] = ("metadata",) if kind == "segment" else ()
    physical = (
        "physicalParameters" if version.startswith("2") else "additionalParameters"
    )
    resolved: list[tuple[str, ...] | None] = []
    previous = default
    for field in fields:
        if field.keyword == COMMENT:
            resolved.append(None)
            continue
        spec = table.get(field.keyword)
        if spec is not None:
            group = tuple(
                physical if name == "additionalParameters" else name
                for name in spec.group
            )
        elif kind == "segment" and field.keyword.startswith(USER_DEFINED_PREFIX):
            group = ("data", "userDefinedParameters")
        else:
            group = previous
        resolved.append(group)
        previous = group
    groups: list[tuple[str, ...]] = []
    for index, candidate in enumerate(resolved):
        if candidate is None:
            upcoming = next((g for g in resolved[index + 1 :] if g is not None), None)
            if upcoming is not None:
                candidate = upcoming
            else:
                candidate = groups[-1] if groups else default
        groups.append(candidate)
    return groups


def _write(
    parent: ElementTree.Element,
    fields: Sequence[Field],
    kind: SectionKind,
    version: str,
) -> None:
    table = KEYWORDS[kind]
    stack: list[tuple[str, ElementTree.Element]] = []
    for field, group in zip(fields, _groups(fields, kind, version), strict=True):
        depth = 0
        while depth < min(len(stack), len(group)) and stack[depth][0] == group[depth]:
            depth += 1
        del stack[depth:]
        for name in group[depth:]:
            container = stack[-1][1] if stack else parent
            stack.append((name, ElementTree.SubElement(container, name)))
        element = ElementTree.SubElement(
            stack[-1][1] if stack else parent, field.keyword
        )
        if field.keyword == COMMENT:
            element.text = str(field.value)
            continue
        spec = table.get(field.keyword)
        if spec is None:
            text, unit = str(field.value), field.unit
        else:
            text, unit = format_value(spec, field.value)
        element.text = text
        if unit is not None:
            element.set("units", unit)


def dumps(cdm: Cdm) -> str:
    """Write a CDM as NDM/XML (elements not namespace-qualified).

    Version 1.0 messages use the CCSDS 508.0-B-1 groups and schema location;
    version 2.0 messages use ``physicalParameters`` and
    ``userDefinedParameters``.

    Parameters
    ----------
    cdm
        The message.
    """
    root = ElementTree.Element("cdm", {"id": "CCSDS_CDM_VERS", "version": cdm.version})
    if cdm.version == "1.0":
        root.set("xmlns:xsi", _XSI)
        root.set("xsi:noNamespaceSchemaLocation", _SCHEMA_1_0)
    header_fields = [f for f in cdm.header.fields if f.keyword != "CCSDS_CDM_VERS"]
    _write(ElementTree.SubElement(root, "header"), header_fields, "header", cdm.version)
    body = ElementTree.SubElement(root, "body")
    _write(
        ElementTree.SubElement(body, "relativeMetadataData"),
        cdm.relative.fields,
        "relative",
        cdm.version,
    )
    for segment in cdm.segments:
        _write(
            ElementTree.SubElement(body, "segment"),
            segment.fields,
            "segment",
            cdm.version,
        )
    ElementTree.indent(root)
    declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
    return declaration + ElementTree.tostring(root, encoding="unicode") + "\n"


def load(path: str | Path) -> Cdm:
    """Read an XML CDM file (UTF-8; a byte-order mark is ignored).

    Parameters
    ----------
    path
        File to read.
    """
    return loads(Path(path).read_text(encoding="utf-8-sig"))


def dump(cdm: Cdm, path: str | Path) -> None:
    """Write a CDM to an XML file (UTF-8).

    Parameters
    ----------
    cdm
        The message.
    path
        File to write.
    """
    Path(path).write_text(dumps(cdm), encoding="utf-8")
