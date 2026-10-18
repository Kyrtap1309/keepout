"""Shared by the report tools: the repository's paths, command-line counts,
JSON read back, README blocks, Markdown tables, links, plain-text numbers
and writing or checking files."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, TypeVar

from typeguard import CollectionCheckStrategy, TypeCheckError, check_type

REPO: Final = Path(__file__).resolve().parents[1]
"""The repository's root directory."""

README: Final = REPO / "README.md"
"""The README, whose summary blocks the report tools write."""

_T = TypeVar("_T")


def positive(text: str) -> int:
    """A command-line count: an integer of at least 1.

    Raises
    ------
    argparse.ArgumentTypeError
        If the integer in ``text`` is less than 1.
    """
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def json_text(value: object) -> str:
    """``value`` as the tools write JSON files: indented by 2, with a final
    newline."""
    return json.dumps(value, indent=2) + "\n"


def parse_json(text: str, shape: type[_T], source: str) -> _T:
    """JSON ``text`` checked to be a ``shape`` (a TypedDict) throughout: every
    key there and no other, every list item, dict entry and value of the type
    it is declared with.

    Raises
    ------
    ValueError
        If ``text`` is not JSON, or not a ``shape``; the message names
        ``source``.
    """
    try:
        return check_type(
            json.loads(text),
            shape,
            collection_check_strategy=CollectionCheckStrategy.ALL_ITEMS,
        )
    except TypeCheckError as exc:
        raise ValueError(f"{source} is not a {shape.__name__}: {exc}") from exc


def read_json(path: Path, shape: type[_T]) -> _T:
    """The JSON file at ``path``, checked to be a ``shape`` (:func:`parse_json`).

    Raises
    ------
    ValueError
        If the file is not JSON, or not a ``shape``.
    """
    return parse_json(path.read_text(encoding="utf-8"), shape, str(path))


def replace_block(text: str, start: str, end: str, block: str) -> str:
    """``text`` with ``block`` between its ``start`` and ``end`` markers.

    Raises
    ------
    ValueError
        If the markers are missing or out of order.
    """
    first = text.find(start)
    last = text.find(end)
    if first < 0 or last < first:
        raise ValueError(f"README needs {start} and {end}, in that order")
    head = text[: first + len(start)]
    return f"{head}\n{block}\n{text[last:]}"


def readme_with(readme: Path, start: str, end: str, block: str) -> str:
    """The text of the file ``readme`` with ``block`` between its ``start``
    and ``end`` markers (:func:`replace_block`).

    Raises
    ------
    ValueError
        If the markers are missing or out of order.
    """
    return replace_block(readme.read_text(encoding="utf-8"), start, end, block)


def table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    """The lines of a Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def relative(path: Path, start: Path) -> str:
    """``path`` as a link from a file in ``start``."""
    return Path(os.path.relpath(path, start)).as_posix()


def plural(count: int, noun: str, nouns: str | None = None) -> str:
    """``count`` and ``noun``, or its plural ``nouns`` (default: ``noun`` and
    ``s``) unless ``count`` is one: ``1 event``, ``2 events``. As
    :func:`keepout.cli.output.plural`, which the tools do not import (it
    needs the ``cli`` extra)."""
    if count == 1:
        return f"{count} {noun}"
    return f"{count} {nouns or noun + 's'}"


def pc_text(pc: float | None) -> str:
    """A Pc for tables, ``1.23e-05``, or ``-`` if there is none (as
    :func:`keepout.cli.output.pc_text`)."""
    return "-" if pc is None else f"{pc:.2e}"


def write_or_check(files: Mapping[Path, str | bytes], *, check: bool, tool: str) -> int:
    """The files whose contents differ, written, or with ``check`` reported.

    Text is written as UTF-8, bytes as they are.

    Returns
    -------
    int
        1 if ``check`` found a file out of date, else 0.
    """
    status = 0
    for path, content in files.items():
        data = content.encode() if isinstance(content, str) else content
        if path.exists() and path.read_bytes() == data:
            continue
        if check:
            print(f"{path} is out of date; run {tool}")
            status = 1
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            print(f"wrote {path}")
    return status
