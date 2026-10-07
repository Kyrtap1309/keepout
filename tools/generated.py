"""Shared by the report tools: README blocks, Markdown tables, links and
writing or checking files."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path


def replace_block(text: str, start: str, end: str, block: str) -> str:
    """Put ``block`` between the ``start`` and ``end`` markers of ``text``.

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


def table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    """The lines of a Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def relative(path: Path, start: Path) -> str:
    """``path`` as a link from a file in ``start``."""
    return Path(os.path.relpath(path, start)).as_posix()


def write_or_check(files: Mapping[Path, str | bytes], *, check: bool, tool: str) -> int:
    """Write the files whose contents differ, or with ``check`` report them.

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
