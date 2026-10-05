"""Shared by the report tools: README blocks and writing or checking files."""

from __future__ import annotations

from collections.abc import Mapping
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


def write_or_check(files: Mapping[Path, str], *, check: bool, tool: str) -> int:
    """Write the files whose contents differ, or with ``check`` report them.

    Returns
    -------
    int
        1 if ``check`` found a file out of date, else 0.
    """
    status = 0
    for path, text in files.items():
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == text:
            continue
        if check:
            print(f"{path} is out of date; run {tool}")
            status = 1
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path}")
    return status
