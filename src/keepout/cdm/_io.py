"""File encoding shared by the CDM readers and writers."""

from __future__ import annotations

from pathlib import Path


def read_text(path: str | Path) -> str:
    """Read a UTF-8 message; a byte-order mark is ignored."""
    return Path(path).read_text(encoding="utf-8-sig")


def write_text(path: str | Path, text: str) -> None:
    """Write a message as UTF-8."""
    Path(path).write_text(text, encoding="utf-8")
