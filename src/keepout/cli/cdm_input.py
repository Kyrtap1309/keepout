"""The CDMs a command reads: files, KVN or XML."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TypeAlias, TypeVar

import typer

from keepout.cli.output import fail
from keepout.errors import KeepoutError

# The library is imported in the function: keepout --help need not load the
# numerical libraries.
if TYPE_CHECKING:
    from keepout.cdm import Cdm

__all__ = ["CdmFiles", "load_cdms", "read_cdm_files"]

_T = TypeVar("_T")

CdmFiles: TypeAlias = Annotated[
    list[Path],
    typer.Argument(
        exists=True,
        dir_okay=False,
        metavar="CDM...",
        help="CDM files, KVN or XML, gzipped or not.",
    ),
]
"""The CDM files a command reads."""


def read_cdm_files(files: Sequence[Path], read: Callable[[Path], _T]) -> list[_T]:
    """``read`` of each of ``files``; one that cannot be read ends the
    command, named.

    Raises
    ------
    typer.Exit
        If ``read`` raises a :class:`~keepout.errors.KeepoutError`:
        ``Error: <file>: <reason>``, status 1.
    """
    messages = []
    for path in files:
        try:
            messages.append(read(path))
        except KeepoutError as exc:
            fail(f"{path}: {exc}")
    return messages


def load_cdms(files: Sequence[Path]) -> list[Cdm]:
    """The CDMs of ``files``; one that cannot be read ends the command, named.

    Raises
    ------
    typer.Exit
        If a file cannot be read as a CDM: ``Error: <file>: <reason>``, status 1.
    """
    from keepout import cdm

    return read_cdm_files(files, cdm.load)
