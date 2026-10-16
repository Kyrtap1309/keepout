"""What the commands print and the status they exit with.

A command prints its result to standard output, as tables and text for
people or, with ``--json``, as one JSON document (:func:`show`). Errors go
to standard error as one line, ``Error: ...``, without a traceback
(:func:`fail`, :func:`reported`).
"""

from __future__ import annotations

import enum
import functools
import json
from collections.abc import Callable, Mapping, Sequence
from typing import NoReturn, ParamSpec, TypeAlias, TypeVar

import typer
from astropy.time import Time
from rich import box
from rich.console import Console, RenderableType
from rich.table import Table

from keepout.errors import KeepoutError

__all__ = [
    "ExitCode",
    "JsonValue",
    "fail",
    "isot",
    "minute",
    "pc_text",
    "plural",
    "reported",
    "show",
    "table",
    "utc",
]


class ExitCode(enum.IntEnum):
    """Exit status of ``keepout``."""

    OK = 0
    """Done."""
    ERROR = 1
    """A data, network or file error (a :class:`~keepout.errors.KeepoutError`
    or :class:`OSError`), a failed validation, or the ``cli`` extra missing."""
    USAGE = 2
    """A wrong command line (Typer's status for it)."""
    MANOEUVRE = 3
    """``keepout assess``: at least one event's decision is to manoeuvre."""


JsonValue: TypeAlias = (
    "str | int | float | bool | Sequence[JsonValue] | Mapping[str, JsonValue] | None"
)
"""A value :func:`json.dumps` writes as JSON."""


def show(
    payload: Mapping[str, JsonValue], *text: RenderableType, as_json: bool
) -> None:
    """Print a command's result.

    Parameters
    ----------
    payload
        The result as JSON values, printed with ``as_json``.
    *text
        The result for people (strings, rich tables), printed otherwise; text
        is printed as is (no rich markup or emoji codes) and long lines are
        not broken.
    as_json
        Whether ``--json`` was given.

    Raises
    ------
    ValueError
        If ``payload`` holds a NaN or an infinity (not valid JSON).
    """
    if as_json:
        typer.echo(json.dumps(payload, indent=2, allow_nan=False))
        return
    console = Console(markup=False, emoji=False, highlight=False, soft_wrap=True)
    for item in text:
        console.print(item)


def fail(message: str, status: ExitCode = ExitCode.ERROR) -> NoReturn:
    """Print ``Error: message`` to standard error and exit with ``status``.

    Raises
    ------
    typer.Exit
        Always.
    """
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(status)


def isot(time: Time) -> str:
    """``time`` in UTC, ISO 8601 to the millisecond, for JSON:
    ``2026-10-06T19:36:42.123``."""
    return str(time.utc.isot)


def utc(time: Time) -> str:
    """``time`` in UTC to the second, ISO 8601: ``2026-10-06T19:36:42``."""
    return isot(time)[:19]


def minute(time: Time) -> str:
    """``time`` in UTC to the minute, for tables: ``2026-10-06 19:36``."""
    return utc(time)[:16].replace("T", " ")


def pc_text(pc: float | None) -> str:
    """A Pc for tables, ``1.23e-05``, or ``-`` if there is none."""
    return "-" if pc is None else f"{pc:.2e}"


def plural(count: int, noun: str, nouns: str | None = None) -> str:
    """``count`` and ``noun``, or its plural ``nouns`` (default: ``noun`` and
    ``s``) unless ``count`` is one: ``1 event``, ``2 events``."""
    if count == 1:
        return f"{count} {noun}"
    return f"{count} {nouns or noun + 's'}"


def table(title: str) -> Table:
    """An empty table with a header rule and no borders, narrow enough for
    rows of events in 80 columns."""
    return Table(
        title=title,
        box=box.SIMPLE_HEAD,
        show_edge=False,
        pad_edge=False,
        collapse_padding=True,
    )


_P = ParamSpec("_P")
_R = TypeVar("_R")


def reported(command: Callable[_P, _R]) -> Callable[_P, _R]:
    """``command``, reporting its keepout and file errors with :func:`fail`.

    A :class:`~keepout.errors.KeepoutError` or :class:`OSError` becomes
    ``Error: <message>`` and :attr:`ExitCode.ERROR`; anything else is a bug
    and keeps its traceback. Every command is registered through it
    (:mod:`keepout.cli.app`).
    """

    @functools.wraps(command)
    def run(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return command(*args, **kwargs)
        except (KeepoutError, OSError) as exc:
            fail(str(exc))

    return run
