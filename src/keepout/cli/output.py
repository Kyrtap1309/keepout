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
from rich.console import Console, RenderableType

from keepout.errors import KeepoutError

__all__ = ["ExitCode", "JsonValue", "fail", "reported", "show"]


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
