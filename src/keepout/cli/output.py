"""What the commands print and the status they exit with.

A command prints its result to standard output, as tables and text for
people or, with ``--json``, as one JSON document (:func:`show`). Errors go
to standard error as one line, ``Error: ...``, without a traceback
(:func:`fail`, :func:`reported`).
"""

from __future__ import annotations

import enum
import functools
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, NoReturn, ParamSpec, TypeAlias, TypeVar

import typer
from rich import box
from rich.console import Console, RenderableType
from rich.table import Table

from keepout._text import (
    ASSUMED_COVARIANCES,
    duration_words,
    isot,
    minute,
    pc_text,
    plural,
    utc,
)
from keepout.errors import HardBodyRadiusError, KeepoutError

if TYPE_CHECKING:
    from keepout.cdm import Cdm

__all__ = [
    "ASSUMED_COVARIANCES",
    "ExitCode",
    "JsonOut",
    "duration_words",
    "fail",
    "isot",
    "minute",
    "objects",
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
    MANEUVER = 3
    """``keepout assess``: at least one event's decision is to manoeuvre."""
    NO_SAFE_BURN = 4
    """``keepout maneuver``: no burn exists, the burn given misses the target,
    or every burn brings an approach at red or the event's pair back above the
    target."""


JsonOut: TypeAlias = (
    "str | int | float | bool | Sequence[JsonOut] | Mapping[str, JsonOut] | None"
)
"""A value :func:`json.dumps` writes as JSON: what a command prints with
``--json``. Unlike :data:`keepout._types.JsonValue`, what :func:`json.loads`
returns (``dict`` and ``list``, to narrow with ``isinstance``), it takes any
sequence and mapping, so a command can pass tuples and read-only mappings."""


def show(payload: Mapping[str, JsonOut], *text: RenderableType, as_json: bool) -> None:
    """The output of a command: its result as JSON or as text for people.

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
    """The end of a failed command: ``Error: message`` on standard error, then
    an exit with ``status``.

    Raises
    ------
    typer.Exit
        Always.
    """
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(status)


def objects(cdm: Cdm) -> str:
    """The pair of a CDM: ``"<object1> / <object2>"``."""
    return f"{cdm.object1.object_id} / {cdm.object2.object_id}"


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


def _radius_hint(command: Callable[..., object]) -> str:
    """How a user of ``command`` gives the combined hard-body radius: the
    options it has for it (``--hbr``, ``hbr_m`` in ``--settings``)."""
    options = inspect.signature(command).parameters
    ways = [
        way
        for name, way in (
            ("hbr", "pass --hbr METRES"),
            ("settings", "set hbr_m in the --settings file"),
        )
        if name in options
    ]
    return " or ".join(ways) or "give HBR for both objects in the CDMs"


def reported(command: Callable[_P, _R]) -> Callable[_P, _R]:
    """``command``, reporting its keepout and file errors with :func:`fail`.

    A :class:`~keepout.errors.KeepoutError` or :class:`OSError` becomes
    ``Error: <message>`` and :attr:`ExitCode.ERROR`; anything else is a bug
    and keeps its traceback. A
    :class:`~keepout.errors.HardBodyRadiusError` keeps its detail with the
    command's own way to give the radius instead of the library's keyword
    argument. Every command is registered through it
    (:mod:`keepout.cli.app`).
    """
    hint = _radius_hint(command)

    @functools.wraps(command)
    def run(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return command(*args, **kwargs)
        except HardBodyRadiusError as exc:
            fail(f"{exc.detail}: {hint}")
        except (KeepoutError, OSError) as exc:
            fail(str(exc))

    return run
