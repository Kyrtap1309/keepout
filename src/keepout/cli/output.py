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
from typing import TYPE_CHECKING, Final, NoReturn, ParamSpec, TypeAlias, TypeVar

import typer
from rich import box
from rich.console import Console, RenderableType
from rich.table import Table

from keepout._text import pc_text, plural
from keepout.errors import KeepoutError
from keepout.units import DAY, HOUR, MINUTE

if TYPE_CHECKING:
    from astropy.time import Time

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


ASSUMED_COVARIANCES: Final = (
    "Covariances are assumed (keepout's model of GP element set errors), "
    "not orbit determination."
)
"""The line under a Pc on the synthetic covariance model."""


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


def objects(cdm: Cdm) -> str:
    """The pair of a CDM: ``"<object1> / <object2>"``."""
    return f"{cdm.object1.object_id} / {cdm.object2.object_id}"


def duration_words(seconds: float) -> str:
    """A whole number of the largest unit of days, hours and minutes that
    ``seconds`` holds, in words: ``a day``, ``an hour``, ``10 minutes``
    (else seconds, ``90 s``); for help texts."""
    for unit, noun, one in ((DAY, "day", "a day"), (HOUR, "hour", "an hour")):
        count = seconds / unit
        if count >= 1.0 and count.is_integer():
            return one if count == 1.0 else f"{count:g} {noun}s"
    count = seconds / MINUTE
    if count >= 1.0 and count.is_integer():
        return "a minute" if count == 1.0 else f"{count:g} minutes"
    return f"{seconds:g} s"


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
