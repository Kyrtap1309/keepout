"""The ``keepout`` command: its commands, help and version.

Every command is a function of a command module (:mod:`keepout.cli.fetch`,
:mod:`keepout.cli.validate`) registered here through
:func:`~keepout.cli.output.reported`, so keepout's errors end the command
with ``Error: ...`` and status 1 instead of a traceback.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

import typer

from keepout import __version__
from keepout.cli.fetch import fetch_cara, fetch_kelvins
from keepout.cli.output import reported
from keepout.cli.validate import validate_command

__all__ = ["app", "fetch_app", "run"]

app = typer.Typer(
    name="keepout",
    no_args_is_help=True,
    # Locals may hold credentials (Space-Track); a bug's traceback must not
    # print them.
    pretty_exceptions_show_locals=False,
)
"""The ``keepout`` command."""

fetch_app = typer.Typer(
    name="fetch",
    help="Download data into keepout's cache.",
    no_args_is_help=True,
)
"""``keepout fetch``: one command per source."""

app.add_typer(fetch_app)


def _print_version(*, value: bool) -> None:
    if value:
        typer.echo(f"keepout {__version__}")
        raise typer.Exit


@app.callback()
def _keepout(
    *,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_print_version,
            is_eager=True,
            help="Print keepout's version and exit.",
        ),
    ] = False,
) -> None:
    """Spacecraft conjunction assessment: screening, CDMs, probability of
    collision and risk."""


def _add(group: typer.Typer, name: str, command: Callable[..., None]) -> None:
    group.command(name)(reported(command))


_add(fetch_app, "cara", fetch_cara)
_add(fetch_app, "kelvins", fetch_kelvins)
_add(app, "validate", validate_command)


def run() -> None:
    """Run ``keepout`` on the command line arguments; exits with the status."""
    app(prog_name="keepout")
