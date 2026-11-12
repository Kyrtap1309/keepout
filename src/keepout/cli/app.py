"""The ``keepout`` command: its commands, help and version.

Every command is a function of a command module (:mod:`keepout.cli.fetch`,
:mod:`keepout.cli.screen`, :mod:`keepout.cli.assess`,
:mod:`keepout.cli.maneuver`, :mod:`keepout.cli.validate`,
:mod:`keepout.cli.store`, :mod:`keepout.cli.jobs`, :mod:`keepout.cli.worker`)
registered here through :func:`~keepout.cli.output.reported`, so keepout's
errors end the command with ``Error: ...`` and status 1 instead of a
traceback.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

import typer

import keepout
from keepout.cli.assess import assess_command
from keepout.cli.fetch import fetch_cara, fetch_catalog, fetch_cdm, fetch_kelvins
from keepout.cli.jobs import jobs_cancel, jobs_list
from keepout.cli.maneuver import maneuver_command
from keepout.cli.output import reported
from keepout.cli.screen import screen_command
from keepout.cli.store import store_add, store_purge, store_rebuild, store_upgrade
from keepout.cli.validate import validate_command
from keepout.cli.worker import worker_command

__all__ = ["app", "fetch_app", "jobs_app", "run", "store_app"]

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

store_app = typer.Typer(
    name="store",
    help=(
        "Keep events, CDMs and their Pc in a conjunction store, a SQLite file "
        "(needs the store extra)."
    ),
    no_args_is_help=True,
)
"""``keepout store``: the conjunction store's commands."""

app.add_typer(store_app)

jobs_app = typer.Typer(
    name="jobs",
    help="List and cancel the store's background jobs (needs the store extra).",
    no_args_is_help=True,
)
"""``keepout jobs``: the job queue's commands."""

app.add_typer(jobs_app)


def _print_version(*, value: bool) -> None:
    if value:
        typer.echo(f"keepout {keepout.__version__}")
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


_add(fetch_app, "catalog", fetch_catalog)
_add(fetch_app, "cdm", fetch_cdm)
_add(fetch_app, "cara", fetch_cara)
_add(fetch_app, "kelvins", fetch_kelvins)
_add(app, "screen", screen_command)
_add(app, "assess", assess_command)
_add(app, "maneuver", maneuver_command)
_add(app, "validate", validate_command)
_add(app, "worker", worker_command)
_add(store_app, "add", store_add)
_add(store_app, "upgrade", store_upgrade)
_add(store_app, "rebuild", store_rebuild)
_add(store_app, "purge", store_purge)
_add(jobs_app, "list", jobs_list)
_add(jobs_app, "cancel", jobs_cancel)


def run() -> None:
    """Run the ``keepout`` command on the command line arguments and exit
    with its status."""
    app(prog_name="keepout")
