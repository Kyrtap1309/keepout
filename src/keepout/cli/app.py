"""The ``keepout`` command: its commands, help and version.

Every command is a function of a command module (:mod:`keepout.cli.fetch`,
:mod:`keepout.cli.screen`, :mod:`keepout.cli.assess`,
:mod:`keepout.cli.maneuver`, :mod:`keepout.cli.validate`,
:mod:`keepout.cli.store`, :mod:`keepout.cli.jobs`, :mod:`keepout.cli.worker`,
:mod:`keepout.cli.dashboard`)
registered here through :func:`~keepout.cli.output.reported`, so keepout's
errors end the command with ``Error: ...`` and status 1 instead of a
traceback.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Annotated, Any, ClassVar, Final, cast

import typer
from typer.core import TyperArgument, TyperCommand, TyperOption

import keepout
from keepout._typing import override
from keepout.cli.assess import assess_command
from keepout.cli.dashboard import dashboard_command
from keepout.cli.fetch import fetch_cara, fetch_catalog, fetch_cdm, fetch_kelvins
from keepout.cli.jobs import jobs_cancel, jobs_list
from keepout.cli.maneuver import HELP_PANELS as MANEUVER_PANELS
from keepout.cli.maneuver import maneuver_command
from keepout.cli.output import ExitCode, reported
from keepout.cli.screen import HELP_PANELS as SCREEN_PANELS
from keepout.cli.screen import screen_command
from keepout.cli.store import store_add, store_purge, store_rebuild, store_upgrade
from keepout.cli.validate import validate_command
from keepout.cli.worker import worker_command

__all__ = ["app", "fetch_app", "jobs_app", "run", "store_app"]

# The help is Rich markup: a square bracket in it is escaped (\[), and an
# epilog keeps its line breaks.
_EPILOG: Final = f"""Exit status:
  {ExitCode.OK:d}  done
  {ExitCode.ERROR:d}  an error in the data, a download or a file
  {ExitCode.USAGE:d}  a usage error: an unknown option, a bad value, a missing file
  {ExitCode.MANEUVER:d}  assess, store add: at least one event needs a maneuver
  {ExitCode.NO_SAFE_BURN:d}  maneuver: no burn can be recommended

Environment:
  SPACETRACK_IDENTITY  the Space-Track account (--source spacetrack, fetch)
  SPACETRACK_PASSWORD  its password
  KEEPOUT_CACHE_DIR    the cache root, as --cache-dir
  KEEPOUT_STORE        the store, as --store (store, jobs, worker, dashboard,
                       maneuver --event)

Start with keepout assess CDM... or keepout screen --fleet FILE; keepout
COMMAND --help gives each command's options. The guides are in the
repository's docs/ folder."""
"""The end of ``keepout --help``: the exit statuses, the environment
variables and where to start."""

_EXAMPLES: Final = {
    "screen": (
        "keepout screen --fleet fleet.toml --days 7 --cdm-dir cdms",
        (
            "keepout screen --fleet fleet.toml --catalog catalog.csv.gz "
            "--start 2026-10-06 --store keepout.db"
        ),
    ),
    "assess": (
        "keepout assess cdms/*.kvn",
        "keepout assess cdms/*.kvn --hbr 1000 --json",
    ),
    "maneuver": (
        (
            "keepout maneuver cdms/44476_55841_*.kvn --target-pc 1e-7 "
            "--catalog catalog.csv.gz"
        ),
        "keepout maneuver --event 3befe6f1305d --store keepout.db --queue",
    ),
    "store add": ("keepout store add cdms/*.kvn --store keepout.db",),
    "validate": ("keepout validate", "keepout validate --json --workers 2"),
    "fetch catalog": (
        "keepout fetch catalog",
        "keepout fetch catalog --source spacetrack --json",
    ),
    "fetch cdm": ("keepout fetch cdm",),
    "fetch cara": ("keepout fetch cara --cache-dir ~/keepout-cache",),
    "fetch kelvins": ("keepout fetch kelvins",),
    "jobs list": ("keepout jobs list --store keepout.db",),
    "jobs cancel": ("keepout jobs cancel 1 --store keepout.db",),
    "dashboard": (
        "keepout dashboard --store keepout.db",
        "keepout dashboard --store keepout.db --as-of 2026-10-07T12:00 --port 8502",
    ),
}
"""Example command lines at the end of a command's help, by command."""


def _examples(command: str) -> str | None:
    """The epilog of ``command``'s help: its examples, if any."""
    lines = _EXAMPLES.get(command)
    if lines is None:
        return None
    # Unindented: Typer strips the indent common to the lines after the first.
    return "\n".join(["Examples:", *lines])


class _Command(TyperCommand):
    """A command whose usage line names its arguments as the help does:
    ``CDM...``, not Typer's ``{CDM...}`` for a required one, and whose help
    lists its parameters in the panels of :attr:`panels`."""

    panels: ClassVar[Mapping[str, str]] = {}
    """The help panel of each parameter listed, by name; the others are in
    Typer's own."""

    @override
    def collect_usage_pieces(self, ctx: object) -> list[str]:
        """The usage line after the command name, braces removed."""
        # The context is of the Click that Typer vendors, which keepout does
        # not import: typed as any object here, passed on as it came.
        pieces: list[str] = super().collect_usage_pieces(cast("Any", ctx))
        return [
            piece[1:-1] if piece.startswith("{") and piece.endswith("}") else piece
            for piece in pieces
        ]

    @override
    def format_help(self, ctx: object, formatter: object) -> None:
        """The help, each parameter of :attr:`panels` in its panel."""
        for param in self.params:
            panel = self.panels.get(param.name or "")
            if panel is not None and isinstance(param, (TyperOption, TyperArgument)):
                param.rich_help_panel = panel
        super().format_help(cast("Any", ctx), cast("Any", formatter))


app = typer.Typer(
    name="keepout",
    no_args_is_help=True,
    epilog=_EPILOG,
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


def _print_version(*, value: bool | None) -> None:
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


def _panelled(by_name: Mapping[str, str]) -> type[_Command]:
    """A :class:`_Command` whose help lists parameters in the panels of
    ``by_name``."""

    class Panelled(_Command):
        panels = by_name

    return Panelled


def _add(
    group: typer.Typer,
    name: str,
    command: Callable[..., None],
    panels: Mapping[str, str] | None = None,
) -> None:
    full = name if group is app else f"{group.info.name} {name}"
    cls = _Command if panels is None else _panelled(panels)
    group.command(name, cls=cls, epilog=_examples(full))(reported(command))


_add(fetch_app, "catalog", fetch_catalog)
_add(fetch_app, "cdm", fetch_cdm)
_add(fetch_app, "cara", fetch_cara)
_add(fetch_app, "kelvins", fetch_kelvins)
_add(app, "screen", screen_command, SCREEN_PANELS)
_add(app, "assess", assess_command)
_add(app, "maneuver", maneuver_command, MANEUVER_PANELS)
_add(app, "validate", validate_command)
_add(app, "worker", worker_command)
_add(app, "dashboard", dashboard_command)
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
