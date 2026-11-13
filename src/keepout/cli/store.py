"""``keepout store``: the conjunction store (the ``store`` extra) on the
command line, and the ``--store`` option of ``keepout assess``,
``keepout screen`` and ``keepout maneuver``.

``keepout.store`` is imported in the functions, never here: ``keepout
--help`` loads neither it nor SQLAlchemy, and without the extra ``--store``
says how to install it (:func:`open_cli_store`).
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, NoReturn, TypeAlias

import typer

from keepout.cli.cdm_input import CdmFiles, read_cdm_files
from keepout.cli.events import Assessed, show_assessed
from keepout.cli.options import (
    JsonFlag,
    SettingsFile,
    dubious_years_ignored,
    operator_settings,
    time_option,
)
from keepout.cli.output import JsonOut, fail, isot, plural, show, utc

if TYPE_CHECKING:
    from keepout.events import ConjunctionEvent
    from keepout.settings import Settings
    from keepout.sources.catalog import CatalogReference
    from keepout.store import IngestReport, Store
    from keepout.synthetic import OperatorEvent

__all__ = [
    "STORE_HINT",
    "STORE_VARIABLE",
    "Before",
    "EventStore",
    "StoreOption",
    "StorePath",
    "Yes",
    "add_and_show",
    "importing_store",
    "ingest_screened",
    "open_cli_store",
    "store_add",
    "store_json",
    "store_lines",
    "store_purge",
    "store_rebuild",
    "store_upgrade",
]

STORE_HINT: Final = (
    "--store needs the store extra: "
    'pip install "keepout[cli,store] @ git+https://github.com/Kyrtap1309/keepout"'
)
"""What a command given a store prints (exit status 1) when the extra is
missing."""

_STORE_MODULES: Final = frozenset({"sqlalchemy", "alembic"})
"""The top-level packages of the ``store`` extra."""

STORE_VARIABLE: Final = "KEEPOUT_STORE"
"""The environment variable ``--store`` is read from when not given, by the
``keepout store`` commands and ``keepout maneuver --event`` only."""

_STORE_HELP: Final = "The conjunction store, a SQLite file"


def _without_extra(exc: ModuleNotFoundError) -> NoReturn:
    """The end of a command that needs the ``store`` extra and lacks it.

    Raises
    ------
    typer.Exit
        If ``exc`` is about a package of the extra: :data:`STORE_HINT`,
        status 1.
    ModuleNotFoundError
        Otherwise: ``exc`` again.
    """
    if (exc.name or "").partition(".")[0] not in _STORE_MODULES:
        raise exc
    fail(STORE_HINT)


@contextmanager
def importing_store() -> Generator[None]:
    """A block importing ``keepout.store`` or its modules, which a command
    given a store runs: without the ``store`` extra, it ends the command
    with :data:`STORE_HINT`.

    Raises
    ------
    typer.Exit
        If the block cannot import a package of the extra: status 1.
    ModuleNotFoundError
        If it cannot import another module.
    """
    try:
        yield
    except ModuleNotFoundError as exc:
        _without_extra(exc)


def _store_extra(ctx: typer.Context, value: Path | None) -> Path | None:
    """``--store``, checked to have the ``store`` extra to open it with.

    Raises
    ------
    typer.Exit
        Without the extra: :data:`STORE_HINT`, status 1.
    """
    if value is not None and not ctx.resilient_parsing:
        with importing_store():
            import keepout.store  # noqa: F401 (the check)
    return value


StoreOption: TypeAlias = Annotated[
    Path | None,
    typer.Option(
        "--store",
        dir_okay=False,
        metavar="FILE",
        callback=_store_extra,
        help=(
            f"{_STORE_HELP} (made if missing): add the CDMs to it, each in its "
            f"stored event (needs the store extra; not read from ${STORE_VARIABLE})."
        ),
    ),
]
"""``--store`` of ``assess`` and ``screen``: a store to add the CDMs to;
``None``: no store. Only given on the command line: ``$KEEPOUT_STORE``
never turns these commands into writes to a store."""

StorePath: TypeAlias = Annotated[
    Path,
    typer.Option(
        "--store",
        envvar=STORE_VARIABLE,
        dir_okay=False,
        metavar="FILE",
        callback=_store_extra,
        help=f"{_STORE_HELP}.",
    ),
]
"""``--store`` of the ``keepout store`` commands: required, also read from
``$KEEPOUT_STORE``."""

EventStore: TypeAlias = Annotated[
    Path | None,
    typer.Option(
        "--store",
        envvar=STORE_VARIABLE,
        dir_okay=False,
        metavar="FILE",
        help=f"{_STORE_HELP}, to read --event from (needs the store extra).",
    ),
]
"""``--store`` of ``maneuver``: the store its ``--event`` is read from, also
read from ``$KEEPOUT_STORE``; ``None``: no store. The ``store`` extra is
checked only when ``--event`` opens it (:func:`open_cli_store`), so CDM
files plan without it whatever the variable says."""

Before: TypeAlias = Annotated[
    str,
    typer.Option(
        "--before",
        metavar="TIME",
        help="Delete the events whose latest TCA is before this time, ISO 8601 UTC.",
    ),
]
"""``--before``: the time of ``store purge``, read by
:func:`~keepout.cli.options.parse_time`."""

Yes: TypeAlias = Annotated[
    bool,
    typer.Option("--yes", help="Delete without asking."),
]
"""``--yes``: purge without the question."""


def open_cli_store(path: Path, *, create: bool = True) -> Store:
    """The store at ``path``, opened (:func:`keepout.store.open_store`).

    Parameters
    ----------
    path
        The database file.
    create
        Whether a missing file is made a new store.

    Raises
    ------
    typer.Exit
        Without the ``store`` extra: :data:`STORE_HINT`, status 1.
    StoreError
        If the store cannot be opened, e.g. it was made by an older keepout
        (``keepout store upgrade``) or is missing and ``create`` is false.
    """
    with importing_store():
        from keepout.store import open_store
    return open_store(path, create=create)


def ingest_screened(
    store: Store,
    approaches: Sequence[OperatorEvent],
    catalog: CatalogReference,
    settings: Settings,
) -> tuple[IngestReport, list[ConjunctionEvent]]:
    """Ingest the CDMs a screening issued, with the catalog it screened
    against (copied beside the store), and cache the Pc it computed of them
    with ``settings``.

    Returns
    -------
    tuple[IngestReport, list[ConjunctionEvent]]
        What the ingest did, and the stored events it touched, in the
        store's order (by first TCA), each with every CDM stored of it.

    Raises
    ------
    StoreError
        As :meth:`~keepout.store.Store.ingest` and
        :meth:`~keepout.store.Store.cache`.
    """
    from keepout.store import CdmInput

    report = store.ingest(
        [CdmInput.of(item.cdm, origin="screen") for item in approaches],
        catalog=catalog,
    )
    store.cache(settings, (item.observation for item in approaches))
    return report, store.events(event_ids=report.events)


def store_json(
    path: Path, report: IngestReport, events: Sequence[str]
) -> dict[str, JsonOut]:
    """What an ingest did, as JSON values: the ``"store"`` of ``--json``;
    ``events`` are the ids of the events it touched, in the store's order."""
    return {
        "path": str(path),
        "added": list(report.added),
        "already_stored": list(report.already_stored),
        "events": list(events),
        "new_events": list(report.new_events),
    }


def store_lines(
    path: Path, report: IngestReport, rows: Sequence[Assessed] = ()
) -> list[str]:
    """What an ingest did, for people; with the rows of the table, their
    events' ids in its order."""
    lines = [
        (
            f"{plural(len(report.added), 'CDM')} added to {path} "
            f"({len(report.already_stored)} already stored), "
            f"{plural(len(report.new_events), 'new event')}."
        )
    ]
    if rows:
        ids = ", ".join(item.event.event_id for item in rows)
        lines.append(f"Event ids, row by row: {ids}.")
    return lines


def add_and_show(
    path: Path,
    files: Sequence[Path],
    settings: Settings,
    *,
    as_json: bool,
) -> None:
    """Add CDM files to a store and show their events as ``keepout assess``
    does, with the store's ids.

    Parameters
    ----------
    path
        The store; made if missing.
    files
        The CDM files.
    settings
        The settings to compute Pc with and assess.
    as_json
        Whether ``--json`` was given.

    Raises
    ------
    typer.Exit
        Without the ``store`` extra (status 1); with status 3 if an event
        needs a manoeuvre.
    """
    from keepout.store import CdmInput

    with open_cli_store(path) as store:
        messages = read_cdm_files(files, CdmInput.read)
        report = store.ingest(messages)
        assessed = store.assess(report.events, settings)
    ids = [item.event.event_id for item in assessed]
    show_assessed(
        assessed,
        cdm_count=len(messages),
        store=store_json(path, report, ids),
        store_lines=store_lines(path, report, assessed),
        as_json=as_json,
    )


def store_add(
    files: CdmFiles,
    *,
    store: StorePath,
    settings: SettingsFile = None,
    as_json: JsonFlag = False,
) -> None:
    """Add CDMs to the store and assess the events they belong to.

    Each CDM joins the stored event of its encounter (same objects, TCAs
    within 10 minutes), or starts a new one; a CDM stored before is
    reported as already stored. The Pc of every CDM of these events is
    computed with the settings, unless the store has it, and each event is
    assessed as keepout assess does, with the store's event ids. Exits with
    status 3 if at least one event's action is to manoeuvre.
    """
    add_and_show(store, files, operator_settings(settings), as_json=as_json)


def store_upgrade(*, store: StorePath, as_json: JsonFlag = False) -> None:
    """Upgrade the store to the schema of this keepout.

    A store made by an older keepout opens only once upgraded; one made by
    a newer keepout is left as it is.
    """
    with importing_store():
        from keepout.store import upgrade
    before, after = upgrade(store)
    text = (
        f"{store} is up to date (revision {after})."
        if before == after
        else f"Upgraded {store}: {before or 'base'} -> {after}."
    )
    show({"path": str(store), "from": before, "to": after}, text, as_json=as_json)


def store_rebuild(
    *, store: StorePath, settings: SettingsFile = None, as_json: JsonFlag = False
) -> None:
    """Compute the Pc of every stored CDM that the store lacks for the
    settings.

    Settings that compute Pc the same way (method, covariance policy,
    hard-body radius) share the stored Pc; the risk thresholds do not
    change them. Without hbr_m in the settings, a CDM without HBR is
    skipped, and the count of those skipped is printed.
    """
    chosen = operator_settings(settings)
    with open_cli_store(store, create=False) as opened:
        refreshed = opened.refresh(chosen)
    text = f"Computed {plural(refreshed.computed, 'Pc observation')} in {store}."
    if refreshed.skipped:
        text += (
            f"\n{plural(len(refreshed.skipped), 'CDM')} without HBR skipped: "
            "set hbr_m in the settings file."
        )
    show(
        {
            "path": str(store),
            "computed": refreshed.computed,
            "skipped": list(refreshed.skipped),
        },
        text,
        as_json=as_json,
    )


def store_purge(
    *,
    before: Before,
    store: StorePath,
    yes: Yes = False,
    as_json: JsonFlag = False,
) -> None:
    """Delete the events whose latest TCA is before a time, with their CDMs,
    their Pc and the catalog copies no other CDM needs.

    Asks first, unless --yes is given.
    """
    when = time_option(before, "--before")
    # A time years ahead is "dubious" to ERFA wherever it is formatted.
    with dubious_years_ignored():
        shown = f"TCA before {utc(when)} UTC"
        payload: dict[str, JsonOut] = {"path": str(store), "before": isot(when)}
        with open_cli_store(store, create=False) as opened:
            count = 0 if yes else opened.count_before(when)
            if count and not typer.confirm(
                f"Delete {plural(count, 'event')} with {shown}?", err=True
            ):
                show(payload | {"deleted": 0}, "Nothing deleted.", as_json=as_json)
                return
            deleted = opened.purge(when)
    show(
        payload | {"deleted": deleted},
        f"Deleted {plural(deleted, 'event')} with {shown} from {store}.",
        as_json=as_json,
    )
