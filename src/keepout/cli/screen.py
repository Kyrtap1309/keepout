"""``keepout screen``: a synthetic operator's week, from screening to risk."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NamedTuple, TypeAlias

import typer

from keepout._defaults import SCREENING_DURATION, SOCRATES_RADIUS
from keepout.cli.catalog_input import CatalogFile, catalog_input, require_one_catalog
from keepout.cli.events import event_json, events_table, reasons
from keepout.cli.options import (
    CacheDir,
    JsonFlag,
    SettingsFile,
    SourceOption,
    Workers,
    operator_settings,
    parse_time,
    positive,
    time_option,
)
from keepout.cli.output import ASSUMED_COVARIANCES, JsonOut, isot, plural, show, utc
from keepout.cli.store import (
    StoreOption,
    ingest_screened,
    open_cli_store,
    store_json,
    store_lines,
)
from keepout.units import DAY, KM

# The library is imported in the command: keepout --help need not load the
# numerical libraries.
if TYPE_CHECKING:
    from astropy.time import Time
    from rich.console import RenderableType

    from keepout.events import ConjunctionEvent
    from keepout.screening import SkippedObject
    from keepout.store import IngestReport
    from keepout.synthetic import Fleet, OperatorEvent, OperatorResult

__all__ = ["parse_time", "screen_command"]


FleetFile: TypeAlias = Annotated[
    Path,
    typer.Option(
        "--fleet",
        exists=True,
        dir_okay=False,
        help="The fleet: a TOML file of the satellites to protect.",
    ),
]
"""``--fleet``: the fleet file (:func:`keepout.synthetic.load_fleet`)."""

Days: TypeAlias = Annotated[
    float,
    typer.Option("--days", callback=positive, help="Length of the window in days."),
]
"""``--days``: the length of the screening window."""

Start: TypeAlias = Annotated[
    str | None,
    typer.Option(
        "--start",
        metavar="TIME",
        help="Start of the window, ISO 8601 UTC (default: now).",
    ),
]
"""``--start``: the start of the window and the CDMs' creation date, read by
:func:`parse_time` in the command."""

CdmDir: TypeAlias = Annotated[
    Path | None,
    typer.Option(
        "--cdm-dir",
        file_okay=False,
        help="Write each approach's CDM (KVN) to this directory.",
    ),
]
"""``--cdm-dir``: where to write the CDMs, ``<MESSAGE_ID>.kvn``."""


def screen_command(
    *,
    fleet: FleetFile,
    days: Days = SCREENING_DURATION / DAY,
    start: Start = None,
    catalog: CatalogFile = None,
    source: SourceOption = None,
    cdm_dir: CdmDir = None,
    store: StoreOption = None,
    settings: SettingsFile = None,
    workers: Workers = None,
    cache_dir: CacheDir = None,
    as_json: JsonFlag = False,
) -> None:
    """Screen a fleet against the catalog, issue a CDM per approach, assess each.

    A synthetic operator on public data: each close approach ({radius:g} km sphere)
    of a fleet satellite in the window becomes a CDM whose covariances come
    from keepout's model of GP element set errors, an assumption, not orbit
    determination. Each CDM's Pc is computed and its risk assessed as of the
    start of the window, which is also the CDMs' creation date. The catalog
    is the cached copy of keepout fetch catalog while it is current, else a
    new download; --catalog reads a file instead. With --store the CDMs are
    added to the store, with a copy of the catalog. --settings gives the Pc
    method, covariance policy, hard-body radius and risk thresholds.
    """
    window_start = None if start is None else time_option(start, "--start")
    require_one_catalog(catalog, source)
    chosen = operator_settings(settings)
    with ExitStack() as stack:
        opened = None if store is None else stack.enter_context(open_cli_store(store))
        from keepout.synthetic import load_fleet, run

        satellites = load_fleet(fleet)
        screened = catalog_input(catalog, source, cache_dir)
        if window_start is None:
            window_start = _now()
        result = run(
            satellites,
            screened.omms,
            window_start,
            days * DAY,
            workers=workers,
            compute=chosen.compute,
            policy=chosen.policy,
            combined_hard_body_radius=chosen.combined_hard_body_radius,
            thresholds=chosen.thresholds,
        )
        files = {} if cdm_dir is None else _write_cdms(result, cdm_dir)
        stored = (
            None
            if store is None or opened is None
            else _Stored(
                store,
                *ingest_screened(opened, result.cdms, screened.reference, chosen),
            )
        )
    events = list(result.events) if stored is None else stored.rows(result.events)
    payload = _payload(result, events, screened.label, files, cdm_dir, stored)
    summary = _summary(result, events, screened.label, files, cdm_dir, stored)
    show(payload, *summary, as_json=as_json)


# The volume in the help is the library's default (none under python -OO).
screen_command.__doc__ = (screen_command.__doc__ or "").format_map(
    {"radius": SOCRATES_RADIUS / KM}
)


class _Stored(NamedTuple):
    """What ``--store`` did: the store, the ingest and the stored events it
    touched, in the store's order."""

    path: Path
    report: IngestReport
    events: Sequence[ConjunctionEvent]

    def rows(self, approaches: Sequence[OperatorEvent]) -> list[OperatorEvent]:
        """The approaches, each under the id of its stored event, in the
        store's order of the events (an event's approaches in screening
        order)."""
        event_of = {
            cdm.message_id: event.event_id
            for event in self.events
            for cdm in event.cdms
        }
        order = {event.event_id: index for index, event in enumerate(self.events)}
        renamed = [
            dataclasses.replace(
                approach,
                event=dataclasses.replace(
                    approach.event, event_id=event_of[approach.cdm.message_id]
                ),
            )
            for approach in approaches
        ]
        return sorted(renamed, key=lambda approach: order[approach.event.event_id])

    def json(self) -> dict[str, JsonOut]:
        """What the store did, as JSON values."""
        ids = [event.event_id for event in self.events]
        return store_json(self.path, self.report, ids)


def _now() -> Time:
    """The current time in UTC, to the second: the default start."""
    from astropy.time import Time

    return Time(int(time.time()), format="unix", scale="utc")


def _write_cdms(result: OperatorResult, cdm_dir: Path) -> dict[str, Path]:
    """Write each CDM to ``<MESSAGE_ID>.kvn`` in ``cdm_dir`` and return the
    files by message ID."""
    from keepout.cdm import kvn

    cdm_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    for cdm in result.cdms:
        path = cdm_dir / f"{cdm.message_id}.kvn"
        kvn.dump(cdm, path)
        files[cdm.message_id] = path
    return files


def _payload(
    result: OperatorResult,
    events: Sequence[OperatorEvent],
    catalog_label: str,
    files: Mapping[str, Path],
    cdm_dir: Path | None,
    stored: _Stored | None,
) -> dict[str, JsonOut]:
    """The result as JSON values, for ``--json``."""
    screening = result.screening
    return {
        "fleet": list(result.fleet.catalog_numbers),
        "catalog": catalog_label,
        "catalog_objects": screening.catalog_count,
        "start": isot(screening.start),
        "end": isot(screening.end),
        "events": [_event_json(e, files) for e in events],
        "assumed_covariance_events": sum(
            e.event.assumed_covariance for e in result.events
        ),
        "skipped": [_skipped_json(s) for s in screening.skipped],
        "cdm_dir": None if cdm_dir is None else str(cdm_dir),
        "store": None if stored is None else stored.json(),
    }


def _summary(
    result: OperatorResult,
    events: Sequence[OperatorEvent],
    catalog_label: str,
    files: Mapping[str, Path],
    cdm_dir: Path | None,
    stored: _Stored | None,
) -> list[RenderableType]:
    """The result for people: what was screened, the fleet satellites left
    out, the table of approaches and the notes after it."""
    from keepout.risk import Action
    from keepout.screening import ScreeningRole

    screening = result.screening
    fleet = result.fleet
    unscreened = [s for s in screening.skipped if s.role is ScreeningRole.PRIMARY]
    count = len(fleet.satellites)
    screened = (
        plural(count, "fleet satellite")
        if not unscreened
        else f"{count - len(unscreened)} of {plural(count, 'fleet satellite')}"
    )
    head = [
        (
            f"Screened {screened} "
            f"against {plural(screening.catalog_count, 'object')} ({catalog_label}) "
            f"from {utc(screening.start)} to {utc(screening.end)} UTC: "
            f"{plural(len(result.events), 'close approach', 'close approaches')}."
        ),
        *(_not_screened(skipped, fleet) for skipped in unscreened),
    ]
    tail: list[str] = []
    if screening.skipped:
        tail.append(
            f"{plural(len(screening.skipped), 'element set')} left out (stale, "
            "future epochs, SGP4 failures; --json lists them)."
        )
    tail += reasons(events, Action.MONITOR)
    tail.append(ASSUMED_COVARIANCES)
    if cdm_dir is not None:
        tail.append(f"{plural(len(files), 'CDM')} written to {cdm_dir}")
    if stored is not None:
        tail += store_lines(stored.path, stored.report, events)
    table = [events_table(events, "Close approaches")] if events else []
    return [*head, *table, *tail]


def _not_screened(skipped: SkippedObject, fleet: Fleet) -> str:
    """The line naming a fleet satellite the screening left out: it is not
    protected, so it must not hide among the catalog's element sets."""
    satellite = fleet.find(skipped.catalog_number)
    name = (satellite.name if satellite else None) or skipped.elements.object_name
    named = f" ({name})" if name else ""
    return (
        f"Fleet satellite {skipped.catalog_number}{named} was NOT screened: "
        f"{skipped.reason}, {skipped.detail}."
    )


def _event_json(event: OperatorEvent, files: Mapping[str, Path]) -> dict[str, JsonOut]:
    path = files.get(event.cdm.message_id)
    return event_json(event) | {
        "relative_speed_m_s": event.approach.relative_speed,
        "cdm_file": None if path is None else str(path),
    }


def _skipped_json(skipped: SkippedObject) -> dict[str, JsonOut]:
    return {
        "catalog_number": skipped.catalog_number,
        "role": str(skipped.role),
        "reason": str(skipped.reason),
        "detail": skipped.detail,
    }
