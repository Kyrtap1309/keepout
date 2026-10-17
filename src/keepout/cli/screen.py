"""``keepout screen``: a synthetic operator's week, from screening to risk."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TypeAlias

import typer

from keepout._units import DAY
from keepout.cli.events import (
    AssessedEvent,
    event_json,
    events_table,
    reasons,
)
from keepout.cli.fetch import SOURCE_NAMES, load_catalog
from keepout.cli.options import (
    CacheDir,
    CatalogSource,
    JsonFlag,
    Workers,
    positive,
)
from keepout.cli.output import JsonValue, isot, plural, show, utc

# The library is imported in the command: keepout --help need not load the
# numerical libraries.
if TYPE_CHECKING:
    from astropy.time import Time

    from keepout.screening import SkippedObject
    from keepout.synthetic import Fleet, OperatorEvent

__all__ = ["parse_time", "screen_command"]


def parse_time(text: str) -> Time:
    """A time given on the command line: ISO 8601 in UTC, e.g.
    ``2026-10-06`` or ``2026-10-06T12:00:00``.

    Raises
    ------
    ValueError
        If ``text`` is not such a time (Typer makes it a usage error).
    """
    from astropy.time import Time

    try:
        return Time(text, scale="utc")
    except ValueError as exc:
        raise ValueError(f"not an ISO 8601 UTC time: {text!r}") from exc


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


def _checked_time(text: str | None) -> str | None:
    """``--start`` as given, once :func:`parse_time` reads it.

    Raises
    ------
    typer.BadParameter
        If it does not: a usage error.
    """
    if text is not None:
        try:
            parse_time(text)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    return text


Start: TypeAlias = Annotated[
    str | None,
    typer.Option(
        "--start",
        callback=_checked_time,
        metavar="TIME",
        help="Start of the window, ISO 8601 UTC (default: now).",
    ),
]
"""``--start``: the start of the window and the CDMs' creation date."""

CatalogFile: TypeAlias = Annotated[
    Path | None,
    typer.Option(
        "--catalog",
        exists=True,
        dir_okay=False,
        help=(
            "Screen against the element sets in this file (OMM KVN or XML, "
            "CelesTrak CSV or JSON, TLE; gzipped or not) instead of a source."
        ),
    ),
]
"""``--catalog``: a local catalog file instead of a source."""

SourceOption: TypeAlias = Annotated[
    CatalogSource | None,
    typer.Option(
        "--source",
        help=(
            "Catalog to screen against, as keepout fetch catalog gets it "
            "(default: celestrak)."
        ),
    ),
]
"""``--source``: the catalog source, unless ``--catalog`` is given."""

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
    days: Days = 7.0,
    start: Start = None,
    catalog: CatalogFile = None,
    source: SourceOption = None,
    cdm_dir: CdmDir = None,
    workers: Workers = None,
    cache_dir: CacheDir = None,
    as_json: JsonFlag = False,
) -> None:
    """Screen a fleet against the catalog, issue a CDM per approach, assess each.

    A synthetic operator on public data: each close approach (5 km sphere)
    of a fleet satellite in the window becomes a CDM whose covariances come
    from keepout's model of GP element set errors, an assumption, not orbit
    determination. Each CDM's Pc is computed and its risk assessed as of the
    start of the window, which is also the CDMs' creation date. The catalog
    is the cached copy of keepout fetch catalog while it is current, else a
    new download; --catalog reads a file instead.
    """
    if catalog is not None and source is not None:
        raise typer.BadParameter(
            "give either --catalog or --source, not both", param_hint="--source"
        )
    from astropy.time import Time

    from keepout.cdm import kvn
    from keepout.omm import load
    from keepout.risk import Action
    from keepout.screening import ScreeningRole
    from keepout.synthetic import load_fleet
    from keepout.synthetic.operator import run

    satellites = load_fleet(fleet)
    if catalog is not None:
        omms = load(catalog)
        described = str(catalog)
    else:
        chosen = source or CatalogSource.CELESTRAK
        omms = load_catalog(chosen, cache_dir).omms
        described = SOURCE_NAMES[chosen]
    began = (
        parse_time(start)
        if start is not None
        else Time(int(time.time()), format="unix", scale="utc")
    )
    result = run(
        satellites,
        omms,
        began,
        days * DAY,
        workers=workers,
    )
    files: dict[str, Path] = {}
    if cdm_dir is not None:
        cdm_dir.mkdir(parents=True, exist_ok=True)
        for cdm in result.cdms:
            files[cdm.message_id] = path = cdm_dir / f"{cdm.message_id}.kvn"
            kvn.dump(cdm, path)
    screening = result.screening
    assessed = [AssessedEvent(e.event, e.assessment) for e in result.events]
    payload: dict[str, JsonValue] = {
        "fleet": list(satellites.catalog_numbers),
        "catalog": described,
        "catalog_objects": screening.catalog_count,
        "start": isot(screening.start),
        "end": isot(screening.end),
        "events": [_event_json(e, files) for e in result.events],
        "assumed_covariances": sum(e.event.assumed_covariance for e in result.events),
        "skipped": [_skipped_json(s) for s in screening.skipped],
        "cdm_dir": None if cdm_dir is None else str(cdm_dir),
    }
    unscreened = [s for s in screening.skipped if s.role is ScreeningRole.PRIMARY]
    count = len(satellites.satellites)
    screened = (
        plural(count, "fleet satellite")
        if not unscreened
        else f"{count - len(unscreened)} of {plural(count, 'fleet satellite')}"
    )
    head = [
        (
            f"Screened {screened} "
            f"against {plural(screening.catalog_count, 'object')} ({described}) "
            f"from {utc(screening.start)} to {utc(screening.end)} UTC: "
            f"{plural(len(result.events), 'close approach', 'close approaches')}."
        ),
        *(_not_screened(skipped, satellites) for skipped in unscreened),
    ]
    tail: list[str] = []
    if screening.skipped:
        tail.append(
            f"{plural(len(screening.skipped), 'element set')} left out (stale, "
            "future epochs, SGP4 failures; --json lists them)."
        )
    tail += reasons(assessed, Action.MONITOR)
    tail.append(
        "Covariances are assumed (keepout's model of GP element set errors), "
        "not orbit determination."
    )
    if cdm_dir is not None:
        tail.append(f"{plural(len(files), 'CDM')} written to {cdm_dir}")
    table = [events_table(assessed, "Close approaches")] if assessed else []
    show(payload, *head, *table, *tail, as_json=as_json)


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


def _event_json(
    event: OperatorEvent, files: Mapping[str, Path]
) -> dict[str, JsonValue]:
    path = files.get(event.cdm.message_id)
    return event_json(AssessedEvent(event.event, event.assessment)) | {
        "relative_speed": event.approach.relative_speed,
        "cdm_file": None if path is None else str(path),
    }


def _skipped_json(skipped: SkippedObject) -> dict[str, JsonValue]:
    return {
        "catalog_number": skipped.catalog_number,
        "role": str(skipped.role),
        "reason": str(skipped.reason),
        "detail": skipped.detail,
    }
