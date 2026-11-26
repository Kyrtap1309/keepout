"""``keepout maneuver``: the burns that clear an event, and what else they do."""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, NamedTuple, TypeAlias

import typer

from keepout._defaults import (
    DECISION_POINT,
    MIN_BURN_LEAD,
    SCREENING_DURATION,
    TARGET_PC,
)
from keepout._validation import require_open_probability
from keepout.cli.catalog_input import (
    CatalogFile,
    CatalogInput,
    catalog_input,
    require_one_catalog,
)
from keepout.cli.cdm_input import load_cdms
from keepout.cli.maneuver_output import show_outcome
from keepout.cli.options import (
    CacheDir,
    HardBodyRadius,
    JsonFlag,
    OverriddenSettingsFile,
    SourceOption,
    Workers,
    operator_settings,
    positive,
)
from keepout.cli.output import (
    JsonOut,
    duration_words,
    help_values,
    minute,
    objects,
    progress,
    show,
)
from keepout.cli.store import EventStore, importing_store, open_cli_store
from keepout.errors import ManeuverError
from keepout.units import HOUR

# The library is imported where it is used: keepout --help need not load the
# numerical libraries.
if TYPE_CHECKING:
    from keepout.events import ConjunctionEvent
    from keepout.maneuver import ManeuverOutcome
    from keepout.settings import Settings
    from keepout.sources.catalog import CatalogSourceName

__all__ = ["HELP_PANELS", "maneuver_command"]


def _probability(value: float | None) -> float | None:
    """``--target-pc``, checked to be in (0, 1) (``None`` passes).

    Raises
    ------
    typer.BadParameter
        If it is not: a usage error.
    """
    if value is not None:
        try:
            require_open_probability(value, "value")
        except ValueError as exc:
            raise typer.BadParameter(f"must be in (0, 1), got {value}") from exc
    return value


def _burn_size(value: float | None) -> float | None:
    """``--dv``, checked to be finite and a burn (``None`` passes).

    Raises
    ------
    typer.BadParameter
        If it is not: a usage error.
    """
    if value is not None and (value == 0.0 or not math.isfinite(value)):
        raise typer.BadParameter(f"must be a finite, non-zero number, got {value}")
    return value


Cdms: TypeAlias = Annotated[
    list[Path] | None,
    typer.Argument(
        exists=True,
        dir_okay=False,
        metavar="CDM...",
        show_default=False,
        help=(
            "The CDM files of one event, KVN or XML, gzipped or not (or --event "
            "instead)."
        ),
    ),
]
"""The CDM files of the event, unless ``--event`` gives it."""

EventId: TypeAlias = Annotated[
    str | None,
    typer.Option(
        "--event",
        metavar="ID",
        help=(
            "Plan the stored event of this id, from the store of --store, "
            "instead of CDM files; re-screened against the catalog it was "
            "screened with, if any, unless --catalog or --source is given."
        ),
    ),
]
"""``--event``: a stored event to plan instead of CDM files."""

TargetPc: TypeAlias = Annotated[
    float | None,
    typer.Option(
        "--target-pc",
        callback=_probability,
        metavar="PC",
        help=(
            "The Pc a burn must stay below (default: the settings' target_pc, "
            f"else {TARGET_PC:g})."
        ),
    ),
]
"""``--target-pc``: the target of the plan; ``None``: the settings'."""

BurnBefore: TypeAlias = Annotated[
    float | None,
    typer.Option(
        "--burn-before",
        callback=positive,
        metavar="HOURS",
        help="Evaluate one burn, this many hours before TCA (needs --dv).",
    ),
]
"""``--burn-before``: when the burn given is."""

DeltaV: TypeAlias = Annotated[
    float | None,
    typer.Option(
        "--dv",
        callback=_burn_size,
        metavar="M/S",
        help=(
            "Evaluate one burn of this Δv along the velocity, in m/s, negative "
            "for retrograde (needs --burn-before)."
        ),
    ),
]
"""``--dv``: how large the burn given is."""

Queue: TypeAlias = Annotated[
    bool,
    typer.Option(
        "--queue",
        help=(
            "Queue the plan of the stored event of --event for keepout worker "
            "instead of planning it now; keepout maneuver --event shows it "
            "once computed."
        ),
    ),
]
"""``--queue``: submit the plan of ``--event`` to the store's job queue."""


@help_values(
    earliest=duration_words(DECISION_POINT),
    latest=duration_words(MIN_BURN_LEAD),
    duration=duration_words(SCREENING_DURATION),
)
def maneuver_command(
    ctx: typer.Context,
    cdms: Cdms = None,
    *,
    event_id: EventId = None,
    store: EventStore = None,
    queue: Queue = False,
    target_pc: TargetPc = None,
    burn_before: BurnBefore = None,
    dv: DeltaV = None,
    catalog: CatalogFile = None,
    source: SourceOption = None,
    workers: Workers = None,
    cache_dir: CacheDir = None,
    hbr: HardBodyRadius = None,
    settings: OverriddenSettingsFile = None,
    as_json: JsonFlag = False,
) -> None:
    """Find the smallest burn each way that takes an event below a target Pc.

    The event is that of CDM files, or a stored one (--event); burns are
    planned from its latest CDM. Along-track burns of the primary are tried
    from {earliest} to {latest} before TCA: tens of seconds, up to a minute
    per try for a slow Pc method. --burn-before and --dv evaluate one burn
    instead.

    --catalog or --source also screens each burn against the catalog for
    {duration} from the burn, for the approaches it brings or brings closer;
    their Pc rests on assumed covariances. A stored event that keepout
    screen found is re-screened against the catalog it was screened with,
    unless one is given.

    A stored event's plan against its stored catalog is cached in the store,
    on its latest CDM, for the settings; --queue leaves it to keepout worker.

    --settings gives the target, Pc method, covariance policy, hard-body
    radius and risk thresholds; --target-pc and --hbr win over it. Exits
    with status 4 if no burn can be recommended.
    """
    _check_options(
        ctx,
        cdms,
        event_id,
        store,
        catalog=catalog,
        source=source,
        burn_before=burn_before,
        dv=dv,
        workers=workers,
        queue=queue,
    )
    chosen = operator_settings(
        settings, combined_hard_body_radius=hbr, target_pc=target_pc
    )
    burn = None if burn_before is None or dv is None else (burn_before * HOUR, dv)
    screened = _Catalog(catalog, source, cache_dir)
    if event_id is None or store is None:
        outcome = _plan_files(cdms or (), chosen, burn, screened, workers)
    elif queue:
        _queue(store, event_id, chosen, as_json=as_json)
        return
    elif burn is None and not screened.given:
        outcome = _cached_or_planned(store, event_id, chosen, workers)
    else:
        outcome = _plan_stored(store, event_id, chosen, burn, screened, workers)
    show_outcome(outcome, as_json=as_json)


HELP_PANELS: Final = {
    "event_id": "Event",
    "store": "Event",
    "queue": "Event",
    "target_pc": "Burns",
    "burn_before": "Burns",
    "dv": "Burns",
    "catalog": "Re-screen",
    "source": "Re-screen",
    "workers": "Re-screen",
    "cache_dir": "Re-screen",
    "hbr": "Settings",
    "settings": "Settings",
}
"""The panel of ``keepout maneuver --help`` each option is listed in (the
others are under Typer's Options)."""


class _Catalog(NamedTuple):
    """The catalog the command line gives to re-screen against, read when
    asked for."""

    path: Path | None
    source: CatalogSourceName | None
    cache_dir: Path | None

    @property
    def given(self) -> bool:
        return self.path is not None or self.source is not None

    def read(self) -> CatalogInput | None:
        """Its element sets (``None``: none given)."""
        if not self.given:
            return None
        return catalog_input(self.path, self.source, self.cache_dir)


def _check_options(
    ctx: typer.Context,
    cdms: Sequence[Path] | None,
    event_id: str | None,
    store: Path | None,
    *,
    catalog: Path | None,
    source: CatalogSourceName | None,
    burn_before: float | None,
    dv: float | None,
    workers: int | None,
    queue: bool,
) -> None:
    """A check that the options go together; a usage error if not."""
    if queue:
        _check_queueable(
            ctx,
            event_id,
            catalog=catalog,
            source=source,
            burn_before=burn_before,
            dv=dv,
            workers=workers,
        )
    if (burn_before is None) != (dv is None):
        ctx.fail("give --burn-before and --dv together")
    if cdms and event_id is not None:
        ctx.fail("give either CDM files or --event, not both")
    if not cdms and event_id is None:
        ctx.fail("give CDM files or --event")
    if event_id is not None and store is None:
        ctx.fail("--event needs --store (or $KEEPOUT_STORE)")
    require_one_catalog(ctx, catalog, source)


def _check_queueable(
    ctx: typer.Context,
    event_id: str | None,
    *,
    catalog: Path | None,
    source: CatalogSourceName | None,
    burn_before: float | None,
    dv: float | None,
    workers: int | None,
) -> None:
    """A check that ``--queue`` has a stored event and none of the options a
    job does not take; a usage error if not."""
    if event_id is None:
        ctx.fail("--queue needs --event")
    planned = "a job plans on the burn grid, against the event's stored catalog"
    given = (
        ("--catalog", catalog, planned),
        ("--source", source, planned),
        ("--burn-before", burn_before, planned),
        ("--dv", dv, planned),
        ("--workers", workers, "keepout worker --workers sets a job's processes"),
    )
    for name, value, reason in given:
        if value is not None:
            ctx.fail(f"--queue cannot take {name}: {reason}")


def _queue(store: Path, event_id: str, chosen: Settings, *, as_json: bool) -> None:
    """Submit the plan of a stored event to the store's job queue, and say
    what became of it."""
    with importing_store():
        from keepout.store import Job, JobQueue, SubmissionOutcome

    with open_cli_store(store, create=False) as opened:
        submission = JobQueue(opened).submit(event_id, chosen)
    job = submission.job
    match submission.outcome, job:
        case SubmissionOutcome.EXISTING, Job():
            text = f"Job {job.id} is already {job.status.value} for event {event_id}."
        case SubmissionOutcome.QUEUED, Job():
            text = f"Job {job.id} queued for event {event_id}."
        case _:
            text = (
                f"Event {event_id} is already planned for these settings: "
                f"keepout maneuver --event {event_id} shows it."
            )
    payload: dict[str, JsonOut] = {
        "event": event_id,
        "job": None if job is None else job.id,
        "status": None if job is None else job.status.value,
        "cached": submission.outcome is SubmissionOutcome.CACHED,
    }
    show(payload, text, as_json=as_json)


def _cached_or_planned(
    store: Path, event_id: str, chosen: Settings, workers: int | None
) -> ManeuverOutcome:
    """The plan of a stored event the store caches; a warning on standard
    error if it could not be cached."""
    with importing_store():
        from keepout.store.planning import cached_or_planned

    with open_cli_store(store, create=False) as opened, _planning():
        planned = cached_or_planned(opened, event_id, chosen, workers=workers)
    if planned.unsaved is not None:
        # Minutes of computing are shown, cached or not.
        typer.echo(f"Warning: the plan was not cached: {planned.unsaved}", err=True)
    return planned.outcome


def _plan_stored(
    store: Path,
    event_id: str,
    chosen: Settings,
    burn: tuple[float, float] | None,
    screened: _Catalog,
    workers: int | None,
) -> ManeuverOutcome:
    """The plan of a stored event, not cached: with a burn given, or
    against a catalog given."""
    with importing_store():
        from keepout.store.planning import plan_stored

    with _planning():
        catalog = screened.read()
        with open_cli_store(store, create=False) as opened:
            return plan_stored(
                opened,
                event_id,
                chosen,
                workers=workers,
                burn=burn,
                catalog=None if catalog is None else catalog.omms,
                catalog_label=None if catalog is None else catalog.label,
            )


def _plan_files(
    files: Sequence[Path],
    chosen: Settings,
    burn: tuple[float, float] | None,
    screened: _Catalog,
    workers: int | None,
) -> ManeuverOutcome:
    """The plan of the event of CDM files (one event, or an error)."""
    from keepout.events import group_events
    from keepout.maneuver import plan_event
    from keepout.settings import plan_arguments

    event = _one_event(group_events(load_cdms(files)))
    with _planning():
        catalog = screened.read()
        return plan_event(
            event,
            **plan_arguments(chosen),
            burn=burn,
            catalog=None if catalog is None else catalog.omms,
            catalog_label=None if catalog is None else catalog.label,
            workers=workers,
        )


def _planning() -> AbstractContextManager[None]:
    return progress("Planning burns (tens of seconds; minutes on a slow Pc)")


def _one_event(events: Sequence[ConjunctionEvent]) -> ConjunctionEvent:
    """The only one of ``events``.

    Raises
    ------
    ManeuverError
        If there are several, which it lists.
    """
    if len(events) > 1:
        listed = ", ".join(
            f"{objects(event.latest)} at {minute(event.latest.tca)}" for event in events
        )
        raise ManeuverError(
            f"these CDMs hold {len(events)} events, and a burn is planned for "
            f"one: {listed}"
        )
    return events[0]
