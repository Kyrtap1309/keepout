"""``keepout maneuver``: the burns that clear an event, and what else they do."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, TypeAlias

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
    ASSUMED_COVARIANCES,
    ExitCode,
    JsonOut,
    duration_words,
    isot,
    minute,
    objects,
    pc_text,
    plural,
    show,
    table,
)
from keepout.cli.store import EventStore, importing_store, open_cli_store
from keepout.errors import ManeuverError, StoreError
from keepout.units import HOUR

# The library is imported where it is used: keepout --help need not load the
# numerical libraries.
if TYPE_CHECKING:
    from rich.console import RenderableType
    from rich.table import Table

    from keepout.events import ConjunctionEvent
    from keepout.maneuver import (
        BurnDirection,
        ManeuverOption,
        ManeuverOutcome,
        NewApproach,
        PlannedBurn,
        RescreenResult,
    )
    from keepout.settings import Settings
    from keepout.sources.catalog import CatalogReference, CatalogSourceName

__all__ = ["maneuver_command"]

_SHOWN_APPROACHES: Final = 5
"""Most approaches of one kind the text lists, the highest Pc first."""


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


def maneuver_command(
    cdms: Cdms = None,
    *,
    event_id: EventId = None,
    store: EventStore = None,
    target_pc: TargetPc = None,
    burn_before: BurnBefore = None,
    dv: DeltaV = None,
    hbr: HardBodyRadius = None,
    settings: OverriddenSettingsFile = None,
    catalog: CatalogFile = None,
    source: SourceOption = None,
    workers: Workers = None,
    cache_dir: CacheDir = None,
    queue: Queue = False,
    as_json: JsonFlag = False,
) -> None:
    """Find the smallest burn each way that takes an event below a target Pc.

    The CDMs must make one event, or --event names a stored one. From its
    latest CDM, along-track burns of the primary are tried from {earliest} to
    {latest} before TCA, which takes tens of seconds, and up to a minute per
    try if the event's Pc method is slow. --burn-before and --dv evaluate that
    one burn instead. --catalog or --source also screens each burn against the
    catalog for {duration} from the burn, for the approaches it brings or
    brings closer; their Pc rests on assumed covariances. A stored event that
    keepout screen found is re-screened against the catalog it was screened
    with unless one is given. Exits with status 4 if there is no burn, the
    burn given misses the target, or every burn brings a red approach or the
    event's pair back above the target. --settings gives the target, the
    Pc method, covariance policy, hard-body radius and risk thresholds;
    --target-pc and --hbr win over it. A stored event's plan against its
    stored catalog is cached in the store, on its latest CDM, and shown
    again for the same settings until a newer CDM arrives. --queue leaves
    it to keepout worker instead.
    """
    mode = _mode(
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
    match mode:
        case _Queued(store=path, event_id=stored_id):
            _queue(path, stored_id, chosen, as_json=as_json)
        case _Stored(store=path, event_id=stored_id):
            _show(_stored_plan(path, stored_id, chosen, workers), as_json=as_json)
        case _Planned(burn=burn):
            from keepout.maneuver import plan_event
            from keepout.settings import plan_arguments

            event, screened = _event_and_catalog(
                cdms, event_id, store, catalog, source, cache_dir
            )
            outcome = plan_event(
                event,
                **plan_arguments(chosen),
                burn=burn,
                catalog=None if screened is None else screened.omms,
                catalog_label=None if screened is None else screened.label,
                workers=workers,
            )
            _show(outcome, as_json=as_json)


@dataclass(frozen=True, slots=True)
class _Queued:
    """Queue the plan of a stored event for ``keepout worker``."""

    store: Path
    event_id: str


@dataclass(frozen=True, slots=True)
class _Stored:
    """Show the plan of a stored event against its stored catalog: the
    cached one, else one planned now and cached."""

    store: Path
    event_id: str


@dataclass(frozen=True, slots=True)
class _Planned:
    """Plan the event now, without the cache: on the grid, or one burn."""

    burn: tuple[float, float] | None
    """The burn given: its time before TCA [s] and its Δv [m/s]; ``None``
    for the grid."""


def _mode(
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
) -> _Queued | _Stored | _Planned:
    """What the command line asks for, its options checked.

    A stored event without a catalog or a burn given is planned through the
    cache. Any other event is planned now.

    Raises
    ------
    typer.BadParameter
        If the options do not go together: a usage error.
    """
    if queue:
        _require_queueable(event_id, catalog, source, burn_before, dv, workers)
    if (burn_before is None) != (dv is None):
        raise typer.BadParameter(
            "give --burn-before and --dv together", param_hint="--burn-before"
        )
    _require_one_event(cdms, event_id, store)
    require_one_catalog(catalog, source)
    if event_id is not None and store is not None:
        if queue:
            return _Queued(store, event_id)
        if catalog is None and source is None and burn_before is None:
            return _Stored(store, event_id)
    if burn_before is None or dv is None:
        return _Planned(None)
    return _Planned((burn_before * HOUR, dv))


def _show(outcome: ManeuverOutcome, *, as_json: bool) -> None:
    """The output of an outcome, as text or JSON.

    Raises
    ------
    typer.Exit
        With status 4 if no burn can be recommended.
    """
    verdict = outcome.verdict
    text = _summary(outcome)
    if not verdict.recommendable:
        text.append(_refusal(outcome))
    show(_payload(outcome), *text, as_json=as_json)
    if not verdict.recommendable:
        raise typer.Exit(ExitCode.NO_SAFE_BURN)


# The defaults in the help are the library's own (none under python -OO).
maneuver_command.__doc__ = (maneuver_command.__doc__ or "").format_map(
    {
        "earliest": duration_words(DECISION_POINT),
        "latest": duration_words(MIN_BURN_LEAD),
        "duration": duration_words(SCREENING_DURATION),
    }
)


def _require_one_event(
    cdms: Sequence[Path] | None, event_id: str | None, store: Path | None
) -> None:
    """A check that the command line gives CDM files or a stored event, not
    both, and a store to read the event from.

    Raises
    ------
    typer.BadParameter
        If it gives both or neither, or ``--event`` without a store: a usage
        error.
    """
    if cdms and event_id is not None:
        raise typer.BadParameter(
            "give either CDM files or --event, not both", param_hint="--event"
        )
    if not cdms and event_id is None:
        raise typer.BadParameter("give CDM files or --event", param_hint="CDMS")
    if event_id is not None and store is None:
        raise typer.BadParameter(
            "--event needs --store (or $KEEPOUT_STORE)", param_hint="--event"
        )


def _require_queueable(
    event_id: str | None,
    catalog: Path | None,
    source: CatalogSourceName | None,
    burn_before: float | None,
    dv: float | None,
    workers: int | None,
) -> None:
    """A check that ``--queue`` is given with a stored event, and without the
    options a job does not take: a job plans on the burn grid, against the
    event's stored catalog, with the processes of ``keepout worker``.

    Raises
    ------
    typer.BadParameter
        If not: a usage error.
    """
    if event_id is None:
        raise typer.BadParameter("--queue needs --event", param_hint="--queue")
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
            raise typer.BadParameter(
                f"--queue cannot take {name}: {reason}", param_hint="--queue"
            )


def _queue(store: Path, event_id: str, chosen: Settings, *, as_json: bool) -> None:
    """Submit the plan of a stored event to the store's job queue, and say
    what became of it.

    Raises
    ------
    typer.Exit
        Without the ``store`` extra (status 1).
    StoreError
        If the store cannot be opened or holds no such event.
    ValueError
        If a settings file cannot hold ``chosen`` (never those the command
        line builds), as :meth:`~keepout.store.JobQueue.submit`.
    """
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


def _stored_plan(
    store: Path, event_id: str, chosen: Settings, workers: int | None
) -> ManeuverOutcome:
    """The plan of a stored event against its stored catalog: the cached
    one, else one planned now and cached on the event's latest CDM.

    If the plan cannot be cached, a warning on standard error says so.

    Raises
    ------
    typer.Exit
        Without the ``store`` extra (status 1).
    StoreError
        If the store cannot be opened, holds no such event, or its cached
        plan cannot be read.
    SourceUnavailable
        If the stored catalog's file is gone; the message names it.
    ChecksumMismatch
        If the stored catalog's file changed; the message names it.
    KeepoutError
        As :func:`~keepout.maneuver.plan_event`.
    ValueError
        If a settings file cannot hold ``chosen`` (never those the command
        line builds), as :func:`~keepout.store.codec.maneuver_key`.
    """
    with importing_store():
        from keepout.store.planning import plan_stored

    with open_cli_store(store, create=False) as opened:
        cached = opened.maneuver(event_id, chosen)
        if cached is not None:
            return cached
        outcome = plan_stored(opened, event_id, chosen, workers=workers)
        # Planned at the event's latest CDM then: the latest of the event planned.
        planned_on = outcome.event.latest.message_id
        try:
            opened.save_maneuver(planned_on, chosen, outcome)
        except StoreError as exc:
            # Minutes of computing are shown, cached or not.
            typer.echo(f"Warning: the plan was not cached: {exc}", err=True)
    return outcome


def _stored_catalog(reference: CatalogReference | None) -> CatalogInput | None:
    """The store's copy of a stored event's catalog, read (``None``: none).

    Raises
    ------
    SourceUnavailable
        If the copy is gone; the message names it.
    ChecksumMismatch
        If the copy changed; the message names it.
    """
    if reference is None:
        return None
    from keepout.sources.catalog import load_reference

    omms = load_reference(reference)
    return CatalogInput(omms, f"stored copy of {reference.label}", reference)


def _event_and_catalog(
    cdms: Sequence[Path] | None,
    event_id: str | None,
    store: Path | None,
    catalog: Path | None,
    source: CatalogSourceName | None,
    cache_dir: Path | None,
) -> tuple[ConjunctionEvent, CatalogInput | None]:
    """The event to plan, and the catalog to re-screen its burns against
    (``None``: no re-screen).

    The event is that of the CDM files, or the stored one of ``--event``.
    The catalog is that of ``--catalog`` or ``--source``; without either,
    a stored event's is the one it was screened against, read again from
    the store's copy, and the CDM files' is none.

    Raises
    ------
    typer.Exit
        If a CDM file cannot be read (status 1).
    ManeuverError
        If the CDM files make several events.
    StoreError
        If the store cannot be opened or holds no such event.
    SourceUnavailable
        If the stored catalog's file is gone; the message names it.
    ChecksumMismatch
        If the stored catalog's file changed; the message names it.
    """
    from keepout.events import group_events

    if event_id is None or store is None:
        event = _one_event(group_events(load_cdms(cdms or ())))
        reference = None
    else:
        with open_cli_store(store, create=False) as opened:
            event = opened.event(event_id)
            reference = opened.catalog(event_id)
    if catalog is not None or source is not None:
        return event, catalog_input(catalog, source, cache_dir)
    return event, _stored_catalog(reference)


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


def _refusal(outcome: ManeuverOutcome) -> str:
    """The line saying why no burn can be recommended."""
    from keepout.maneuver.wording import refusal

    return f"{refusal(outcome)} (exit status {int(ExitCode.NO_SAFE_BURN)})."


def _option_json(option: ManeuverOption) -> dict[str, JsonOut]:
    return {
        "pc": option.pc,
        "range_at_tca_m": option.range_at_tca,
        "method": option.method,
        "meets_target": option.meets_target,
        "notes": list(option.notes),
    }


def _approach_json(item: NewApproach) -> dict[str, JsonOut]:
    approach = item.approach
    return {
        "secondary": approach.secondary_number,
        "tca": isot(approach.tca),
        "range_at_tca_m": approach.range_at_tca,
        "relative_speed_m_s": approach.relative_speed,
        "pc": item.pc,
        "level": item.level.key,
        "assumed_covariance": True,
    }


def _rescreen_json(result: RescreenResult) -> dict[str, JsonOut]:
    start, end = result.window
    return {
        "conflict": result.conflict,
        "secondary_screened": result.secondary_screened,
        "event_range_m": result.event_range,
        "catalog_objects": result.catalog_objects,
        "skipped_objects": result.skipped_objects,
        "window_start": isot(start),
        "window_end": isot(end),
        "new_approaches": [_approach_json(item) for item in result.new_approaches],
        "closer_approaches": [
            _approach_json(item) for item in result.closer_approaches
        ],
        "event_returns": [_approach_json(item) for item in result.event_returns],
    }


def _burn_json(burn: PlannedBurn) -> dict[str, JsonOut]:
    option = burn.option
    return {
        "direction": str(burn.direction),
        "burn_lead_s": option.burn_lead,
        "delta_v_m_s": option.delta_v,
        **_option_json(option),
        "rescreen": None if burn.rescreen is None else _rescreen_json(burn.rescreen),
    }


def _payload(outcome: ManeuverOutcome) -> dict[str, JsonOut]:
    """The result as JSON values, for ``--json``; its ``refusal`` is the line
    saying why no burn can be recommended (``None`` if one can)."""
    latest = outcome.event.latest
    return {
        "event": outcome.event.event_id,
        "primary": latest.object1.object_id,
        "secondary": latest.object2.object_id,
        "target_pc": outcome.target_pc,
        "method": outcome.baseline.method,
        "assumed_covariance": outcome.assumed_covariance,
        "baseline": _option_json(outcome.baseline),
        "options": [_burn_json(burn) for burn in outcome.burns],
        "notes": list(outcome.notes),
        "refusal": None if outcome.verdict.recommendable else _refusal(outcome),
    }


def _row(name: str, lead: str, delta_v: str, option: ManeuverOption) -> list[str]:
    return [
        name,
        lead,
        delta_v,
        pc_text(option.pc),
        f"{option.range_at_tca:.0f}",
        option.method,
        "met" if option.meets_target else "missed",
    ]


def _burns_table(outcome: ManeuverOutcome) -> Table:
    """A table of the baseline and the burns."""
    rows = table("Burns of the primary")
    rows.add_column("Option", no_wrap=True)
    for column in ("Before TCA (h)", "Δv (m/s)", "Pc", "Range (m)"):
        rows.add_column(column, justify="right", no_wrap=True)
    rows.add_column("Method", no_wrap=True)
    rows.add_column("Target", no_wrap=True)
    rows.add_row(*_row("no burn", "-", "-", outcome.baseline))
    for burn in outcome.burns:
        option = burn.option
        lead = f"{option.burn_lead / HOUR:.1f}"
        delta_v = f"{option.delta_v:+.3g}"
        rows.add_row(*_row(burn.direction, lead, delta_v, option))
    return rows


def _approach_lines(
    approaches: Sequence[NewApproach], noun: str, nouns: str
) -> list[str]:
    """The approaches of one kind, the highest Pc first and at most
    :data:`_SHOWN_APPROACHES`, each labelled: the Pc is on assumed covariances."""
    if not approaches:
        return []
    count = plural(len(approaches), noun, nouns)
    more = len(approaches) - _SHOWN_APPROACHES
    head = (
        f"{count}, the {_SHOWN_APPROACHES} of highest Pc:" if more > 0 else f"{count}:"
    )
    highest = sorted(approaches, key=lambda item: item.pc, reverse=True)
    lines = [
        (
            f"  {item.approach.secondary_number} at {minute(item.approach.tca)} UTC: "
            f"range {item.approach.range_at_tca:.0f} m, "
            f"assumed Pc {pc_text(item.pc)}, {item.level.key}"
        )
        for item in highest[:_SHOWN_APPROACHES]
    ]
    tail = [f"  and {more} more (--json lists them)."] if more > 0 else []
    return [head, *lines, *tail]


def _rescreen_lines(
    direction: BurnDirection, result: RescreenResult, secondary: str, label: str
) -> list[str]:
    """What the re-screen of a burn found, for people."""
    from keepout.maneuver.wording import (
        conflict_line,
        event_line,
        rescreen_head,
        return_lines,
    )

    approaches = [
        *_approach_lines(result.new_approaches, "new approach", "new approaches"),
        *_approach_lines(
            result.closer_approaches, "closer approach", "closer approaches"
        ),
    ]
    body = [
        event_line(result, secondary),
        conflict_line(result),
        *return_lines(result),
        *approaches,
    ]
    if approaches or result.event_returns:
        body.append(ASSUMED_COVARIANCES)
    return [rescreen_head(direction, result, label), *(f"  {line}" for line in body)]


def _summary(outcome: ManeuverOutcome) -> list[RenderableType]:
    """The result for people: the event, the table of burns, the notes and
    what each re-screen found."""
    event = outcome.event
    latest = event.latest
    catalog_label = outcome.catalog_label
    head = (
        f"Event {event.event_id}: {objects(latest)}, TCA {minute(latest.tca)} UTC, "
        f"target Pc {pc_text(outcome.target_pc)}."
    )
    notes = [f"Note: {note}" for note in outcome.notes]
    if catalog_label is None:
        rescreens = [
            (
                "Not re-screened: nothing is known of the approaches a burn brings "
                "(--catalog or --source screen for them)."
            )
        ]
    else:
        secondary = latest.object2.object_id
        rescreens = [
            line
            for burn in outcome.burns
            if burn.rescreen is not None
            for line in _rescreen_lines(
                burn.direction, burn.rescreen, secondary, catalog_label
            )
        ]
    from keepout.maneuver.wording import gap_lines

    return [head, _burns_table(outcome), *gap_lines(outcome), *notes, *rescreens]
