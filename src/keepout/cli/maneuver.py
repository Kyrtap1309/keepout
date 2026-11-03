"""``keepout maneuver``: the burns that clear an event, and what else they do."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, NamedTuple, TypeAlias

import typer

from keepout._defaults import (
    DECISION_POINT,
    MIN_BURN_LEAD,
    SCREENING_DURATION,
    TARGET_PC,
    TCA_TOLERANCE,
)
from keepout.cli.assess import HardBodyRadius, load_cdms
from keepout.cli.catalog_input import CatalogFile, catalog_input, require_one_catalog
from keepout.cli.options import CacheDir, JsonFlag, SourceOption, Workers, positive
from keepout.cli.output import (
    ExitCode,
    JsonOut,
    duration_words,
    isot,
    minute,
    pc_text,
    plural,
    show,
    table,
    utc,
)
from keepout.errors import ManeuverError
from keepout.units import HOUR, MINUTE

# The library is imported where it is used: keepout --help need not load the
# numerical libraries.
if TYPE_CHECKING:
    from rich.console import RenderableType
    from rich.table import Table

    from keepout.events import ConjunctionEvent
    from keepout.maneuver import (
        ManeuverOption,
        ManeuverPlan,
        NewApproach,
        RescreenResult,
    )

__all__ = ["maneuver_command"]

_SHOWN_APPROACHES: Final = 5
"""Most approaches of one kind the text lists, the highest Pc first."""


def _probability(value: float) -> float:
    """``--target-pc``, checked to be in (0, 1).

    Raises
    ------
    typer.BadParameter
        If it is not: a usage error.
    """
    if not 0.0 < value < 1.0:
        raise typer.BadParameter(f"must be in (0, 1), got {value}")
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
    list[Path],
    typer.Argument(
        exists=True,
        dir_okay=False,
        metavar="CDMS",
        help="The CDM files of one event, KVN or XML, gzipped or not.",
    ),
]
"""The CDM files of the event."""

TargetPc: TypeAlias = Annotated[
    float,
    typer.Option(
        "--target-pc",
        callback=_probability,
        help="The Pc a burn must stay below.",
    ),
]
"""``--target-pc``: the target of the plan."""

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


class _Burn(NamedTuple):
    """A burn to show, and its re-screen if there was a catalog."""

    option: ManeuverOption
    rescreen: RescreenResult | None

    @property
    def direction(self) -> str:
        """The burn's direction along the velocity."""
        return "prograde" if self.option.delta_v > 0.0 else "retrograde"


def maneuver_command(
    cdms: Cdms,
    *,
    target_pc: TargetPc = TARGET_PC,
    burn_before: BurnBefore = None,
    dv: DeltaV = None,
    hbr: HardBodyRadius = None,
    catalog: CatalogFile = None,
    source: SourceOption = None,
    workers: Workers = None,
    cache_dir: CacheDir = None,
    as_json: JsonFlag = False,
) -> None:
    """Find the smallest burn each way that takes an event below a target Pc.

    The CDMs must make one event. From its latest CDM, along-track burns of the
    primary are tried from {earliest} to {latest} before TCA, which takes tens
    of seconds, and up to a minute per try if the event's Pc method is slow.
    --burn-before and --dv evaluate that one burn instead. --catalog or --source
    also screens each burn against the catalog for {duration} from the burn,
    for the approaches it brings or brings closer; their Pc rests on assumed
    covariances. Exits with status 4 if there is no burn, the burn given misses
    the target, or every burn brings a red approach or the event's pair back
    above the target.
    """
    if (burn_before is None) != (dv is None):
        raise typer.BadParameter(
            "give --burn-before and --dv together", param_hint="--burn-before"
        )
    require_one_catalog(catalog, source)
    from keepout.events import group_events
    from keepout.maneuver import evaluate_burn, plan_maneuver, rescreen

    event = _one_event(group_events(load_cdms(cdms)))
    screened = (
        None
        if catalog is None and source is None
        else catalog_input(catalog, source, cache_dir)
    )
    plan = plan_maneuver(event, target_pc=target_pc, combined_hard_body_radius=hbr)
    if burn_before is None or dv is None:
        found = (plan.prograde, plan.retrograde)
        options = [item for item in found if item is not None]
    else:
        burn_lead = burn_before * HOUR
        options = [evaluate_burn(plan, burn_lead=burn_lead, delta_v=dv)]
    burns = [
        _Burn(
            option,
            None
            if screened is None
            else rescreen(plan, option, screened[0], workers=workers),
        )
        for option in options
    ]
    refusal = _refusal(plan, burns)
    gaps = [] if burn_before is not None else _gaps(burns)
    show(
        _payload(plan, burns),
        *_summary(plan, burns, gaps, None if screened is None else screened[1]),
        *([] if refusal is None else [refusal]),
        as_json=as_json,
    )
    if refusal is not None:
        raise typer.Exit(ExitCode.NO_SAFE_BURN)


# The defaults in the help are the library's own (none under python -OO).
maneuver_command.__doc__ = (maneuver_command.__doc__ or "").format_map(
    {
        "earliest": duration_words(DECISION_POINT),
        "latest": duration_words(MIN_BURN_LEAD),
        "duration": duration_words(SCREENING_DURATION),
    }
)


def _one_event(events: Sequence[ConjunctionEvent]) -> ConjunctionEvent:
    """The only one of ``events``.

    Raises
    ------
    ManeuverError
        If there are several, which it lists.
    """
    if len(events) > 1:
        listed = ", ".join(
            f"{_objects(event)} at {minute(event.latest.tca)}" for event in events
        )
        raise ManeuverError(
            f"these CDMs hold {len(events)} events, and a burn is planned for "
            f"one: {listed}"
        )
    return events[0]


def _objects(event: ConjunctionEvent) -> str:
    latest = event.latest
    return f"{latest.object1.object_id} / {latest.object2.object_id}"


def _refusal(plan: ManeuverPlan, burns: Sequence[_Burn]) -> str | None:
    """The line saying why no burn can be recommended; ``None`` if one can."""
    target = f"the target Pc {pc_text(plan.target_pc)}"
    results = [burn.rescreen for burn in burns]
    if not burns:
        reason = f"No burn on the grid meets {target}"
    elif not all(burn.option.meets_target for burn in burns):
        reason = f"This burn misses {target}"
    elif any(result is None or result.clean for result in results):
        return None
    else:
        reds = any(result.conflict for result in results if result is not None)
        back = any(result.event_returns for result in results if result is not None)
        what = " or ".join(
            [
                *(["an approach at red"] if reds else []),
                *([f"this pair back above {target}"] if back else []),
            ]
        )
        reason = f"{'Every' if len(burns) > 1 else 'This'} burn brings {what}"
    return f"{reason} (exit status {int(ExitCode.NO_SAFE_BURN)})."


def _gaps(burns: Sequence[_Burn]) -> list[str]:
    """The line of the direction without an option, if the other has one."""
    found = {burn.direction for burn in burns}
    return [
        f"No {direction} burn on the grid meets the target."
        for direction in ("prograde", "retrograde")
        if found and direction not in found
    ]


def _option_json(option: ManeuverOption) -> dict[str, JsonOut]:
    return {
        "pc": option.pc,
        "range_at_tca_m": option.range_at_tca,
        "method": option.method,
        "meets_target": option.meets_target,
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


def _burn_json(burn: _Burn) -> dict[str, JsonOut]:
    option = burn.option
    return {
        "direction": burn.direction,
        "burn_lead_s": option.burn_lead,
        "delta_v_m_s": option.delta_v,
        **_option_json(option),
        "rescreen": None if burn.rescreen is None else _rescreen_json(burn.rescreen),
    }


def _payload(plan: ManeuverPlan, burns: Sequence[_Burn]) -> dict[str, JsonOut]:
    """The result as JSON values, for ``--json``."""
    latest = plan.event.latest
    return {
        "event": plan.event.event_id,
        "primary": latest.object1.object_id,
        "secondary": latest.object2.object_id,
        "target_pc": plan.target_pc,
        "method": plan.baseline.method,
        "assumed_covariance": plan.assumed_covariance,
        "baseline": _option_json(plan.baseline),
        "options": [_burn_json(burn) for burn in burns],
        "notes": list(plan.notes),
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


def _burns_table(plan: ManeuverPlan, burns: Sequence[_Burn]) -> Table:
    """A table of the baseline and the burns."""
    rows = table("Burns of the primary")
    rows.add_column("Option", no_wrap=True)
    for column in ("Before TCA (h)", "Δv (m/s)", "Pc", "Range (m)"):
        rows.add_column(column, justify="right", no_wrap=True)
    rows.add_column("Method", no_wrap=True)
    rows.add_column("Target", no_wrap=True)
    rows.add_row(*_row("no burn", "-", "-", plan.baseline))
    for burn in burns:
        option = burn.option
        lead = f"{option.burn_lead / HOUR:.1f}"
        rows.add_row(*_row(burn.direction, lead, f"{option.delta_v:+.3g}", option))
    return rows


def _event_line(result: RescreenResult, secondary: str) -> str:
    """What the re-screen says of the event: possibly nothing. It screens
    element sets with SGP4, not the CDM's orbits, so it says so."""
    if not result.secondary_screened:
        return (
            f"Event secondary {secondary} was NOT screened: this re-screen says "
            "nothing of the event."
        )
    view = (
        f"Event secondary {secondary} on the element sets (SGP4, not the CDM's orbit)"
    )
    if result.event_range is None:
        return (
            f"{view}: no approach within {TCA_TOLERANCE / MINUTE:g} min of the "
            "CDM's TCA after the burn."
        )
    return f"{view}: range {result.event_range:.0f} m after the burn."


def _conflict_line(result: RescreenResult) -> str:
    if result.conflict:
        return "CONFLICT: a new or closer approach is at red."
    return "No new or closer approach at red among the objects screened."


def _return_lines(result: RescreenResult) -> list[str]:
    """The times the burn brings the event's pair back above the target."""
    return [
        (
            f"The burn brings this pair back at {minute(item.approach.tca)} UTC at "
            f"Pc {pc_text(item.pc)} (assumed), above the target."
        )
        for item in result.event_returns
    ]


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
    direction: str, result: RescreenResult, secondary: str, label: str
) -> list[str]:
    """What the re-screen of a burn found, for people."""
    start, end = result.window
    left_out = plural(result.skipped_objects, "catalog object")
    approaches = [
        *_approach_lines(result.new_approaches, "new approach", "new approaches"),
        *_approach_lines(
            result.closer_approaches, "closer approach", "closer approaches"
        ),
    ]
    assumed = (
        [
            (
                "Covariances are assumed (keepout's model of GP element set errors), "
                "not orbit determination."
            )
        ]
        if approaches or result.event_returns
        else []
    )
    body = [
        _event_line(result, secondary),
        _conflict_line(result),
        *_return_lines(result),
        *approaches,
        *assumed,
    ]
    return [
        (
            f"{direction} re-screened against "
            f"{plural(result.catalog_objects, 'object')} ({label}) from {utc(start)} "
            f"to {utc(end)} UTC; {left_out} left out, of which it says nothing."
        ),
        *(f"  {line}" for line in body),
    ]


def _summary(
    plan: ManeuverPlan,
    burns: Sequence[_Burn],
    gaps: Sequence[str],
    catalog_label: str | None,
) -> list[RenderableType]:
    """The result for people: the event, the table of burns, the notes and
    what each re-screen found."""
    event = plan.event
    latest = event.latest
    head = (
        f"Event {event.event_id}: {_objects(event)}, TCA {minute(latest.tca)} UTC, "
        f"target Pc {pc_text(plan.target_pc)}."
    )
    notes = [f"Note: {note}" for note in plan.notes]
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
            for burn in burns
            if burn.rescreen is not None
            for line in _rescreen_lines(
                burn.direction, burn.rescreen, secondary, catalog_label
            )
        ]
    return [head, _burns_table(plan, burns), *gaps, *notes, *rescreens]
