"""What ``keepout maneuver`` prints of an outcome: text, or JSON with
``--json``, and its exit status."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

import typer

from keepout.cli.output import (
    ASSUMED_COVARIANCES,
    ExitCode,
    JsonOut,
    isot,
    minute,
    objects,
    pc_text,
    plural,
    show,
    table,
)

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

__all__ = ["show_outcome"]

_SHOWN_APPROACHES: Final = 5
"""Most approaches of one kind the text lists, the highest Pc first."""


def show_outcome(outcome: ManeuverOutcome, *, as_json: bool) -> None:
    """The output of an outcome, as text or JSON.

    Raises
    ------
    typer.Exit
        With status 4 if no burn can be recommended.
    """
    from keepout.maneuver.wording import NO_BURN_NEEDED

    text = _summary(outcome)
    refused = _refused(outcome)
    if outcome.verdict.no_burn_needed:
        text.append(NO_BURN_NEEDED)
    elif refused:
        text.append(_refusal(outcome))
    show(_payload(outcome), *text, as_json=as_json)
    if refused:
        raise typer.Exit(ExitCode.NO_SAFE_BURN)


def _refused(outcome: ManeuverOutcome) -> bool:
    """Whether no burn can be recommended where one is needed: exit status 4."""
    verdict = outcome.verdict
    return not (verdict.recommendable or verdict.no_burn_needed)


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
    """The result as JSON values; its ``refusal`` is the line saying why no
    burn can be recommended (``None`` if one can)."""
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
        "refusal": _refusal(outcome) if _refused(outcome) else None,
    }


def _burns_table(outcome: ManeuverOutcome) -> Table:
    """A table of the baseline and the burns."""
    from keepout.maneuver.wording import burn_rows

    rows = table("Burns of the primary")
    rows.add_column("Option", no_wrap=True)
    for column in ("Before TCA (h)", "Δv (m/s)", "Pc", "Range (m)"):
        rows.add_column(column, justify="right", no_wrap=True)
    rows.add_column("Method", no_wrap=True)
    rows.add_column("Target", no_wrap=True)
    for row in burn_rows(outcome):
        lead, delta_v = row["Before TCA (h)"], row["Δv (m/s)"]
        rows.add_row(
            row["Option"],
            "-" if lead is None else f"{lead:.1f}",
            "-" if delta_v is None else f"{delta_v:+.3g}",
            pc_text(row["Pc"]),
            f"{row['Range (m)']:.0f}",
            row["Method"],
            row["Target"],
        )
    return rows


def _approach_lines(
    approaches: Sequence[NewApproach], noun: str, nouns: str
) -> list[str]:
    """The approaches of one kind, ranked, at most :data:`_SHOWN_APPROACHES`,
    each labelled: the Pc is on assumed covariances."""
    if not approaches:
        return []
    count = plural(len(approaches), noun, nouns)
    more = len(approaches) - _SHOWN_APPROACHES
    head = (
        f"{count}, the {_SHOWN_APPROACHES} of highest Pc:" if more > 0 else f"{count}:"
    )
    lines = [
        (
            f"  {item.approach.secondary_number} at {minute(item.approach.tca)} UTC: "
            f"range {item.approach.range_at_tca:.0f} m, "
            f"assumed Pc {pc_text(item.pc)}, {item.level.key}"
        )
        for item in approaches[:_SHOWN_APPROACHES]
    ]
    tail = [f"  and {more} more (--json lists them)."] if more > 0 else []
    return [head, *lines, *tail]


def _rescreen_lines(
    direction: BurnDirection,
    result: RescreenResult,
    event: ConjunctionEvent,
    label: str,
) -> list[str]:
    """What the re-screen of a burn found, for people."""
    from keepout.maneuver.wording import (
        conflict_line,
        event_line,
        ranked_approaches,
        rescreen_head,
        return_lines,
    )

    ranked = ranked_approaches(result)
    new = [item for kind, item in ranked if kind == "new"]
    closer = [item for kind, item in ranked if kind == "closer"]
    approaches = [
        *_approach_lines(new, "new approach", "new approaches"),
        *_approach_lines(closer, "closer approach", "closer approaches"),
    ]
    body = [
        event_line(result, event.latest.object2.object_id, event.tca_tolerance),
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
    from keepout.maneuver.wording import gap_lines

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
        rescreens = [
            line
            for burn in outcome.burns
            if burn.rescreen is not None
            for line in _rescreen_lines(
                burn.direction, burn.rescreen, event, catalog_label
            )
        ]
    return [head, _burns_table(outcome), *gap_lines(outcome), *notes, *rescreens]
