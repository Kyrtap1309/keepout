"""What a manoeuvre outcome says, in words: why no burn can be recommended,
the table of burns, the direction without a burn, and what a re-screen
found.

Shared by ``keepout maneuver`` and the dashboard's planner page, so both read
an outcome the same way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal, TypeAlias, TypedDict

from keepout._text import minute, pc_text, plural, utc
from keepout.maneuver.planner import BurnDirection
from keepout.units import HOUR, MINUTE

if TYPE_CHECKING:
    from keepout.maneuver.outcome import ManeuverOutcome
    from keepout.maneuver.planner import ManeuverOption
    from keepout.maneuver.rescreen import NewApproach, RescreenResult

__all__ = [
    "NO_BURN_NEEDED",
    "ApproachKind",
    "BurnRow",
    "burn_rows",
    "conflict_line",
    "event_line",
    "gap_lines",
    "ranked_approaches",
    "refusal",
    "rescreen_head",
    "return_lines",
]


NO_BURN_NEEDED: Final = "No burn needed: the baseline Pc already meets the target."
"""The verdict of an outcome whose baseline already meets the target."""


def refusal(outcome: ManeuverOutcome) -> str:
    """Why no burn of an outcome can be recommended, by its
    :attr:`~keepout.maneuver.ManeuverOutcome.verdict`, without a full stop:
    ``No burn on the grid meets the target Pc 1.00e-05``. Only for an
    outcome whose verdict is neither recommendable nor
    :attr:`~keepout.maneuver.BurnVerdict.no_burn_needed`."""
    verdict = outcome.verdict
    target = f"the target Pc {pc_text(outcome.target_pc)}"
    if not verdict.burns:
        return f"No burn on the grid meets {target}"
    if verdict.missed_target:
        return f"This burn misses {target}"
    brought = []
    if verdict.brings_red:
        brought.append("an approach at red")
    if verdict.brings_pair_back:
        brought.append(f"this pair back above {target}")
    burns = "Every burn" if verdict.burns > 1 else "This burn"
    return f"{burns} brings {' or '.join(brought)}"


BurnRow = TypedDict(
    "BurnRow",
    {
        "Option": str,
        "Before TCA (h)": float | None,
        "Δv (m/s)": float | None,
        "Pc": float,
        "Range (m)": float,
        "Method": str,
        "Target": str,
    },
)
"""A row of the table of burns, by column: the option (``no burn`` or the
direction), its time before TCA [h] and signed Δv [m/s] (``None`` without
a burn), its Pc and range at the closest approach [m], the method, and
whether it meets the target (``met`` or ``missed``)."""


def _burn_row(name: str, option: ManeuverOption, *, burn: bool) -> BurnRow:
    return {
        "Option": name,
        "Before TCA (h)": option.burn_lead / HOUR if burn else None,
        "Δv (m/s)": option.delta_v if burn else None,
        "Pc": option.pc,
        "Range (m)": option.range_at_tca,
        "Method": option.method,
        "Target": "met" if option.meets_target else "missed",
    }


def burn_rows(outcome: ManeuverOutcome) -> list[BurnRow]:
    """The table of burns of an outcome: the baseline (``no burn``), then
    each burn."""
    rows = [_burn_row("no burn", outcome.baseline, burn=False)]
    rows += [
        _burn_row(str(burn.direction), burn.option, burn=True) for burn in outcome.burns
    ]
    return rows


ApproachKind: TypeAlias = Literal["new", "closer"]
"""Whether a burn brings an approach (``new``) or brings it closer."""


def ranked_approaches(result: RescreenResult) -> list[tuple[ApproachKind, NewApproach]]:
    """The approaches a re-screen found, new then closer, each with its
    kind, the highest Pc first."""
    found: list[tuple[ApproachKind, NewApproach]] = []
    found.extend(("new", item) for item in result.new_approaches)
    found.extend(("closer", item) for item in result.closer_approaches)
    return sorted(found, key=lambda pair: pair[1].pc, reverse=True)


def gap_lines(outcome: ManeuverOutcome) -> list[str]:
    """The line of the direction without a burn, if the other has one; none
    for a burn of one's own (no grid)."""
    if outcome.grid is None:
        return []
    found = {burn.direction for burn in outcome.burns}
    return [
        f"No {direction} burn on the grid meets the target."
        for direction in BurnDirection
        if found and direction not in found
    ]


def rescreen_head(direction: BurnDirection, result: RescreenResult, label: str) -> str:
    """What the re-screen of a burn looked at: the catalog ``label``, its
    window and the objects it left out."""
    start, end = result.window
    left_out = plural(result.skipped_objects, "catalog object")
    return (
        f"{direction} re-screened against "
        f"{plural(result.catalog_objects, 'object')} ({label}) from {utc(start)} "
        f"to {utc(end)} UTC; {left_out} left out (not checked)."
    )


def event_line(result: RescreenResult, secondary: str, tolerance: float) -> str:
    """What the re-screen says of the event: possibly nothing. It screens
    element sets with SGP4, not the CDM's orbits, so it says so; ``tolerance``
    is the event's TCA tolerance [s], which the re-screen used."""
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
            f"{view}: no approach within {tolerance / MINUTE:g} min of the "
            "CDM's TCA after the burn."
        )
    return f"{view}: range {result.event_range:.0f} m after the burn."


def conflict_line(result: RescreenResult) -> str:
    """Whether the burn brings a new or closer approach at red."""
    if result.conflict:
        return "CONFLICT: a new or closer approach is at red."
    return "No new or closer approach at red among the objects screened."


def return_lines(result: RescreenResult) -> list[str]:
    """The times the burn brings the event's pair back above the target."""
    return [
        (
            f"The burn brings this pair back at {minute(item.approach.tca)} UTC at "
            f"Pc {pc_text(item.pc)} (assumed), above the target."
        )
        for item in result.event_returns
    ]
