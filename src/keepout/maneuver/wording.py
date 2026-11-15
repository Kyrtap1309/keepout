"""What a manoeuvre outcome says, in words: why no burn can be recommended,
the direction without a burn, and what a re-screen found.

Shared by ``keepout maneuver`` and the dashboard's planner page, so both read
an outcome the same way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from keepout._defaults import TCA_TOLERANCE
from keepout._text import minute, pc_text, plural, utc
from keepout.maneuver.planner import BurnDirection
from keepout.units import MINUTE

if TYPE_CHECKING:
    from keepout.maneuver.outcome import ManeuverOutcome
    from keepout.maneuver.rescreen import RescreenResult

__all__ = [
    "conflict_line",
    "event_line",
    "gap_lines",
    "refusal",
    "rescreen_head",
    "return_lines",
]


def refusal(outcome: ManeuverOutcome) -> str:
    """Why no burn of an outcome can be recommended, by its
    :attr:`~keepout.maneuver.ManeuverOutcome.verdict`, without a full stop:
    ``No burn on the grid meets the target Pc 1.00e-05``. Only for an
    outcome whose verdict is not recommendable."""
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


def event_line(result: RescreenResult, secondary: str) -> str:
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
