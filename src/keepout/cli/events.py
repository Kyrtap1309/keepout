"""How ``keepout screen``, ``keepout assess`` and ``keepout store add`` show
assessed events."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, Protocol

import typer
from rich.table import Table

from keepout.cli.output import (
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

if TYPE_CHECKING:
    from keepout.cdm import CdmObject
    from keepout.events import ConjunctionEvent
    from keepout.risk import Action, Assessment, PcObservation

__all__ = [
    "Assessed",
    "event_json",
    "events_table",
    "reasons",
    "show_assessed",
]


class Assessed(Protocol):
    """Anything with a conjunction event and its risk assessment: a
    :class:`~keepout.events.AssessedEvent` or a
    :class:`~keepout.synthetic.OperatorEvent`."""

    @property
    def event(self) -> ConjunctionEvent:
        """The event."""
        ...

    @property
    def assessment(self) -> Assessment:
        """Its risk assessment."""
        ...


def _pc(observation: PcObservation | None) -> str:
    if observation is None:
        return "-"
    if observation.pc is None:
        return "failed"
    return pc_text(observation.pc)


def events_table(events: Sequence[Assessed], title: str) -> Table:
    """A table of the events: TCA, both objects' ids, miss distance,
    CDMs, the latest Pc before TCA, its risk level and the action."""
    rows = table(title)
    rows.add_column("TCA (UTC)", no_wrap=True)
    rows.add_column("Objects", no_wrap=True)
    for column in ("Miss (m)", "CDMs", "Pc"):
        rows.add_column(column, justify="right", no_wrap=True)
    rows.add_column("Level", no_wrap=True)
    rows.add_column("Action")
    for item in events:
        latest = item.event.latest
        level = item.assessment.level
        rows.add_row(
            minute(latest.tca),
            objects(latest),
            f"{latest.miss_distance:.0f}",
            str(len(item.event.cdms)),
            _pc(item.assessment.trend.latest_observation),
            "-" if level is None else level.key,
            item.assessment.decision.action.label,
        )
    return rows


_DIAGNOSTIC: Final = "diagnostic: "
"""How :func:`keepout.risk.decide` starts a reason that is a Pc diagnostic."""


def reasons(events: Sequence[Assessed], least: Action) -> list[str]:
    """Why each event with an action of at least ``least`` got it: a line
    per event, then each diagnostic of its Pc on a line of its own,
    indented."""
    lines = []
    for item in events:
        decision = item.assessment.decision
        if decision.action < least:
            continue
        diagnostics = [r for r in decision.reasons if r.startswith(_DIAGNOSTIC)]
        others = [r for r in decision.reasons if not r.startswith(_DIAGNOSTIC)]
        lines.append(
            f"{item.event.event_id}: {decision.action.label} ({'; '.join(others)})"
        )
        lines += (f"  {diagnostic}" for diagnostic in diagnostics)
    return lines


def _cdm_object_json(cdm_object: CdmObject) -> dict[str, JsonOut]:
    return {"object_id": cdm_object.object_id, "name": cdm_object.name}


def _observation_json(observation: PcObservation) -> dict[str, JsonOut]:
    return {
        "message_id": observation.message_id,
        "time_to_tca_s": observation.time_to_tca,
        "pc": observation.pc,
        "method": observation.method,
        "error": observation.error,
        "reported_pc": observation.reported_pc,
        "diagnostics": list(observation.diagnostics),
        "assumed_covariance": observation.assumed_covariance,
    }


def event_json(item: Assessed) -> dict[str, JsonOut]:
    """The event, its CDMs' Pc and its assessment as JSON values."""
    event, assessment = item.event, item.assessment
    latest = event.latest
    trend, decision = assessment.trend, assessment.decision
    observation = trend.latest_observation
    return {
        "event_id": event.event_id,
        "tca": isot(latest.tca),
        "object1": _cdm_object_json(latest.object1),
        "object2": _cdm_object_json(latest.object2),
        "miss_distance_m": latest.miss_distance,
        "cdms": [cdm.message_id for cdm in event.cdms],
        "pc": None if observation is None else observation.pc,
        "level": None if assessment.level is None else assessment.level.key,
        "trend": str(trend.direction),
        "action": decision.action.key,
        "reasons": list(decision.reasons),
        "notes": list(trend.notes),
        "observations": [_observation_json(o) for o in trend.observations],
        "assumed_covariance": event.assumed_covariance,
    }


def show_assessed(
    assessed: Sequence[Assessed],
    *,
    cdm_count: int,
    store: Mapping[str, JsonOut] | None,
    store_lines: Sequence[str] = (),
    as_json: bool,
) -> None:
    """The output of ``keepout assess`` and ``keepout store add``: the events
    table, the counts of actions and why each event needs more than
    ``ignore``; then an exit with :attr:`ExitCode.MANEUVER` if one needs a
    manoeuvre.

    Parameters
    ----------
    assessed
        The events and their assessments, in the table's order.
    cdm_count
        The CDMs read.
    store
        What the store did, as JSON values (``None`` without a store).
    store_lines
        The same for people, after the counts.
    as_json
        Whether ``--json`` was given.

    Raises
    ------
    typer.Exit
        With :attr:`ExitCode.MANEUVER` if at least one event's action is to
        manoeuvre.
    """
    from keepout.risk import Action

    actions = Counter(item.assessment.decision.action for item in assessed)
    maneuver = actions[Action.MANEUVER] > 0
    assumed = sum(item.event.assumed_covariance for item in assessed)
    payload: dict[str, JsonOut] = {
        "cdm_count": cdm_count,
        "events": [event_json(item) for item in assessed],
        "assumed_covariance_events": assumed,
        "maneuver": maneuver,
        "store": store,
    }
    counted = ", ".join(
        f"{actions[action]} {action.label}" for action in sorted(actions, reverse=True)
    )
    # A stored event holds its earlier CDMs too, not only those read.
    source = (
        f"from {plural(cdm_count, 'CDM')}"
        if store is None
        else f"in the store for the {plural(cdm_count, 'CDM')} read"
    )
    text = [f"{plural(len(assessed), 'event')} {source}: {counted}."]
    if assumed:
        text.append(
            f"{assumed} of {plural(len(assessed), 'event')} "
            f"{'rests' if assumed == 1 else 'rest'} on assumed covariances "
            "(COVARIANCE_METHOD = DEFAULT), not orbit determination."
        )
    text += store_lines
    text += reasons(assessed, Action.MONITOR)
    if maneuver:
        text.append(
            "At least one event needs a manoeuvre "
            f"(exit status {int(ExitCode.MANEUVER)})."
        )
    show(
        payload,
        events_table(assessed, "Conjunction events"),
        *text,
        as_json=as_json,
    )
    if maneuver:
        raise typer.Exit(ExitCode.MANEUVER)
