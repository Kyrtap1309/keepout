"""How ``keepout screen`` and ``keepout assess`` show assessed events."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

from rich.table import Table

from keepout.cli.output import JsonValue, isot, minute, pc_text, table

if TYPE_CHECKING:
    from keepout.cdm import CdmObject
    from keepout.risk import Action, Assessment, ConjunctionEvent, PcObservation

__all__ = [
    "AssessedEvent",
    "event_json",
    "events_table",
    "reasons",
]


class AssessedEvent(NamedTuple):
    """A conjunction event and its risk assessment."""

    event: ConjunctionEvent
    assessment: Assessment


def _level(item: AssessedEvent) -> str | None:
    level = item.assessment.level
    return None if level is None else level.label


def _pc(observation: PcObservation | None) -> str:
    if observation is None:
        return "-"
    if observation.pc is None:
        return "failed"
    return pc_text(observation.pc)


def events_table(events: Sequence[AssessedEvent], title: str) -> Table:
    """A table of the events: TCA, both objects' designators, miss distance,
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
        rows.add_row(
            minute(latest.tca),
            f"{latest.object1.designator} / {latest.object2.designator}",
            f"{latest.miss_distance:.0f}",
            str(len(item.event.cdms)),
            _pc(item.assessment.trend.latest_observation),
            _level(item) or "-",
            item.assessment.decision.action.label,
        )
    return rows


def reasons(events: Sequence[AssessedEvent], least: Action) -> list[str]:
    """Why each event with an action of at least ``least`` got it, a line
    per event."""
    return [
        f"{item.event.event_id}: {decision.action.label} "
        f"({'; '.join(decision.reasons)})"
        for item in events
        if (decision := item.assessment.decision).action >= least
    ]


def _cdm_object_json(cdm_object: CdmObject) -> dict[str, JsonValue]:
    return {"designator": cdm_object.designator, "name": cdm_object.name}


def _observation_json(observation: PcObservation) -> dict[str, JsonValue]:
    result = observation.result
    return {
        "message_id": observation.message_id,
        "time_to_tca": observation.time_to_tca,
        "pc": observation.pc,
        "method": None if result is None else result.method,
        "error": observation.error,
        "reported_pc": observation.reported_pc,
        "diagnostics": list(observation.diagnostics),
        "assumed_covariance": observation.assumed_covariance,
    }


def event_json(item: AssessedEvent) -> dict[str, JsonValue]:
    """The event, its CDMs' Pc and its assessment as JSON values."""
    event, assessment = item
    latest = event.latest
    trend, decision = assessment.trend, assessment.decision
    observation = trend.latest_observation
    return {
        "event_id": event.event_id,
        "tca": isot(latest.tca),
        "object1": _cdm_object_json(latest.object1),
        "object2": _cdm_object_json(latest.object2),
        "miss_distance": latest.miss_distance,
        "cdms": [cdm.message_id for cdm in event.cdms],
        "pc": None if observation is None else observation.pc,
        "level": _level(item),
        "trend": str(trend.direction),
        "action": decision.action.name.lower(),
        "reasons": list(decision.reasons),
        "notes": list(trend.notes),
        "observations": [_observation_json(o) for o in trend.observations],
        "assumed_covariances": event.assumed_covariance,
    }
