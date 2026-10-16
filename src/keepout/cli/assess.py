"""``keepout assess``: the risk of conjunction events, from their CDMs."""

from __future__ import annotations

import enum
from collections import Counter
from pathlib import Path
from typing import Annotated, TypeAlias, TypeGuard, get_args

import typer

from keepout import cdm
from keepout.cli.events import (
    AssessedEvent,
    event_json,
    events_table,
    reasons,
)
from keepout.cli.options import JsonFlag, positive
from keepout.cli.output import ExitCode, JsonValue, fail, plural, show
from keepout.errors import KeepoutError
from keepout.pc import Method, MethodChoice
from keepout.risk import Action, assess, group_events

__all__ = ["PcMethod", "assess_command", "method_choice"]


class PcMethod(enum.StrEnum):
    """``--method``: the Pc methods of :func:`keepout.pc.compute_pc`."""

    AUTO = "auto"
    FOSTER = "foster"
    ELROD = "elrod"
    HALL3D = "hall3d"
    MONTECARLO = "montecarlo"
    FRISBEE = "frisbee"
    DILUTION = "dilution"
    ALFANO = "alfano"


def _is_method(name: str) -> TypeGuard[MethodChoice]:
    return name == "auto" or name in get_args(Method)


def method_choice(name: str) -> MethodChoice:
    """The :data:`~keepout.pc.MethodChoice` named ``name`` (a :class:`PcMethod`).

    Raises
    ------
    ValueError
        If ``name`` names no Pc method.
    """
    if not _is_method(name):
        raise ValueError(f"not a Pc method: {name!r}")
    return name


Files: TypeAlias = Annotated[
    list[Path],
    typer.Argument(
        exists=True,
        dir_okay=False,
        help="CDM files, KVN or XML, gzipped or not.",
    ),
]
"""The CDM files to assess."""

MethodOption: TypeAlias = Annotated[
    PcMethod,
    typer.Option(
        "--method",
        help="Pc method; auto picks one per CDM (a 3-D Pc can take a minute).",
    ),
]
"""``--method``: the Pc method."""

HardBodyRadius: TypeAlias = Annotated[
    float | None,
    typer.Option(
        "--hbr",
        callback=positive,
        metavar="METRES",
        help="Combined hard-body radius [m] (default: the sum of the CDMs' HBR).",
    ),
]
"""``--hbr``: the combined hard-body radius instead of the CDMs'."""


def assess_command(
    files: Files,
    *,
    method: MethodOption = PcMethod.AUTO,
    hbr: HardBodyRadius = None,
    as_json: JsonFlag = False,
) -> None:
    """Compute the Pc of each CDM and the risk and action of each event.

    CDMs of the same two objects whose TCAs are within 10 minutes form one
    event. Each CDM's Pc is computed from its states and covariances, and
    the event's risk level, Pc trend and recommended action are decided as
    of the latest CDM's creation date (a manoeuvre needs a red Pc within two
    days of TCA). Exits with status 3 if at least one event's action is to
    manoeuvre.
    """
    messages = []
    for path in files:
        try:
            messages.append(cdm.load(path))
        except KeepoutError as exc:
            fail(f"{path}: {exc}")
    choice = method_choice(method.value)
    assessed = [
        AssessedEvent(
            event, assess(event.observations(method=choice, hard_body_radius=hbr))
        )
        for event in group_events(messages)
    ]
    actions = Counter(item.assessment.decision.action for item in assessed)
    manoeuvre = actions[Action.MANOEUVRE] > 0
    assumed = sum(item.event.assumed_covariance for item in assessed)
    payload: dict[str, JsonValue] = {
        "cdms": len(messages),
        "events": [event_json(item) for item in assessed],
        "assumed_covariances": assumed,
        "manoeuvre": manoeuvre,
    }
    counted = ", ".join(
        f"{actions[action]} {action.label}" for action in sorted(actions, reverse=True)
    )
    text = [
        f"{plural(len(assessed), 'event')} from {plural(len(messages), 'CDM')}: "
        f"{counted}.",
    ]
    if assumed:
        text.append(
            f"{assumed} of {plural(len(assessed), 'event')} "
            f"{'rests' if assumed == 1 else 'rest'} on assumed covariances "
            "(COVARIANCE_METHOD = DEFAULT), not orbit determination."
        )
    text += reasons(assessed, Action.MONITOR)
    if manoeuvre:
        text.append("At least one event needs a manoeuvre (exit status 3).")
    show(
        payload,
        events_table(assessed, "Conjunction events"),
        *text,
        as_json=as_json,
    )
    if manoeuvre:
        raise typer.Exit(ExitCode.MANOEUVRE)
