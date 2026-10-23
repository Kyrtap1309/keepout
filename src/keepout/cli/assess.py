"""``keepout assess``: the risk of conjunction events, from their CDMs."""

from __future__ import annotations

import enum
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, TypeAlias

import typer

from keepout._defaults import DECISION_POINT, TCA_TOLERANCE
from keepout._methods import METHOD_CHOICES, MethodChoice
from keepout.cli.events import (
    AssessedEvent,
    event_json,
    events_table,
    reasons,
)
from keepout.cli.options import JsonFlag, positive
from keepout.cli.output import (
    ExitCode,
    JsonOut,
    duration_words,
    fail,
    plural,
    show,
)
from keepout.errors import KeepoutError

# The library is imported where it is used: keepout --help need not load the
# numerical libraries (keepout._methods loads none).

__all__ = ["AUTO_METHOD", "PcMethod", "assess_command", "method_choice"]

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    class PcMethod(enum.StrEnum):
        """``--method``: the Pc methods of :func:`keepout.pc.compute_pc`."""

else:
    # Built from the names, so a new method needs no edit here; the type
    # checkers see an enum whose members they do not know.
    PcMethod = enum.StrEnum(
        "PcMethod", {name.upper(): name for name in METHOD_CHOICES}, module=__name__
    )
    PcMethod.__doc__ = "``--method``: the Pc methods of :func:`keepout.pc.compute_pc`."

AUTO_METHOD: Final = PcMethod("auto")
"""``--method``'s default."""


def _is_method(name: str) -> TypeIs[MethodChoice]:
    return name in METHOD_CHOICES


def method_choice(name: str) -> MethodChoice:
    """The :data:`~keepout.pc.MethodChoice` named ``name`` (a :class:`PcMethod`).

    Raises
    ------
    ValueError
        If ``name`` names no Pc method.
    """
    if not _is_method(name):
        raise ValueError(f"unknown Pc method {name!r}")
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
    method: MethodOption = AUTO_METHOD,
    hbr: HardBodyRadius = None,
    as_json: JsonFlag = False,
) -> None:
    """Compute the Pc of each CDM and the risk and action of each event.

    CDMs of the same two objects whose TCAs are within {tca_tolerance} form one
    event. Each CDM's Pc is computed from its states and covariances, and
    the event's risk level, Pc trend and recommended action are decided as
    of the latest CDM's creation date (a manoeuvre needs a red Pc within
    {decision_point} of TCA). Exits with status 3 if at least one event's
    action is to manoeuvre.
    """
    from keepout import cdm
    from keepout.events import group_events
    from keepout.pc import computation
    from keepout.risk import Action, assess

    messages = []
    for path in files:
        try:
            messages.append(cdm.load(path))
        except KeepoutError as exc:
            fail(f"{path}: {exc}")
    compute = computation(method_choice(method.value))
    assessed = [
        AssessedEvent(
            event, assess(event.observations(compute=compute, hard_body_radius=hbr))
        )
        for event in group_events(messages)
    ]
    actions = Counter(item.assessment.decision.action for item in assessed)
    manoeuvre = actions[Action.MANOEUVRE] > 0
    assumed = sum(item.event.assumed_covariance for item in assessed)
    payload: dict[str, JsonOut] = {
        "cdms": len(messages),
        "events": [event_json(item) for item in assessed],
        "assumed_covariances": assumed,
        "manoeuvre": manoeuvre,
    }
    counted = ", ".join(
        f"{actions[action]} {action.label}" for action in sorted(actions, reverse=True)
    )
    text = [
        (
            f"{plural(len(assessed), 'event')} from {plural(len(messages), 'CDM')}: "
            f"{counted}."
        ),
    ]
    if assumed:
        text.append(
            f"{assumed} of {plural(len(assessed), 'event')} "
            f"{'rests' if assumed == 1 else 'rest'} on assumed covariances "
            "(COVARIANCE_METHOD = DEFAULT), not orbit determination."
        )
    text += reasons(assessed, Action.MONITOR)
    if manoeuvre:
        text.append(
            "At least one event needs a manoeuvre "
            f"(exit status {int(ExitCode.MANOEUVRE)})."
        )
    show(
        payload,
        events_table(assessed, "Conjunction events"),
        *text,
        as_json=as_json,
    )
    if manoeuvre:
        raise typer.Exit(ExitCode.MANOEUVRE)


# The defaults in the help are the library's own (none under python -OO).
assess_command.__doc__ = (assess_command.__doc__ or "").format_map(
    {
        "tca_tolerance": duration_words(TCA_TOLERANCE),
        "decision_point": duration_words(DECISION_POINT),
    }
)
