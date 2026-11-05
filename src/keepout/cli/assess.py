"""``keepout assess``: the risk of conjunction events, from their CDMs."""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Annotated, Final, TypeAlias

import typer

from keepout._defaults import DECISION_POINT, TCA_TOLERANCE
from keepout._methods import METHOD_CHOICES, MethodChoice
from keepout.cli.cdm_input import CdmFiles, load_cdms
from keepout.cli.events import AssessedEvent, show_assessed
from keepout.cli.options import HardBodyRadius, JsonFlag
from keepout.cli.output import duration_words
from keepout.cli.store import StoreOption, add_and_show

# The library is imported where it is used: keepout --help need not load the
# numerical libraries (keepout._methods loads none).

__all__ = [
    "AUTO_METHOD",
    "PcMethod",
    "assess_command",
    "method_choice",
]

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from keepout.store.config import Settings

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


MethodOption: TypeAlias = Annotated[
    PcMethod,
    typer.Option(
        "--method",
        help="Pc method; auto picks one per CDM (a 3-D Pc can take a minute).",
    ),
]
"""``--method``: the Pc method."""


def assess_command(
    files: CdmFiles,
    *,
    method: MethodOption = AUTO_METHOD,
    hbr: HardBodyRadius = None,
    store: StoreOption = None,
    as_json: JsonFlag = False,
) -> None:
    """Compute the Pc of each CDM and the risk and action of each event.

    CDMs of the same two objects whose TCAs are within {tca_tolerance} form one
    event. Each CDM's Pc is computed from its states and covariances, and
    the event's risk level, Pc trend and recommended action are decided as
    of the latest CDM's creation date (a manoeuvre needs a red Pc within
    {decision_point} of TCA). With --store the CDMs are added to the store
    and each event is the stored one, with its id and every CDM stored of
    it. Exits with status 3 if at least one event's action is to manoeuvre.
    """
    from keepout.pc import computation

    compute = computation(method_choice(method.value))
    if store is not None:

        def settings() -> Settings:
            from keepout.store.config import Settings

            return Settings(compute=compute, combined_hard_body_radius=hbr)

        add_and_show(store, files, settings, as_json=as_json)
        return

    from keepout.events import group_events
    from keepout.risk import assess

    messages = load_cdms(files)
    assessed = [
        AssessedEvent(
            event,
            assess(event.observations(compute=compute, combined_hard_body_radius=hbr)),
        )
        for event in group_events(messages)
    ]
    show_assessed(assessed, cdm_count=len(messages), store=None, as_json=as_json)


# The defaults in the help are the library's own (none under python -OO).
assess_command.__doc__ = (assess_command.__doc__ or "").format_map(
    {
        "tca_tolerance": duration_words(TCA_TOLERANCE),
        "decision_point": duration_words(DECISION_POINT),
    }
)
