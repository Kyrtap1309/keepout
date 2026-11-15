"""Colours shared by the figures: risk levels, actions, series and a ramp.

The risk colours stay apart for readers with a red-green colour vision
deficiency (deuteranopia and protanopia), and every figure shows the level
in words as well (:data:`RISK_LABELS`): colour is never the only cue.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from keepout.risk import Action, RiskLevel

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "ACTION_COLOURS",
    "ACTION_LABELS",
    "RISK_COLOURS",
    "RISK_LABELS",
    "SEQUENTIAL",
    "SERIES",
]

RISK_COLOURS: Final[Mapping[RiskLevel, str]] = MappingProxyType(
    {
        RiskLevel.GREEN: "#009e73",
        RiskLevel.YELLOW: "#f0e442",
        RiskLevel.RED: "#8b0000",
    }
)
"""The colour of each risk level: Okabe and Ito's bluish green and yellow,
and a dark red that stays apart from both under a deuteranopia or
protanopia simulation."""

RISK_LABELS: Final[Mapping[RiskLevel, str]] = MappingProxyType(
    {level: level.key for level in RiskLevel}
)
"""Each risk level in words: ``green``, ``yellow``, ``red``."""

ACTION_COLOURS: Final[Mapping[Action, str]] = MappingProxyType(
    {
        Action.IGNORE: "#009e73",
        Action.MONITOR: "#e69f00",
        Action.PREPARE_MANEUVER: "#d55e00",
        Action.MANEUVER: "#8b0000",
    }
)
"""The colour of each recommended action, from green to dark red."""

ACTION_LABELS: Final[Mapping[Action, str]] = MappingProxyType(
    {action: action.label for action in Action}
)
"""Each action in words, as keepout prints it: ``prepare maneuver``."""

SERIES: Final = (
    "#e69f00",
    "#56b4e9",
    "#009e73",
    "#f0e442",
    "#0072b2",
    "#d55e00",
    "#cc79a7",
)
"""Colours of unordered series, in Okabe and Ito's order (without black)."""

SEQUENTIAL: Final = (
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
)
"""A sequential blue ramp, light to dark: a higher Pc is darker."""
