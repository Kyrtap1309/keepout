"""Colours shared by the figures: risk levels, series, a ramp and marks;
and the words of risk levels, actions and provenances.

The risk colours stay apart for readers with a red-green colour vision
deficiency (deuteranopia and protanopia), and every figure shows the level
in words as well (:data:`RISK_LABELS`): colour is never the only cue.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from keepout.risk import Action, Provenance, RiskLevel

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "ACTION_LABELS",
    "FIT_COLOUR",
    "NO_PC",
    "PROVENANCE_HELP",
    "PROVENANCE_LABELS",
    "RISK_COLOURS",
    "RISK_LABELS",
    "SEQUENTIAL",
    "SERIES",
    "TARGET_COLOUR",
    "THRESHOLD_LABELS",
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

THRESHOLD_LABELS: Final[Mapping[RiskLevel, str]] = MappingProxyType(
    {RiskLevel.YELLOW: "green below", RiskLevel.RED: "red from"}
)
"""The words of the threshold line at which each level starts."""

ACTION_LABELS: Final[Mapping[Action, str]] = MappingProxyType(
    {action: action.label for action in Action}
)
"""Each action in words, as keepout prints it: ``prepare maneuver``."""

PROVENANCE_LABELS: Final[Mapping[Provenance, str]] = MappingProxyType(
    {
        Provenance.CDM_COVARIANCE: "CDM covariance",
        Provenance.GP_AGE_MODEL: "assumed (GP age model)",
        Provenance.ASSUMED: "assumed covariance",
        Provenance.MAX_PC: "max-Pc (no covariance)",
    }
)
"""Each provenance in words, as a badge or a table names it."""

PROVENANCE_HELP: Final[Mapping[Provenance, str]] = MappingProxyType(
    {
        Provenance.CDM_COVARIANCE: (
            "The Pc uses the covariances the CDM gives, from orbit determination."
        ),
        Provenance.GP_AGE_MODEL: (
            "The covariances are assumed by keepout's synthetic operator from "
            "the age of the GP data, not from orbit determination."
        ),
        Provenance.ASSUMED: (
            "The CDM marks its covariances as assumed (COVARIANCE_METHOD = "
            "DEFAULT), from a model it does not name."
        ),
        Provenance.MAX_PC: "A maximum Pc: an upper bound found without a covariance.",
    }
)
"""What each provenance means, in a sentence: a badge's text on hover."""

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

TARGET_COLOUR: Final = "#eb6834"
"""A manoeuvre plan's target contour and the smallest burn meeting it."""

NO_PC: Final = "#e4e2dc"
"""Heatmap cells without a Pc (no closest approach to trust): a neutral
grey."""

FIT_COLOUR: Final = "#757575"
"""The line joining an event's Pc: a mid grey that shows on light and dark
backgrounds."""
