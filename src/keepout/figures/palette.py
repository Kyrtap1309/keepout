"""Colours shared by the figures: risk levels, series, a ramp, marks and the
text and rules of a page's figure; and the words of threshold lines and
provenances.

The risk colours stay apart for readers with a red-green colour vision
deficiency (deuteranopia and protanopia), and every figure shows the level
in words as well (:attr:`RiskLevel.key <keepout.risk.RiskLevel>`): colour is
never the only cue.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from keepout.risk import Provenance, RiskLevel

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "DECISION_POINT_COLOUR",
    "FIT_COLOUR",
    "INK_COLOUR",
    "MUTED_COLOUR",
    "NO_PC",
    "PROVENANCE_HELP",
    "PROVENANCE_LABELS",
    "RISK_COLOURS",
    "RULE_COLOUR",
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

THRESHOLD_LABELS: Final[Mapping[RiskLevel, str]] = MappingProxyType(
    {RiskLevel.YELLOW: "green below", RiskLevel.RED: "red from"}
)
"""The words of the threshold line at which each level starts."""

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

DECISION_POINT_COLOUR: Final = "#424242"
"""The decision point's line on a trend: a dark grey, apart from the fit."""

INK_COLOUR: Final = "#0b0b0b"
"""A figure's titles and labels on a white page: near black."""

MUTED_COLOUR: Final = "#52514e"
"""A figure's axis labels and ticks on a white page: a dark warm grey."""

RULE_COLOUR: Final = "#c3c2b7"
"""A figure's spines and outlines on a white page: a light warm grey."""
