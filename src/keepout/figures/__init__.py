"""Plot data without a plotting library, for the reports and the dashboard.

Each figure is a frozen dataclass of arrays and labels, drawn by whoever
holds a plotting library (matplotlib for the reports, Plotly for the
dashboard): :func:`trend_figure` gives an event's Pc against the time to
TCA (:class:`TrendFigure`), :func:`grid_figure` the heatmap of a manoeuvre
plan (:class:`GridFigure`, a :class:`GridPanel` per direction),
:func:`encounter_figure` the encounter-plane picture
(:class:`EncounterFigure`) and :func:`geometry_figure` the 3-D relative
geometry at TCA in the primary's RTN frame (:class:`GeometryFigure`).
:mod:`~keepout.figures.palette` holds the shared colour-blind-safe colours
and the words of risk levels, actions and what a Pc rests on
(:data:`PROVENANCE_LABELS`, explained by :data:`PROVENANCE_HELP`, for a
:class:`keepout.risk.Provenance`).
"""

from keepout.figures.encounter import SIGMAS, EncounterFigure, encounter_figure
from keepout.figures.geometry import GeometryFigure, geometry_figure
from keepout.figures.grid import LOWEST_LOG_PC, GridFigure, GridPanel, grid_figure
from keepout.figures.palette import (
    ACTION_LABELS,
    FIT_COLOUR,
    NO_PC,
    PROVENANCE_HELP,
    PROVENANCE_LABELS,
    RISK_COLOURS,
    RISK_LABELS,
    SEQUENTIAL,
    SERIES,
    TARGET_COLOUR,
    THRESHOLD_LABELS,
)
from keepout.figures.trend import TrendFigure, trend_figure

__all__ = [
    "ACTION_LABELS",
    "FIT_COLOUR",
    "LOWEST_LOG_PC",
    "NO_PC",
    "PROVENANCE_HELP",
    "PROVENANCE_LABELS",
    "RISK_COLOURS",
    "RISK_LABELS",
    "SEQUENTIAL",
    "SERIES",
    "SIGMAS",
    "TARGET_COLOUR",
    "THRESHOLD_LABELS",
    "EncounterFigure",
    "GeometryFigure",
    "GridFigure",
    "GridPanel",
    "TrendFigure",
    "encounter_figure",
    "geometry_figure",
    "grid_figure",
    "trend_figure",
]
