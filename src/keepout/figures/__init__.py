"""Plot data without a plotting library, for the reports and the dashboard.

Each figure is a frozen dataclass of arrays and labels, drawn by whoever
holds a plotting library (matplotlib for the reports, Plotly for the
dashboard): :func:`trend_figure` gives an event's Pc against the time to
TCA (:class:`TrendFigure`), :func:`grid_figure` the heatmap of a manoeuvre
plan (:class:`GridFigure`, a :class:`GridPanel` per direction),
:func:`encounter_figure` the encounter-plane picture
(:class:`EncounterFigure`) and :func:`geometry_figure` the 3-D relative
geometry at TCA in the primary's RTN frame (:class:`GeometryFigure`).
:mod:`~keepout.figures.palette` holds the shared colour-blind-safe colours,
and :func:`provenance` labels what a Pc rests on (:class:`Provenance`),
:func:`covariance_provenance` that of a plan on a CDM's covariances.
"""

from keepout.figures.encounter import SIGMAS, EncounterFigure, encounter_figure
from keepout.figures.geometry import GeometryFigure, geometry_figure
from keepout.figures.grid import LOWEST_LOG_PC, GridFigure, GridPanel, grid_figure
from keepout.figures.palette import (
    ACTION_COLOURS,
    ACTION_LABELS,
    RISK_COLOURS,
    RISK_LABELS,
    SEQUENTIAL,
    SERIES,
)
from keepout.figures.provenance import Provenance, covariance_provenance, provenance
from keepout.figures.trend import TrendFigure, trend_figure

__all__ = [
    "ACTION_COLOURS",
    "ACTION_LABELS",
    "LOWEST_LOG_PC",
    "RISK_COLOURS",
    "RISK_LABELS",
    "SEQUENTIAL",
    "SERIES",
    "SIGMAS",
    "EncounterFigure",
    "GeometryFigure",
    "GridFigure",
    "GridPanel",
    "Provenance",
    "TrendFigure",
    "covariance_provenance",
    "encounter_figure",
    "geometry_figure",
    "grid_figure",
    "provenance",
    "trend_figure",
]
