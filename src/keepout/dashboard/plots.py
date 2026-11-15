"""The figures of :mod:`keepout.figures` drawn with Plotly, as the reports
draw them with matplotlib: the trend of an event's Pc, the encounter plane,
the 3-D geometry at TCA and the heatmap of a manoeuvre plan.

Every trace is named, so the legend says in words what colour shows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from keepout.figures import (
    LOWEST_LOG_PC,
    RISK_COLOURS,
    RISK_LABELS,
    SEQUENTIAL,
    SERIES,
    SIGMAS,
)
from keepout.risk import RiskLevel

if TYPE_CHECKING:
    from keepout._types import FloatArray
    from keepout.figures import (
        EncounterFigure,
        GeometryFigure,
        GridFigure,
        GridPanel,
        TrendFigure,
    )

__all__ = ["encounter", "geometry", "grid", "trend"]

TARGET_COLOUR: Final = "#eb6834"
"""The target contour and the smallest burn meeting it, as in the
manoeuvre report."""

NO_PC: Final = "#e4e2dc"
"""Cells without a Pc: a neutral grey behind the heatmap."""

FIT_COLOUR: Final = "#757575"
"""The line joining an event's Pc, and the decision point's: a mid grey
that shows on light and dark themes."""

_THRESHOLD_LABELS: Final = {RiskLevel.YELLOW: "green below", RiskLevel.RED: "red from"}
"""The words of the threshold line at which each level starts."""

_COLOURSCALE: Final = tuple(
    (i / (len(SEQUENTIAL) - 1), colour) for i, colour in enumerate(SEQUENTIAL)
)
"""The heatmap's colours, light (low Pc) to dark."""


def _decision_side(
    hours: FloatArray, decision_point_hours: float
) -> Literal["top left", "top right"]:
    """Where the decision point's label goes: on the screen's right (nearer
    TCA on the reversed axis) unless the points reach further to its left
    (further from TCA)."""
    if not hours.size:
        return "top right"
    nearer = decision_point_hours - float(hours.min())
    further = float(hours.max()) - decision_point_hours
    return "top left" if further > nearer else "top right"


def trend(fig: TrendFigure, *, decision_point_hours: float | None = None) -> go.Figure:
    """An event's Pc against the hours to TCA (TCA on the right), a point
    per observation coloured by its risk level (hollow below the source's
    floor), with the threshold lines.

    The legend names each level, and a maximum Pc or one below the floor
    as such; hovering a point gives its CDM. With ``decision_point_hours``
    (the decision point, hours before TCA), a dotted vertical line marks
    it, as the risk report draws it, its label on the side with more of
    the points' span (the x axis is reversed: the right is nearer TCA), so
    that it stays over the plot.
    """
    drawing = go.Figure()
    if fig.fit is not None:
        hours, values = fig.fit
        drawing.add_trace(
            go.Scatter(
                x=hours,
                y=values,
                mode="lines",
                name="trend",
                line={"color": FIT_COLOUR, "width": 1},
                hoverinfo="skip",
            )
        )
    groups: dict[tuple[RiskLevel, bool, bool], list[int]] = {}
    for index, key in enumerate(
        zip(fig.levels, fig.below_floor, fig.upper_bound, strict=True)
    ):
        groups.setdefault(key, []).append(index)
    for (level, floored, bound), indices in groups.items():
        name = RISK_LABELS[level]
        if floored:
            name += " (below floor)"
        elif bound:
            name += " (max-Pc)"
        drawing.add_trace(
            go.Scatter(
                x=fig.hours_to_tca[indices],
                y=fig.pc[indices],
                mode="markers",
                name=name,
                text=[fig.labels[i] for i in indices],
                marker={
                    "color": RISK_COLOURS[level],
                    "size": 10,
                    "symbol": "circle-open" if floored else "circle",
                    "line": {"color": RISK_COLOURS[level], "width": 2},
                },
                hovertemplate="%{text}<br>%{x:.1f} h to TCA<br>Pc %{y:.2e}",
            )
        )
    for value, level in reversed(fig.thresholds):
        drawing.add_hline(
            y=value,
            line={"color": RISK_COLOURS[level], "dash": "dash", "width": 1},
            annotation_text=f"{_THRESHOLD_LABELS[level]} {value:.0e}",
            annotation_position="top left",
        )
    if decision_point_hours is not None:
        drawing.add_vline(
            x=decision_point_hours,
            line={"color": FIT_COLOUR, "dash": "dot", "width": 1},
            annotation_text="decision point",
            annotation_position=_decision_side(fig.hours_to_tca, decision_point_hours),
        )
    drawing.update_layout(
        xaxis={"title": {"text": "hours to TCA"}, "autorange": "reversed"},
        yaxis={"title": {"text": "Pc"}, "type": "log", "exponentformat": "power"},
        legend={"title": {"text": "risk"}},
    )
    return drawing


def encounter(fig: EncounterFigure, *, covariance: bool = True) -> go.Figure:
    """The encounter plane: the 1, 2 and 3 sigma ellipses of the combined
    covariance around the miss, and the hard-body circle around the
    primary, on equal axes [m].

    Without ``covariance`` (a maximum Pc, found without one), only the miss
    and the hard-body circle are drawn.
    """
    drawing = go.Figure()
    ellipses = zip(SIGMAS, fig.ellipses, (SERIES[4], SERIES[1], SERIES[6]), strict=True)
    for sigma, ellipse, colour in ellipses if covariance else ():
        drawing.add_trace(
            go.Scatter(
                x=ellipse[:, 0],
                y=ellipse[:, 1],
                mode="lines",
                name=f"{sigma:g} sigma",
                line={"color": colour, "width": 2},
            )
        )
    drawing.add_trace(
        go.Scatter(
            x=fig.hbr_circle[:, 0],
            y=fig.hbr_circle[:, 1],
            mode="lines",
            name="hard-body radius",
            fill="toself",
            line={"color": SERIES[5], "width": 2},
        )
    )
    drawing.add_trace(
        go.Scatter(
            x=[fig.miss[0]],
            y=[fig.miss[1]],
            mode="markers",
            name="miss",
            marker={"color": SERIES[0], "size": 10, "symbol": "x"},
        )
    )
    drawing.update_layout(
        xaxis={"title": {"text": "x [m]"}},
        yaxis={"title": {"text": "z [m]"}, "scaleanchor": "x", "scaleratio": 1},
    )
    return drawing


def geometry(fig: GeometryFigure, *, covariance: bool = True) -> go.Figure:
    """The relative geometry at TCA in the primary's RTN frame [m]: the
    secondary's path, its position at TCA, the 3 sigma ellipsoid around it
    and the hard-body sphere around the primary.

    Without ``covariance`` (a maximum Pc, found without one), the ellipsoid
    is left out.
    """
    path = fig.path
    drawing = go.Figure()
    drawing.add_trace(
        go.Scatter3d(
            x=path[:, 0],
            y=path[:, 1],
            z=path[:, 2],
            mode="lines",
            name="secondary's path",
            line={"color": SERIES[4], "width": 4},
        )
    )
    drawing.add_trace(
        go.Scatter3d(
            x=[fig.at_tca[0]],
            y=[fig.at_tca[1]],
            z=[fig.at_tca[2]],
            mode="markers",
            name="secondary at TCA",
            marker={"color": SERIES[0], "size": 5},
        )
    )
    surfaces = [(fig.hbr_sphere, "hard-body sphere", SERIES[5])]
    if covariance:
        surfaces.insert(0, (fig.ellipsoid, "3 sigma ellipsoid", SERIES[1]))
    for (x, y, z), name, colour in surfaces:
        drawing.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                name=name,
                opacity=0.35,
                showscale=False,
                showlegend=True,
                colorscale=((0.0, colour), (1.0, colour)),
            )
        )
    r, t, n = fig.axes
    drawing.update_layout(
        scene={
            "xaxis": {"title": {"text": r}},
            "yaxis": {"title": {"text": t}},
            "zaxis": {"title": {"text": n}},
            "aspectmode": "data",
        }
    )
    return drawing


def _crosses(panel: GridPanel, target: float) -> bool:
    """Whether the panel's Pc pass the target: the contour is drawn then."""
    values = panel.log_pc
    known = values[np.isfinite(values)]
    return bool(
        min(values.shape) > 1 and known.size and known.min() < target < known.max()
    )


def grid(fig: GridFigure) -> go.Figure:
    """The heatmap of a plan: log10 Pc over |Δv| (retrograde on the left,
    mirrored) and the hours before TCA, the target contour and the smallest
    burn meeting it in each direction.

    The colour scale runs from :data:`~keepout.figures.LOWEST_LOG_PC` (lower
    Pc share its lightest colour) to the figure's top; cells without a Pc
    are left grey (:data:`NO_PC`).
    """
    drawing = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        subplot_titles=[str(panel.direction) for panel in fig.directions],
    )
    last = len(fig.directions)
    for column, panel in enumerate(fig.directions, start=1):
        first = column == 1
        drawing.add_trace(
            go.Heatmap(
                x=panel.delta_v,
                y=panel.hours_before_tca,
                z=panel.log_pc,
                zmin=LOWEST_LOG_PC,
                zmax=fig.top_log_pc,
                colorscale=_COLOURSCALE,
                showscale=column == last,
                colorbar={"title": {"text": "log10 Pc"}},
                name=str(panel.direction),
                hovertemplate=(
                    "|Δv| %{x:.3g} m/s<br>%{y:.1f} h before TCA<br>"
                    "log10 Pc %{z:.2f}<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )
        if _crosses(panel, fig.target_log_pc):
            drawing.add_trace(
                go.Contour(
                    x=panel.delta_v_centres,
                    y=panel.hour_centres,
                    z=panel.log_pc,
                    contours={
                        "start": fig.target_log_pc,
                        "end": fig.target_log_pc,
                        "size": 1.0,
                        "coloring": "none",
                    },
                    line={"color": TARGET_COLOUR, "width": 2},
                    showscale=False,
                    name=f"target Pc {fig.target_pc:.0e}",
                    showlegend=first,
                    legendgroup="target",
                    hoverinfo="skip",
                ),
                row=1,
                col=column,
            )
        if panel.minimum is not None:
            hours, delta_v = panel.minimum
            drawing.add_trace(
                go.Scatter(
                    x=[delta_v],
                    y=[hours],
                    mode="markers",
                    name="smallest burn meeting it",
                    marker={
                        "color": TARGET_COLOUR,
                        "size": 12,
                        "line": {"color": "#ffffff", "width": 1.5},
                    },
                    showlegend=first,
                    legendgroup="minimum",
                ),
                row=1,
                col=column,
            )
    if any(np.isnan(panel.log_pc).any() for panel in fig.directions):
        drawing.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="no Pc",
                marker={"color": NO_PC, "size": 12, "symbol": "square"},
                showlegend=True,
            )
        )
    drawing.update_xaxes(type="log", title={"text": "|Δv| [m/s]"})
    drawing.update_xaxes(autorange="reversed", row=1, col=1)
    drawing.update_yaxes(title={"text": "burn time before TCA (h)"}, row=1, col=1)
    drawing.update_layout(plot_bgcolor=NO_PC, legend={"orientation": "h", "y": -0.2})
    return drawing
