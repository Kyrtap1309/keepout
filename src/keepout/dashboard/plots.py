"""The figures of :mod:`keepout.figures` drawn with Plotly, as the reports
draw them with matplotlib: the trend of an event's Pc, the encounter plane,
the 3-D geometry at TCA and the heatmap of a manoeuvre plan.

Every trace is named, so the legend says in words what colour shows. A
threshold line has the colour of the level its label names (``green
below`` is green); a yellow point has a dark edge, to show on a light
background.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from keepout.figures import (
    FIT_COLOUR,
    LOWEST_LOG_PC,
    NO_PC,
    RISK_COLOURS,
    SEQUENTIAL,
    SERIES,
    SIGMAS,
    TARGET_COLOUR,
    THRESHOLD_LABELS,
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

_MARGIN: Final = {"t": 30, "b": 40, "l": 10, "r": 10}
"""The space around a plot [px]: room for a label above it, no more."""

_EDGE: Final = "#3c3c3c"
"""The edge of a yellow point."""

_LINE_LEVEL: Final = {RiskLevel.YELLOW: RiskLevel.GREEN, RiskLevel.RED: RiskLevel.RED}
"""The level whose colour a threshold line takes, by the level it starts:
the one its label names (:data:`~keepout.figures.THRESHOLD_LABELS`)."""

_COLOURSCALE: Final = tuple(
    (i / (len(SEQUENTIAL) - 1), colour) for i, colour in enumerate(SEQUENTIAL)
)
"""The heatmap's colours, light (low Pc) to dark, evenly spaced (the
stubs do not take a bare list of colours)."""


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


def _fit_trace(fig: TrendFigure) -> go.Scatter:
    """The line joining an event's Pc."""
    return go.Scatter(
        x=fig.hours_to_tca,
        y=fig.pc,
        mode="lines",
        name="trend",
        line={"color": FIT_COLOUR, "width": 1},
        hoverinfo="skip",
    )


def _point_traces(fig: TrendFigure) -> list[go.Scatter]:
    """The observations, a trace per risk level and kind of Pc (below the
    floor: hollow; maximum Pc), in the order each first appears."""
    groups: defaultdict[tuple[RiskLevel, bool, bool], list[int]] = defaultdict(list)
    for index, key in enumerate(
        zip(fig.levels, fig.below_floor, fig.upper_bound, strict=True)
    ):
        groups[key].append(index)
    traces: list[go.Scatter] = []
    for (level, floored, bound), indices in groups.items():
        name = level.key
        if floored:
            name += " (below floor)"
        elif bound:
            name += " (max-Pc)"
        traces.append(
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
                    "line": {"color": _edge(level, floored=floored), "width": 2},
                },
                hovertemplate="%{text}<br>%{x:.1f} h to TCA<br>Pc %{y:.2e}",
            )
        )
    return traces


def _edge(level: RiskLevel, *, floored: bool) -> str:
    """The edge of a point: its level's colour, but dark for a filled
    yellow one."""
    if level is RiskLevel.YELLOW and not floored:
        return _EDGE
    return RISK_COLOURS[level]


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
    if len(fig.pc) >= 2:
        drawing.add_trace(_fit_trace(fig))
    for trace in _point_traces(fig):
        drawing.add_trace(trace)
    for value, level in reversed(fig.thresholds):
        drawing.add_hline(
            y=value,
            line={
                "color": RISK_COLOURS[_LINE_LEVEL[level]],
                "dash": "dash",
                "width": 1.5,
            },
            annotation_text=f"{THRESHOLD_LABELS[level]} {value:.0e}",
            annotation_position="top left",
            # Plotly places a line's label at y itself, which a log axis
            # reads as 10**y: given as log10, it sits on its line.
            annotation={"y": math.log10(value)},
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
        margin=_MARGIN,
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
        margin=_MARGIN,
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
        },
        margin=_MARGIN,
    )
    return drawing


def _colour_bar(fig: GridFigure) -> dict[str, object]:
    """The colour bar: a tick per decade of Pc, and one at the target."""
    decades = range(math.ceil(LOWEST_LOG_PC), math.floor(fig.top_log_pc) + 1)
    # A decade too near the target would overlap its label.
    ticks = {
        float(d): f"{10.0**d:.0e}" for d in decades if abs(d - fig.target_log_pc) >= 0.5
    }
    ticks[fig.target_log_pc] = f"target {fig.target_pc:.0e}"
    return {
        "title": {"text": "Pc"},
        "tickvals": list(ticks),
        "ticktext": list(ticks.values()),
    }


def _heatmap(panel: GridPanel, fig: GridFigure, *, scale: bool) -> go.Heatmap:
    """One direction's cells; with ``scale``, the colour bar."""
    return go.Heatmap(
        x=panel.delta_v,
        y=panel.hours_before_tca,
        z=panel.log_pc,
        zmin=LOWEST_LOG_PC,
        zmax=fig.top_log_pc,
        colorscale=_COLOURSCALE,
        showscale=scale,
        colorbar=_colour_bar(fig),
        name=str(panel.direction),
        hovertemplate=(
            "|Δv| %{x:.3g} m/s<br>%{y:.1f} h before TCA<br>"
            "log10 Pc %{z:.2f}<extra></extra>"
        ),
    )


def _target_contour(panel: GridPanel, fig: GridFigure, *, legend: bool) -> go.Contour:
    """The target contour over one direction's cells."""
    return go.Contour(
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
        showlegend=legend,
        legendgroup="target",
        hoverinfo="skip",
    )


def _minimum_marker(minimum: tuple[float, float], *, legend: bool) -> go.Scatter:
    """The smallest burn meeting the target in one direction."""
    hours, delta_v = minimum
    return go.Scatter(
        x=[delta_v],
        y=[hours],
        mode="markers",
        name="smallest burn meeting it",
        marker={
            "color": TARGET_COLOUR,
            "size": 12,
            "line": {"color": "#ffffff", "width": 1.5},
        },
        showlegend=legend,
        legendgroup="minimum",
    )


def _no_pc_key() -> go.Scatter:
    """The legend's key of the cells without a Pc (an empty trace)."""
    return go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        name="no Pc",
        marker={"color": NO_PC, "size": 12, "symbol": "square"},
        showlegend=True,
    )


def grid(fig: GridFigure) -> go.Figure:
    """The heatmap of a plan: log10 Pc over |Δv| (retrograde on the left,
    mirrored) and the hours before TCA, the target contour and the smallest
    burn meeting it in each direction.

    The colour scale runs from :data:`~keepout.figures.LOWEST_LOG_PC` (lower
    Pc share its lightest colour) to the figure's top; cells without a Pc
    are left grey (:data:`~keepout.figures.NO_PC`).
    """
    drawing = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        # Room between the panels for their end ticks on a narrow page.
        horizontal_spacing=0.12,
        subplot_titles=[str(panel.direction) for panel in fig.directions],
    )
    last = len(fig.directions)
    # The legend names the contour and the marker once, on the first panel
    # that draws them (a direction may have neither).
    contoured = marked = False
    for column, panel in enumerate(fig.directions, start=1):
        traces: list[go.Heatmap | go.Contour | go.Scatter] = [
            _heatmap(panel, fig, scale=column == last)
        ]
        if panel.crosses(fig.target_log_pc):
            traces.append(_target_contour(panel, fig, legend=not contoured))
            contoured = True
        if panel.minimum is not None:
            traces.append(_minimum_marker(panel.minimum, legend=not marked))
            marked = True
        for trace in traces:
            drawing.add_trace(trace, row=1, col=column)
    if any(np.isnan(panel.log_pc).any() for panel in fig.directions):
        drawing.add_trace(_no_pc_key())
    # A tick per decade: the log axis's 2 and 5 overlap on a narrow page.
    drawing.update_xaxes(
        type="log", dtick=1, automargin=True, title={"text": "|Δv| [m/s]"}
    )
    drawing.update_xaxes(autorange="reversed", row=1, col=1)
    drawing.update_yaxes(title={"text": "burn time before TCA (h)"}, row=1, col=1)
    drawing.update_layout(
        plot_bgcolor=NO_PC,
        # Under the axis titles, however tall the ticks: at the bottom of
        # the figure, in a margin of its own.
        legend={"orientation": "h", "yref": "container", "y": 0, "yanchor": "bottom"},
        margin={**_MARGIN, "t": 40, "b": 110},
    )
    return drawing
