"""The manoeuvre report's heatmap: a plan's Pc over burn size and time.

``tools/maneuver_report.py`` draws it for the page and the README from a
:class:`keepout.figures.GridFigure`; this module only draws.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from keepout.figures import (
    INK_COLOUR,
    LOWEST_LOG_PC,
    MUTED_COLOUR,
    NO_PC,
    RULE_COLOUR,
    SEQUENTIAL,
    TARGET_COLOUR,
)

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.colors import Colormap

    from keepout.figures import GridFigure, GridPanel

FACECOLOR: Final = "white"
"""The heatmap's background, as :func:`generated.save_svg` takes it."""

DPI: Final = 200
"""The resolution of the heatmap's rasterised cells."""


def delta_v_text(delta_v: float) -> str:
    """A signed Δv to 3 significant figures, trailing zeros kept: ``-0.00200``."""
    return f"{delta_v:+#.3g}"


def _mark(axes: Axes, panel: GridPanel) -> None:
    """The option of a panel: a ringed dot and its label."""
    if panel.minimum is None:
        return
    sign = panel.direction.sign
    lead, size = panel.minimum
    axes.scatter(
        [size],
        [lead],
        s=64,
        color=TARGET_COLOUR,
        edgecolors="white",
        linewidths=2.0,
        zorder=3,
    )
    axes.annotate(
        f"{delta_v_text(sign * size)} m/s\n{lead:.1f} h before TCA",
        (size, lead),
        xytext=(10 * sign, -10),
        textcoords="offset points",
        ha="left" if sign > 0 else "right",
        va="top",
        fontsize="small",
        color=INK_COLOUR,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none"},
    )


def _panel(
    axes: Axes, figure: GridFigure, panel: GridPanel, cmap: Colormap
) -> QuadMesh:
    """One direction's cells, the target contour and the option."""
    values = panel.log_pc
    mesh = axes.pcolormesh(
        panel.delta_v,
        panel.hours_before_tca,
        np.ma.masked_invalid(values),
        cmap=cmap,
        vmin=LOWEST_LOG_PC,
        vmax=figure.top_log_pc,
        shading="flat",
        edgecolors="none",
        rasterized=True,
    )
    if panel.crosses(figure.target_log_pc):
        axes.contour(
            panel.delta_v_centres,
            panel.hour_centres,
            np.ma.masked_invalid(values),
            levels=[figure.target_log_pc],
            colors=TARGET_COLOUR,
            linewidths=2.0,
            linestyles="solid",
        )
    _mark(axes, panel)
    axes.set_xscale("log")
    axes.set_title(panel.direction, color=INK_COLOUR)
    axes.set_xlabel("|Δv| along the velocity (m/s)", color=MUTED_COLOUR)
    axes.tick_params(colors=MUTED_COLOUR)
    for spine in axes.spines.values():
        spine.set_color(RULE_COLOUR)
    return mesh


def _legend(drawing: Figure, figure: GridFigure) -> None:
    handles: list[Artist] = [
        Line2D([], [], color=TARGET_COLOUR, linewidth=2.0),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=8,
            markerfacecolor=TARGET_COLOUR,
            markeredgecolor="white",
        ),
    ]
    labels = [f"target Pc {figure.target_pc:.0e}", "smallest burn meeting it"]
    if any(np.isnan(panel.log_pc).any() for panel in figure.directions):
        handles.append(Patch(facecolor=NO_PC))
        labels.append("no Pc (no minimum to trust)")
    drawing.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        fontsize="small",
        labelcolor=INK_COLOUR,
    )


def heatmap_figure(figure: GridFigure, title: str) -> Figure:
    """The heatmap of a plan: log10 Pc over |Δv| (retrograde mirrored to the
    left) and hours before TCA."""
    cmap = LinearSegmentedColormap.from_list("pc", SEQUENTIAL).with_extremes(bad=NO_PC)
    retrograde, prograde = figure.directions
    drawing = Figure(figsize=(9.0, 4.8))
    grid = drawing.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.05), wspace=0.12)
    left = drawing.add_subplot(grid[0, 0])
    right = drawing.add_subplot(grid[0, 1], sharey=left)
    _panel(left, figure, retrograde, cmap)
    mesh = _panel(right, figure, prograde, cmap)
    left.invert_xaxis()
    left.set_ylabel("burn time before TCA (h)", color=MUTED_COLOUR)
    right.tick_params(labelleft=False)
    colourbar = drawing.colorbar(
        mesh, cax=drawing.add_subplot(grid[0, 2]), extend="min"
    )
    colourbar.set_label("log10 Pc (Foster)", color=MUTED_COLOUR)
    colourbar.ax.axhline(figure.target_log_pc, color=TARGET_COLOUR, linewidth=2.0)
    colourbar.ax.tick_params(colors=MUTED_COLOUR)
    colourbar.outline.set_edgecolor(RULE_COLOUR)
    _legend(drawing, figure)
    drawing.suptitle(title, color=INK_COLOUR, fontsize=11)
    drawing.subplots_adjust(left=0.08, right=0.93, top=0.84, bottom=0.2)
    return drawing
