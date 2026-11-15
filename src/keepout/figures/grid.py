"""The heatmap of a manoeuvre plan: log10 Pc over burn time and |Δv|."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from keepout._types import FloatArray, read_only
from keepout.maneuver import BurnDirection, ManeuverOutcome
from keepout.units import HOUR

if TYPE_CHECKING:
    from keepout.maneuver import ManeuverGrid, ManeuverOption, ManeuverPlan

__all__ = ["LOWEST_LOG_PC", "GridFigure", "GridPanel", "grid_figure"]

LOWEST_LOG_PC: Final = -12.0
"""The heatmap's colour scale stops here: lower Pc share its lightest colour."""


@dataclass(frozen=True, eq=False, slots=True)
class GridPanel:
    """The cells of one burn direction.

    Parameters
    ----------
    direction
        Prograde or retrograde.
    hours_before_tca
        Edges of the cells along the burn time [h before TCA], ascending:
        one more than the burn times.
    delta_v
        Edges of the cells along |Δv| [m/s], ascending and geometric: one
        more than the burn sizes.
    hour_centres
        The burn times [h before TCA], ascending: the rows' centres.
    delta_v_centres
        The burn sizes |Δv| [m/s], ascending: the columns' centres.
    log_pc
        log10 Pc of each cell, a row per burn time and a column per |Δv|;
        ``LOWEST_LOG_PC - 1`` where the Pc is 0 (below the colour scale,
        which stops at :data:`LOWEST_LOG_PC`); NaN where a cell has no Pc.
    minimum
        The smallest burn meeting the target, as hours before TCA and |Δv|
        [m/s]; ``None`` if the direction has none.
    """

    direction: BurnDirection
    hours_before_tca: FloatArray
    delta_v: FloatArray
    hour_centres: FloatArray
    delta_v_centres: FloatArray
    log_pc: FloatArray
    minimum: tuple[float, float] | None

    def crosses(self, target_log_pc: float) -> bool:
        """Whether the cells' Pc pass a target, so that its contour can be
        drawn: some known Pc on either side of it, on at least two rows and
        columns. The contour stops at cells without a Pc.

        Parameters
        ----------
        target_log_pc
            log10 of the target Pc.

        Returns
        -------
        bool
            ``True`` if a contour of the target crosses the cells.
        """
        values = self.log_pc
        known = values[np.isfinite(values)]
        return bool(
            min(values.shape) > 1
            and known.size
            and known.min() < target_log_pc < known.max()
        )


@dataclass(frozen=True, eq=False, slots=True)
class GridFigure:
    """The heatmap of a plan.

    Parameters
    ----------
    directions
        A panel per direction: retrograde (drawn mirrored, on the left), then
        prograde.
    target_pc
        The plan's target Pc.
    target_log_pc
        log10 of the target Pc, for the contour.
    top_log_pc
        The top of the colour scale: the highest log10 Pc rounded up (-1
        without any Pc), and at least ``LOWEST_LOG_PC + 1``.
    """

    directions: tuple[GridPanel, ...]
    target_pc: float
    target_log_pc: float
    top_log_pc: float


def _edges(centres: FloatArray, *, geometric: bool) -> FloatArray:
    """Cell edges around ascending ``centres``: midway between them, in log
    if ``geometric``, and as far beyond the ends."""
    values = np.log(centres) if geometric else np.asarray(centres, dtype=np.float64)
    if len(values) == 1:
        inner: FloatArray = np.empty(0)
        first, last = values[0] - 0.5, values[0] + 0.5
    else:
        inner = (values[1:] + values[:-1]) / 2.0
        first = 2.0 * values[0] - inner[0]
        last = 2.0 * values[-1] - inner[-1]
    edges: FloatArray = np.concatenate(([first], inner, [last]))
    return np.exp(edges) if geometric else edges


def _panel(
    delta_v: FloatArray,
    direction: BurnDirection,
    option: ManeuverOption | None,
    hours: FloatArray,
    log_pc: FloatArray,
) -> GridPanel:
    columns = np.flatnonzero(np.sign(delta_v) == direction.sign)
    if columns.size == 0:
        raise ValueError(f"the plan has no {direction} burn")
    columns = columns[np.argsort(np.abs(delta_v[columns]))]
    magnitudes = np.abs(delta_v[columns])
    values = np.ascontiguousarray(log_pc[:, columns])
    hour_edges = _edges(hours, geometric=False)
    delta_v_edges = _edges(magnitudes, geometric=True)
    centres = hours.copy()
    read_only(values, hour_edges, delta_v_edges, centres, magnitudes)
    minimum = None if option is None else (option.burn_lead / HOUR, abs(option.delta_v))
    return GridPanel(
        direction, hour_edges, delta_v_edges, centres, magnitudes, values, minimum
    )


def grid_figure(plan: ManeuverPlan | ManeuverOutcome) -> GridFigure:
    """The heatmap of a manoeuvre plan.

    Parameters
    ----------
    plan
        The plan of :func:`~keepout.maneuver.plan_maneuver`, or the outcome
        of :func:`~keepout.maneuver.plan_event` on the grid (as the store
        caches it): its grid and its burns.

    Returns
    -------
    GridFigure
        A panel per direction, the target and the colour scale.

    Raises
    ------
    ValueError
        If the plan has no burn in a direction (a grid of
        :func:`~keepout.maneuver.plan_maneuver` always has both), or the
        outcome has no grid (its one burn was given, not planned).
    """
    cells: ManeuverPlan | ManeuverGrid
    if isinstance(plan, ManeuverOutcome):
        if plan.grid is None:
            raise ValueError("the outcome has no grid: its burn was given")
        cells = plan.grid
        found = {burn.direction: burn.option for burn in plan.burns}
        options = {direction: found.get(direction) for direction in BurnDirection}
    else:
        cells = plan
        options = {direction: plan.option(direction) for direction in BurnDirection}
    hours = cells.burn_leads[::-1] / HOUR
    with np.errstate(divide="ignore"):
        log_pc = np.log10(cells.pc[::-1])
    # A Pc of 0 is below the scale, like any Pc under LOWEST_LOG_PC.
    log_pc = np.where(np.isneginf(log_pc), LOWEST_LOG_PC - 1.0, log_pc)
    top = float(np.ceil(np.nanmax(log_pc))) if np.isfinite(log_pc).any() else -1.0
    return GridFigure(
        tuple(
            _panel(cells.delta_v, direction, options[direction], hours, log_pc)
            for direction in (BurnDirection.RETROGRADE, BurnDirection.PROGRADE)
        ),
        plan.target_pc,
        float(np.log10(plan.target_pc)),
        max(top, LOWEST_LOG_PC + 1.0),
    )
