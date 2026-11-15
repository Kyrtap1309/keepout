"""The trend chart of an event: its Pc against the time to TCA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from keepout._types import FloatArray, read_only
from keepout.risk import PC_FLOOR, RiskLevel
from keepout.units import HOUR

if TYPE_CHECKING:
    from keepout.risk import Assessment, Thresholds

__all__ = ["TrendFigure", "trend_figure"]


@dataclass(frozen=True, eq=False, slots=True)
class TrendFigure:
    """What a trend chart draws: one point per observation with a Pc before
    TCA, oldest first.

    Parameters
    ----------
    hours_to_tca
        Hours from each observation to TCA, descending.
    pc
        The value of each observation compared with the thresholds (its Pc,
        or the upper end of its confidence interval on the conservative
        basis), floored at :data:`~keepout.risk.PC_FLOOR` for a log scale.
    levels
        The risk level of each point.
    upper_bound
        Whether each point is a maximum Pc (an upper bound).
    below_floor
        Whether each point's Pc is below the floor its source reports (a
        ``below_floor`` observation): the value is an upper bound on it.
    labels
        Each point's message identifier; empty when it has none.
    thresholds
        The threshold lines: the Pc from which each level starts, yellow
        then red.
    fit
        The trend line as hours and Pc (the points joined in order);
        ``None`` with fewer than two points.
    """

    hours_to_tca: FloatArray
    pc: FloatArray
    levels: tuple[RiskLevel, ...]
    upper_bound: tuple[bool, ...]
    below_floor: tuple[bool, ...]
    labels: tuple[str, ...]
    thresholds: tuple[tuple[float, RiskLevel], ...]
    fit: tuple[FloatArray, FloatArray] | None


def trend_figure(assessment: Assessment, thresholds: Thresholds) -> TrendFigure:
    """The trend chart of an assessed event.

    Parameters
    ----------
    assessment
        The event's assessment; its trend's points are drawn.
    thresholds
        The thresholds of the assessment, drawn as lines.

    Returns
    -------
    TrendFigure
        The points, the threshold lines and the trend line.

    Raises
    ------
    ValueError
        If ``thresholds`` are not the assessment's: the levels of the points
        would not match the lines.
    """
    if thresholds != assessment.thresholds:
        raise ValueError(
            f"thresholds {thresholds} are not the assessment's {assessment.thresholds}"
        )
    points = assessment.trend.points
    hours = np.array([point.time_to_tca / HOUR for point in points], dtype=np.float64)
    pc = np.array([max(point.value, PC_FLOOR) for point in points], dtype=np.float64)
    read_only(hours, pc)
    return TrendFigure(
        hours_to_tca=hours,
        pc=pc,
        levels=tuple(point.level for point in points),
        upper_bound=tuple(point.observation.kind == "upper_bound" for point in points),
        below_floor=tuple(point.observation.kind == "below_floor" for point in points),
        labels=tuple(point.observation.message_id or "" for point in points),
        thresholds=(
            (thresholds.green_below, RiskLevel.YELLOW),
            (thresholds.red_from, RiskLevel.RED),
        ),
        fit=(hours, pc) if len(points) >= 2 else None,
    )
