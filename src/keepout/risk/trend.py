"""The trend of an event's Pc over its sequence of messages."""

from __future__ import annotations

import enum
import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from keepout._types import require_positive
from keepout.risk.observation import PC_FLOOR, PcObservation
from keepout.risk.thresholds import RiskLevel, Thresholds
from keepout.units import DAY

__all__ = [
    "Direction",
    "Trend",
    "TrendConfig",
    "TrendPoint",
    "analyse_trend",
]


class Direction(enum.StrEnum):
    """Where the Pc is heading: the sign of the fitted slope of log10 Pc."""

    RISING = "rising"
    FALLING = "falling"
    STEADY = "steady"
    UNDETERMINED = "undetermined"
    """Fewer than two observations to fit, or all fitted observations at one
    time."""


@dataclass(frozen=True, slots=True)
class TrendConfig:
    """How the direction of a trend is decided.

    The direction comes from a least-squares line through log10 Pc against
    time over the last ``window`` observations with a Pc. The defaults are a
    heuristic, not a standard.

    Parameters
    ----------
    window
        Number of latest observations fitted (at least 2).
    steady_within
        Largest slope [decades of Pc per day] still called :attr:`Direction.STEADY`.

    Raises
    ------
    ValueError
        If ``window`` is below 2 or ``steady_within`` is not finite and
        positive.
    """

    window: int = 3
    steady_within: float = 0.5

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError(f"window must be at least 2, got {self.window}")
        require_positive(self.steady_within, "steady_within")


@dataclass(frozen=True, eq=False, slots=True)
class TrendPoint:
    """One observation with a Pc, before TCA, as the trend sees it.

    Parameters
    ----------
    observation
        The observation.
    value
        The value compared with the thresholds (see
        :meth:`Thresholds.value_of`).
    level
        Its risk level.
    """

    observation: PcObservation
    value: float
    level: RiskLevel

    @property
    def time_to_tca(self) -> float:
        """Seconds from the observation to TCA."""
        return self.observation.time_to_tca


def _order(observation: PcObservation) -> tuple[object, ...]:
    # Oldest first; among simultaneous ones a failed, higher or doubtful Pc
    # comes later, so it counts as the latest (the cautious choice). "Higher"
    # ranks by the larger of the estimate and the interval's upper end first,
    # so the order is cautious under either threshold basis; "doubtful" is an
    # upper bound or a Pc with diagnostics. The remaining fields only make the
    # order total, so the result does not depend on the input order.
    pc = observation.pc or 0.0
    ci_high = observation.ci_high or 0.0
    return (
        -observation.time_to_tca,
        not observation.ok,
        max(pc, ci_high),
        pc,
        ci_high,
        observation.kind == "upper_bound" or bool(observation.diagnostics),
        observation.kind,
        observation.diagnostics,
        observation.message_id or "",
        observation.error or "",
        observation.source,
        observation.reported_pc is not None,
        observation.reported_pc or 0.0,
    )


def _chronological(
    observations: Iterable[PcObservation],
) -> tuple[PcObservation, ...]:
    """The observations from the oldest to the latest (by time to TCA)."""
    return tuple(sorted(observations, key=_order))


def _label(observation: PcObservation) -> str:
    when = f"{observation.time_to_tca / DAY:.2f} d before TCA"
    if observation.message_id is None:
        return when
    return f"{observation.message_id} ({when})"


def _slope(points: Sequence[TrendPoint]) -> float | None:
    """Least-squares slope of log10 Pc per day; ``None`` if all times agree."""
    days = [-point.time_to_tca / DAY for point in points]
    logs = [math.log10(max(point.value, PC_FLOOR)) for point in points]
    try:
        return statistics.linear_regression(days, logs).slope
    except statistics.StatisticsError:  # all days equal
        return None


@dataclass(frozen=True, eq=False, slots=True)
class Trend:
    """The Pc of an event over time.

    Parameters
    ----------
    observations
        Every observation, from the oldest to the latest.
    points
        The observations with a Pc issued before TCA, oldest first.
    direction
        Where the Pc is heading; see :data:`Direction`.
    slope
        Fitted slope [decades of Pc per day], if there is one.
    notes
        What the trend left out or could not do, in words.
    thresholds
        The thresholds of the points' levels.
    """

    observations: tuple[PcObservation, ...]
    points: tuple[TrendPoint, ...]
    direction: Direction
    slope: float | None
    notes: tuple[str, ...]
    thresholds: Thresholds

    @property
    def latest(self) -> TrendPoint | None:
        """The latest observation with a Pc, before TCA."""
        return self.points[-1] if self.points else None

    @property
    def peak(self) -> TrendPoint | None:
        """The observation with the highest value (the latest of equal ones)."""
        if not self.points:
            return None
        return max(reversed(self.points), key=lambda point: point.value)

    @property
    def latest_observation(self) -> PcObservation | None:
        """The latest observation before TCA, with a Pc or not."""
        before = [o for o in self.observations if o.time_to_tca >= 0.0]
        return before[-1] if before else None


def _points(
    ordered: Sequence[PcObservation], limits: Thresholds
) -> tuple[list[TrendPoint], list[str]]:
    """The points of ``ordered`` observations, and notes on those left out."""
    notes: list[str] = []
    after_tca = 0
    points: list[TrendPoint] = []
    for observation in ordered:
        if observation.time_to_tca < 0.0:
            after_tca += 1
        elif observation.pc is None:
            notes.append(f"no Pc for {_label(observation)}: {observation.error}")
        else:
            value = limits.value_of(observation)
            points.append(TrendPoint(observation, value, limits.level(value)))
    if after_tca:
        notes.insert(0, f"{after_tca} observation(s) issued after TCA left out")
    return points, notes


def _direction(
    fitted: Sequence[TrendPoint], settings: TrendConfig
) -> tuple[Direction, float | None, str | None]:
    """The direction and slope of the ``fitted`` points; a note if undetermined."""
    if len(fitted) < 2:
        note = "fewer than 2 observations with a Pc before TCA"
        return Direction.UNDETERMINED, None, f"{note}: direction undetermined"
    slope = _slope(fitted)
    if slope is None:
        note = "observations at one time"
        return Direction.UNDETERMINED, None, f"{note}: direction undetermined"
    if slope > settings.steady_within:
        return Direction.RISING, slope, None
    if slope < -settings.steady_within:
        return Direction.FALLING, slope, None
    return Direction.STEADY, slope, None


def analyse_trend(
    observations: Iterable[PcObservation],
    *,
    thresholds: Thresholds | None = None,
    config: TrendConfig | None = None,
) -> Trend:
    """The trend of an event's observations, in any order.

    Observations issued after TCA and observations without a Pc are left out
    of the points, with a note.

    Parameters
    ----------
    observations
        The event's observations.
    thresholds
        Thresholds of the points' levels; :class:`Thresholds` defaults if
        ``None``.
    config
        How the direction is decided; :class:`TrendConfig` defaults if
        ``None``.

    Raises
    ------
    ValueError
        If there are no observations.
    """
    limits = thresholds if thresholds is not None else Thresholds()
    settings = config if config is not None else TrendConfig()
    ordered = _chronological(observations)
    if not ordered:
        raise ValueError("a trend needs at least one observation")
    points, notes = _points(ordered, limits)
    fitted = points[-settings.window :]
    if any(point.value < PC_FLOOR for point in fitted):
        notes.append(f"Pc below {PC_FLOOR:g} fitted at {PC_FLOOR:g}")
    direction, slope, note = _direction(fitted, settings)
    if note is not None:
        notes.append(note)
    return Trend(ordered, tuple(points), direction, slope, tuple(notes), limits)
