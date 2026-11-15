"""What to do about a conjunction event: ignore it, monitor it or manoeuvre."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from keepout._defaults import DECISION_POINT
from keepout._text import days_text, pc_text
from keepout._types import OrderedEnum
from keepout._validation import require_positive
from keepout.risk.observation import PC_FLOOR, PcObservation
from keepout.risk.thresholds import DEFAULT_THRESHOLDS, RiskLevel, Thresholds
from keepout.risk.trend import (
    DEFAULT_TREND_CONFIG,
    Direction,
    Trend,
    TrendConfig,
    TrendPoint,
    analyse_trend,
)
from keepout.units import DAY

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "DEFAULT_DECISION_CONFIG",
    "Action",
    "Assessment",
    "Decision",
    "DecisionConfig",
    "assess",
    "decide",
]


class Action(OrderedEnum):
    """Recommended action, ordered from the mildest to the most drastic."""

    IGNORE = 0
    """Nothing to do."""
    MONITOR = 1
    """Watch the next messages."""
    PREPARE_MANEUVER = 2
    """Watch the next messages and prepare an avoidance manoeuvre (a red
    event before the decision point, or a red event whose latest Pc is
    missing)."""
    MANEUVER = 3
    """Plan and execute an avoidance manoeuvre."""

    @property
    def label(self) -> str:
        """The action in words, as keepout prints it: ``prepare maneuver``."""
        return self.key.replace("_", " ")


@dataclass(frozen=True, slots=True)
class DecisionConfig:
    """When a red event becomes a manoeuvre.

    Parameters
    ----------
    decision_point
        Seconds before TCA from which a red event is manoeuvred; earlier, it
        is monitored while a manoeuvre is prepared. Default: two days, which
        leaves time to plan the manoeuvre and screen the new orbit.

    Raises
    ------
    ValueError
        If ``decision_point`` is not finite and positive.
    """

    decision_point: float = DECISION_POINT

    def __post_init__(self) -> None:
        require_positive(self.decision_point, "decision_point")


DEFAULT_DECISION_CONFIG: Final = DecisionConfig()
"""The default decision point."""


@dataclass(frozen=True, eq=False, slots=True)
class Decision:
    """A recommended action and why.

    Parameters
    ----------
    action
        The recommended action.
    reasons
        Why, in words: the risk level, the trend, and every concern about
        the data.
    """

    action: Action
    reasons: tuple[str, ...]


def _describe_latest(point: TrendPoint) -> str:
    """The latest point's Pc, level and time."""
    return _with_qualifiers(
        f"latest Pc {pc_text(point.value)} is {point.level.key}", point
    )


def _describe_before_failure(point: TrendPoint) -> str:
    """The point's Pc, level and time, as the latest observation with a Pc,
    followed by one without."""
    text = (
        f"the latest observation with a Pc was {point.level.key}: "
        f"Pc {pc_text(point.value)}"
    )
    return _with_qualifiers(text, point)


def _with_qualifiers(text: str, point: TrendPoint) -> str:
    """``text`` about the point's Pc, followed by what kind of value it is
    and its time before TCA."""
    observation = point.observation
    if point.value != observation.pc:
        text += " (upper end of the Monte Carlo confidence interval)"
    if observation.kind == "upper_bound":
        text += " (an upper bound: a covariance is unknown)"
    elif observation.kind == "below_floor":
        text += " (at most this: below the source's reporting floor)"
    return f"{text}, {days_text(point.time_to_tca)} before TCA"


def _concerns(point: TrendPoint) -> list[str]:
    """What makes the latest Pc less than a reliable estimate."""
    observation = point.observation
    concerns = [f"diagnostic: {note}" for note in observation.diagnostics]
    if observation.kind == "upper_bound":
        concerns.append("the Pc is an upper bound, not an estimate")
    return concerns


def _reaches(point: TrendPoint, slope: float, level: float) -> bool:
    """Whether the Pc, rising at ``slope`` decades per day, reaches ``level``
    by TCA."""
    at_tca = math.log10(max(point.value, PC_FLOOR)) + slope * point.time_to_tca / DAY
    return at_tca >= math.log10(level)


def _trend_reason(trend: Trend) -> str:
    if trend.slope is None:
        return f"trend {trend.direction}"
    return f"trend {trend.direction} ({trend.slope:+.2f} decades per day)"


def _red(
    time_to_tca: float,
    config: DecisionConfig,
    reasons: Sequence[str],
    concerns: Sequence[str],
) -> Decision:
    """The decision on a red event, judged ``time_to_tca`` seconds before TCA."""
    decision_point_text = (
        f"the decision point ({days_text(config.decision_point)} before TCA)"
    )
    if time_to_tca <= config.decision_point:
        within = f"within {decision_point_text}"
        return Decision(Action.MANEUVER, (*reasons, within, *concerns))
    timing = f"before {decision_point_text}: prepare a maneuver"
    return Decision(Action.PREPARE_MANEUVER, (*reasons, timing, *concerns))


def decide(
    trend: Trend, *, config: DecisionConfig = DEFAULT_DECISION_CONFIG
) -> Decision:
    """The recommended action for an event's trend.

    Rules, on the latest observation before TCA:

    - no Pc (it could not be computed): the decision follows the latest
      observation with a Pc. If that one was red, the red rule below applies,
      judged at the time of the observation without a Pc; otherwise (no
      observation with a Pc, or it was not red) ``MONITOR``;
    - red: ``MANEUVER`` within the decision point, else
      ``PREPARE_MANEUVER``;
    - yellow: ``MONITOR``;
    - green: ``IGNORE``, unless the Pc is an upper bound, has diagnostics
      (covariance repairs, questionable assumptions) or is rising fast
      enough to reach the yellow threshold of the trend's thresholds by TCA
      (extrapolating the fitted slope): then ``MONITOR``. A doubtful green
      is never ignored.

    With no observation before TCA the event is past; it is ``MONITOR``-ed
    so the outcome is not lost.

    Parameters
    ----------
    trend
        The event's trend, from :func:`~keepout.risk.analyse_trend`.
    config
        How long before TCA a red event is to be manoeuvred; by default two
        days (see :class:`DecisionConfig`).

    Returns
    -------
    Decision
        The recommended action and the reasons for it.
    """
    latest = trend.latest_observation
    if latest is None:
        return Decision(
            Action.MONITOR, ("no observation before TCA: nothing to decide on",)
        )
    point = trend.latest
    # With a Pc, the latest observation is the latest point.
    if latest.ok and point is not None:
        return _decide_on(point, trend, config)
    reasons = [f"latest observation has no Pc: {latest.error}"]
    if point is None or point.level is not RiskLevel.RED:
        return Decision(Action.MONITOR, tuple(reasons))
    reasons += [_describe_before_failure(point), _trend_reason(trend)]
    return _red(latest.time_to_tca, config, reasons, _concerns(point))


def _decide_on(point: TrendPoint, trend: Trend, config: DecisionConfig) -> Decision:
    """:func:`decide` when the latest observation, ``point``, has a Pc."""
    thresholds = trend.thresholds
    reasons = [_describe_latest(point), _trend_reason(trend)]
    concerns = _concerns(point)
    if point.level is RiskLevel.RED:
        return _red(point.time_to_tca, config, reasons, concerns)
    if point.level is RiskLevel.YELLOW:
        return Decision(Action.MONITOR, (*reasons, *concerns))
    if trend.direction is Direction.RISING and trend.slope is not None:
        reaches = _reaches(point, trend.slope, thresholds.green_below)
        if reaches:
            concerns.insert(
                0,
                f"rising fast enough to reach {thresholds.green_below:g} before TCA",
            )
        else:
            reasons.append(
                f"rising, but at this rate below {thresholds.green_below:g} at TCA"
            )
    action = Action.MONITOR if concerns else Action.IGNORE
    return Decision(action, (*reasons, *concerns))


@dataclass(frozen=True, eq=False, slots=True)
class Assessment:
    """An event's trend and the recommended action.

    Parameters
    ----------
    trend
        The trend of the event's observations.
    decision
        The recommended action.
    """

    trend: Trend
    decision: Decision

    @property
    def thresholds(self) -> Thresholds:
        """The thresholds used."""
        return self.trend.thresholds

    @property
    def level(self) -> RiskLevel | None:
        """Risk level of the latest observation before TCA, if it has a Pc."""
        latest = self.trend.latest_observation
        point = self.trend.latest
        if latest is None or point is None or point.observation is not latest:
            return None
        return point.level


def assess(
    observations: Iterable[PcObservation],
    *,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    trend_config: TrendConfig = DEFAULT_TREND_CONFIG,
    decision_config: DecisionConfig = DEFAULT_DECISION_CONFIG,
) -> Assessment:
    """The assessment of an event from its observations, in any order.

    Parameters
    ----------
    observations
        The event's observations, e.g. from
        :meth:`~keepout.events.ConjunctionEvent.observations`.
    thresholds
        The Pc thresholds of the risk levels and which value of an
        observation is compared with them; by default green below 1e-7, red
        from 1e-4, on the conservative basis (see
        :class:`~keepout.risk.Thresholds`).
    trend_config
        How many of the latest observations the trend fits and the slope
        still called steady; by default 3 observations and 0.5 decades of
        Pc per day (see :class:`~keepout.risk.TrendConfig`).
    decision_config
        How long before TCA a red event is to be manoeuvred rather than
        monitored while a manoeuvre is prepared; by default two days (see
        :class:`~keepout.risk.DecisionConfig`).

    Returns
    -------
    Assessment
        The trend of the observations and the action decided from it, with
        its reasons. :attr:`Assessment.level` is the risk level of the latest
        Pc.

    Raises
    ------
    ValueError
        If there are no observations.

    Examples
    --------
    ```python
    from keepout import PcObservation, assess
    from keepout.units import DAY

    observations = [  # seconds before TCA, who computed the Pc, the Pc
        PcObservation(5 * DAY, "operator", pc=2e-6),
        PcObservation(3 * DAY, "operator", pc=3e-5),
        PcObservation(1.5 * DAY, "operator", pc=2e-4),
    ]
    assessment = assess(observations)
    action = assessment.decision.action
    print(assessment.level.key, assessment.trend.direction, action.label)
    # red rising maneuver
    for reason in assessment.decision.reasons:
        print(reason)
    # latest Pc 2.00e-04 is red, 1.50 d before TCA
    # trend rising (+0.57 decades per day)
    # within the decision point (2.00 d before TCA)
    ```
    """
    trend = analyse_trend(observations, thresholds=thresholds, config=trend_config)
    return Assessment(trend, decide(trend, config=decision_config))
