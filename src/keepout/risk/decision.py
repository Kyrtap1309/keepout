"""What to do about a conjunction event: ignore it, monitor it or manoeuvre."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from keepout._types import OrderedEnum, require_positive
from keepout.risk.observation import PC_FLOOR, PcObservation
from keepout.risk.thresholds import RiskLevel, Thresholds
from keepout.risk.trend import (
    Direction,
    Trend,
    TrendConfig,
    TrendPoint,
    analyse_trend,
)
from keepout.units import DAY

__all__ = [
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
    PREPARE_MANOEUVRE = 2
    """Watch the next messages and prepare an avoidance manoeuvre (a red
    event before the decision point, or a red event whose latest Pc is
    missing)."""
    MANOEUVRE = 3
    """Plan and execute an avoidance manoeuvre."""

    @property
    def label(self) -> str:
        """The action in words, as keepout prints it: ``prepare manoeuvre``."""
        return self.name.lower().replace("_", " ")


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

    decision_point: float = 2 * DAY

    def __post_init__(self) -> None:
        require_positive(self.decision_point, "decision_point")


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


def _days(seconds: float) -> str:
    return f"{seconds / DAY:.2f} d"


def _describe(point: TrendPoint, *, before_a_failure: bool = False) -> str:
    """The point's Pc, level and time; ``before_a_failure`` phrases it as the
    latest observation with a Pc, followed by one without."""
    observation = point.observation
    level = point.level.label
    if before_a_failure:
        text = f"the latest observation with a Pc was {level}: Pc {point.value:.2e}"
    else:
        text = f"latest Pc {point.value:.2e} is {level}"
    if point.value != observation.pc:
        text += " (upper end of the Monte Carlo confidence interval)"
    if observation.kind == "upper_bound":
        text += " (an upper bound: a covariance is unknown)"
    elif observation.kind == "below_floor":
        text += " (at most this: below the source's reporting floor)"
    return f"{text}, {_days(point.time_to_tca)} before TCA"


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
    settings: DecisionConfig,
    reasons: Sequence[str],
    concerns: Sequence[str],
) -> Decision:
    """The decision on a red event, judged ``time_to_tca`` seconds before TCA."""
    point = f"the decision point ({_days(settings.decision_point)} before TCA)"
    if time_to_tca <= settings.decision_point:
        return Decision(Action.MANOEUVRE, (*reasons, f"within {point}", *concerns))
    timing = f"before {point}: prepare a manoeuvre"
    return Decision(Action.PREPARE_MANOEUVRE, (*reasons, timing, *concerns))


def decide(trend: Trend, *, config: DecisionConfig | None = None) -> Decision:
    """The recommended action for an event's trend.

    Rules, on the latest observation before TCA:

    - no Pc (it could not be computed): the decision follows the latest
      observation with a Pc. If that one was red, the red rule below applies,
      judged at the time of the observation without a Pc; otherwise (no
      observation with a Pc, or it was not red) ``MONITOR``;
    - red: ``MANOEUVRE`` within the decision point, else
      ``PREPARE_MANOEUVRE``;
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
        The event's trend.
    config
        The decision point; :class:`DecisionConfig` defaults if ``None``.
    """
    settings = config if config is not None else DecisionConfig()
    limits = trend.thresholds
    latest = trend.latest_observation
    if latest is None:
        return Decision(
            Action.MONITOR, ("no observation before TCA: nothing to decide on",)
        )
    point = trend.latest
    if not latest.ok:
        reasons = [f"latest observation has no Pc: {latest.error}"]
        if point is None or point.level is not RiskLevel.RED:
            return Decision(Action.MONITOR, tuple(reasons))
        reasons += [_describe(point, before_a_failure=True), _trend_reason(trend)]
        return _red(latest.time_to_tca, settings, reasons, _concerns(point))
    assert point is not None  # noqa: S101 (the latest observation has a Pc)
    reasons = [_describe(point), _trend_reason(trend)]
    concerns = _concerns(point)
    if point.level is RiskLevel.RED:
        return _red(point.time_to_tca, settings, reasons, concerns)
    if point.level is RiskLevel.YELLOW:
        return Decision(Action.MONITOR, (*reasons, *concerns))
    if trend.direction is Direction.RISING and trend.slope is not None:
        reaches = _reaches(point, trend.slope, limits.green_below)
        if reaches:
            concerns.insert(
                0,
                f"rising fast enough to reach {limits.green_below:g} before TCA",
            )
        else:
            reasons.append(
                f"rising, but at this rate below {limits.green_below:g} at TCA"
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
    thresholds: Thresholds | None = None,
    trend_config: TrendConfig | None = None,
    decision_config: DecisionConfig | None = None,
) -> Assessment:
    """Assess an event from its observations, in any order.

    Parameters
    ----------
    observations
        The event's observations, e.g. from
        :meth:`~keepout.risk.event.ConjunctionEvent.observations`.
    thresholds
        Risk thresholds; :class:`~keepout.risk.thresholds.Thresholds`
        defaults if ``None``.
    trend_config
        How the trend's direction is decided; defaults if ``None``.
    decision_config
        The decision point; defaults if ``None``.

    Raises
    ------
    ValueError
        If there are no observations.
    """
    trend = analyse_trend(observations, thresholds=thresholds, config=trend_config)
    return Assessment(trend, decide(trend, config=decision_config))
