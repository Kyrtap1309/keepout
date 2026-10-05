"""Risk levels: Pc thresholds and which value of an observation is compared."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Literal, TypeAlias, get_args

from keepout.risk.observation import PcObservation

__all__ = ["Basis", "RiskLevel", "Thresholds", "classify"]


class RiskLevel(enum.IntEnum):
    """Risk of a conjunction, ordered from the least to the most severe."""

    GREEN = 0
    """Pc below :attr:`Thresholds.green_below`."""
    YELLOW = 1
    """Pc between the two thresholds."""
    RED = 2
    """Pc at or above :attr:`Thresholds.red_from`."""


Basis: TypeAlias = Literal["conservative", "estimate"]
"""Which value of an observation is compared with the thresholds:
``"conservative"`` takes the upper end of a Monte Carlo confidence interval
where there is one, ``"estimate"`` always the Pc itself."""


@dataclass(frozen=True)
class Thresholds:
    """Pc thresholds of the risk levels.

    The defaults are those in common operational use (e.g. NASA's
    conjunction assessment practice: mitigate at Pc >= 1e-4); they are a
    starting point, not a standard for every mission.

    Parameters
    ----------
    green_below
        Pc below which a conjunction is :attr:`RiskLevel.GREEN`.
    red_from
        Pc from which a conjunction is :attr:`RiskLevel.RED`.
    basis
        Which value of an observation is compared; see :data:`Basis`.

    Raises
    ------
    ValueError
        Unless ``0 < green_below < red_from <= 1``, or if ``basis`` is unknown.
    """

    green_below: float = 1e-7
    red_from: float = 1e-4
    basis: Basis = "conservative"

    def __post_init__(self) -> None:
        values = (self.green_below, self.red_from)
        if not (
            all(map(math.isfinite, values))
            and 0.0 < self.green_below < self.red_from <= 1.0
        ):
            raise ValueError(
                "thresholds need 0 < green_below < red_from <= 1, got "
                f"green_below={self.green_below}, red_from={self.red_from}"
            )
        if self.basis not in get_args(Basis):
            raise ValueError(
                f"basis must be one of {get_args(Basis)}, got {self.basis!r}"
            )

    def value_of(self, observation: PcObservation) -> float:
        """The value of ``observation`` compared with the thresholds.

        Raises
        ------
        ValueError
            If the observation has no Pc.
        """
        if observation.pc is None:
            raise ValueError(f"observation has no Pc: {observation.error}")
        if self.basis == "conservative" and observation.ci_high is not None:
            return observation.ci_high
        return observation.pc

    def level(self, value: float) -> RiskLevel:
        """The risk level of a Pc ``value``."""
        if value >= self.red_from:
            return RiskLevel.RED
        if value >= self.green_below:
            return RiskLevel.YELLOW
        return RiskLevel.GREEN


def classify(
    observation: PcObservation, thresholds: Thresholds | None = None
) -> RiskLevel:
    """The risk level of an observation.

    Parameters
    ----------
    observation
        An observation with a Pc.
    thresholds
        The thresholds; :class:`Thresholds` defaults if ``None``.

    Raises
    ------
    ValueError
        If the observation has no Pc.
    """
    limits = thresholds if thresholds is not None else Thresholds()
    return limits.level(limits.value_of(observation))
