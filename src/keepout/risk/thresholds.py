"""Risk levels: Pc thresholds and which value of an observation is compared."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, get_args

from keepout._types import OrderedEnum
from keepout._validation import require_choice

if TYPE_CHECKING:
    from keepout.pc import PcResult
    from keepout.risk.observation import PcObservation

__all__ = ["BASES", "DEFAULT_THRESHOLDS", "Basis", "RiskLevel", "Thresholds"]


class RiskLevel(OrderedEnum):
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

BASES: Final[tuple[Basis, ...]] = get_args(Basis)
"""Every :data:`Basis`."""


@dataclass(frozen=True, slots=True)
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
        require_choice(self.basis, BASES, "basis")

    def value_of(self, observation: PcObservation) -> float:
        """The value of ``observation`` compared with the thresholds.

        On the conservative basis this is the upper end of the observation's
        confidence interval when it has one; otherwise it is its Pc.

        Parameters
        ----------
        observation
            An observation with a Pc.

        Raises
        ------
        ValueError
            If the observation has no Pc.
        """
        if observation.pc is None:
            raise ValueError(f"observation has no Pc: {observation.error}")
        return self._compared(observation.pc, observation.ci_high)

    def value_of_result(self, result: PcResult) -> float:
        """The value of a keepout result compared with the thresholds, as
        :meth:`value_of` takes it from an observation of the result.

        Parameters
        ----------
        result
            A result of a Pc method.
        """
        high = None if result.interval is None else result.interval[1]
        return self._compared(result.value, high)

    def _compared(self, pc: float, ci_high: float | None) -> float:
        """``ci_high`` on the conservative basis when there is one, else ``pc``."""
        if self.basis == "conservative" and ci_high is not None:
            return ci_high
        return pc

    def level(self, value: float) -> RiskLevel:
        """The risk level of a Pc ``value``.

        Parameters
        ----------
        value
            A Pc, e.g. from :meth:`value_of`.
        """
        if value >= self.red_from:
            return RiskLevel.RED
        if value >= self.green_below:
            return RiskLevel.YELLOW
        return RiskLevel.GREEN


DEFAULT_THRESHOLDS: Final = Thresholds()
"""The default thresholds."""
