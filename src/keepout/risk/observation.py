"""One probability of collision of an event, as known some time before TCA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, get_args

from keepout.pc.result import Method, PcResult

__all__ = ["PC_FLOOR", "UPPER_BOUND_METHODS", "PcKind", "PcObservation"]

PcKind: TypeAlias = Literal["estimate", "upper_bound", "below_floor"]
"""What a Pc value means: an ``"estimate"`` of the probability; an
``"upper_bound"`` (a maximum Pc, used when a covariance is unknown); or
``"below_floor"``, when the source reports only that the probability is at
most the value given (ESA's Kelvins data floors log10 Pc at -30)."""

UPPER_BOUND_METHODS: Final[frozenset[Method]] = frozenset(
    {"frisbee", "dilution", "alfano"}
)
"""Pc methods whose value is an upper bound, not an estimate."""

PC_FLOOR: Final = 1e-30
"""Smallest Pc on a logarithmic scale; smaller values (and 0) are drawn and
fitted at this floor."""


def _probability(name: str, value: float) -> None:
    if not (math.isfinite(value) and 0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True, eq=False)
class PcObservation:
    """The Pc of one message of a conjunction event.

    Either ``pc`` or ``error`` is given: an observation whose Pc could not be
    computed is kept, with the reason, so that it is never silently lost.

    Parameters
    ----------
    time_to_tca
        Seconds from the message's creation to the time of closest approach;
        negative for a message issued after TCA.
    source
        Who computed ``pc``, e.g. ``"keepout (foster)"`` or ``"ESA (Kelvins)"``.
    pc
        Probability of collision in ``[0, 1]``; ``None`` if it could not be
        computed.
    kind
        What ``pc`` means; see :data:`PcKind`.
    ci_high
        Upper end of a Monte Carlo confidence interval, if ``pc`` has one.
    error
        Why ``pc`` could not be computed; given exactly when ``pc`` is ``None``.
    message_id
        Identifier of the message, e.g. its CDM ``MESSAGE_ID``.
    reported_pc
        The Pc the message's originator reported, shown for comparison.
    diagnostics
        Warnings about ``pc`` (covariance repairs, questionable assumptions).
    result
        keepout's full result, when keepout computed ``pc``.

    Raises
    ------
    ValueError
        If ``time_to_tca`` is not finite, ``source`` or ``error`` is empty,
        both or neither of ``pc`` and ``error`` are given, a probability is
        outside ``[0, 1]``, ``ci_high`` is below ``pc`` or given without it,
        or ``kind`` is unknown.
    """

    time_to_tca: float
    source: str
    pc: float | None = None
    kind: PcKind = "estimate"
    ci_high: float | None = None
    error: str | None = None
    message_id: str | None = None
    reported_pc: float | None = None
    diagnostics: tuple[str, ...] = ()
    result: PcResult | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_to_tca):
            raise ValueError(f"time_to_tca must be finite, got {self.time_to_tca}")
        if not self.source:
            raise ValueError("source must not be empty")
        if (self.pc is None) == (self.error is None):
            raise ValueError("give exactly one of pc and error")
        if self.error == "":
            raise ValueError("error must not be empty")
        if self.kind not in get_args(PcKind):
            raise ValueError(
                f"kind must be one of {get_args(PcKind)}, got {self.kind!r}"
            )
        if self.pc is not None:
            _probability("pc", self.pc)
            object.__setattr__(self, "pc", float(self.pc))
        if self.ci_high is not None:
            if self.pc is None:
                raise ValueError("ci_high needs pc")
            _probability("ci_high", self.ci_high)
            if self.ci_high < self.pc:
                raise ValueError(
                    f"ci_high ({self.ci_high}) must not be below pc ({self.pc})"
                )
            object.__setattr__(self, "ci_high", float(self.ci_high))
        if self.reported_pc is not None:
            _probability("reported_pc", self.reported_pc)
        object.__setattr__(self, "time_to_tca", float(self.time_to_tca))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def ok(self) -> bool:
        """Whether the observation has a Pc."""
        return self.pc is not None

    @classmethod
    def from_result(
        cls,
        result: PcResult,
        *,
        time_to_tca: float,
        message_id: str | None = None,
        reported_pc: float | None = None,
    ) -> PcObservation:
        """The observation of a keepout result.

        A maximum-Pc method gives an ``"upper_bound"``; Monte Carlo keeps the
        upper end of its confidence interval in ``ci_high``.

        Parameters
        ----------
        result
            keepout's result.
        time_to_tca
            Seconds from the message's creation to TCA.
        message_id
            Identifier of the message.
        reported_pc
            The Pc the message's originator reported.
        """
        ci_high = (
            result.details.get("ci_high") if result.method == "montecarlo" else None
        )
        return cls(
            time_to_tca=time_to_tca,
            source=f"keepout ({result.method})",
            pc=result.value,
            kind="upper_bound" if result.method in UPPER_BOUND_METHODS else "estimate",
            ci_high=ci_high,
            message_id=message_id,
            reported_pc=reported_pc,
            diagnostics=result.diagnostics,
            result=result,
        )
