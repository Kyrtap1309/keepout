"""One probability of collision of an event, as known some time before TCA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from keepout._validation import require_choice, require_probability
from keepout.pc.result import RESULT_KINDS, PcResult, ResultKind

__all__ = ["PC_FLOOR", "PC_KINDS", "PcKind", "PcObservation"]

PcKind: TypeAlias = ResultKind | Literal["below_floor"]
"""What a Pc value means: a :data:`~keepout.pc.ResultKind` (an
``"estimate"`` of the probability, or an ``"upper_bound"``: a maximum Pc,
used when a covariance is unknown); or ``"below_floor"``, when the source
reports only that the probability is at most the value given (ESA's Kelvins
data floors log10 Pc at -30)."""

PC_KINDS: Final[tuple[PcKind, ...]] = (*RESULT_KINDS, "below_floor")
"""Every :data:`PcKind`."""

PC_FLOOR: Final = 1e-30
"""Smallest Pc on a logarithmic scale; smaller values (and 0) are drawn and
fitted at this floor."""


@dataclass(frozen=True, eq=False, slots=True)
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
    assumed_covariance
        Whether the message gives an object's covariance as assumed
        (``COVARIANCE_METHOD = DEFAULT``), not from orbit determination:
        ``pc`` then rests on a model.
    method
        The Pc method that gave ``pc``, e.g. ``"foster"``, when keepout
        computed it; kept without :attr:`result`, as when the observation is
        read back from a store.

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
    assumed_covariance: bool = False
    method: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_to_tca):
            raise ValueError(f"time_to_tca must be finite, got {self.time_to_tca}")
        if not self.source:
            raise ValueError("source must not be empty")
        if (self.pc is None) == (self.error is None):
            raise ValueError("give exactly one of pc and error")
        if self.error == "":
            raise ValueError("error must not be empty")
        require_choice(self.kind, PC_KINDS, "kind")
        if self.pc is not None:
            require_probability(self.pc, "pc")
            object.__setattr__(self, "pc", float(self.pc) + 0.0)
        if self.ci_high is not None:
            if self.pc is None:
                raise ValueError("ci_high needs pc")
            require_probability(self.ci_high, "ci_high")
            if self.ci_high < self.pc:
                raise ValueError(
                    f"ci_high ({self.ci_high}) must not be below pc ({self.pc})"
                )
            object.__setattr__(self, "ci_high", float(self.ci_high) + 0.0)
        if self.reported_pc is not None:
            require_probability(self.reported_pc, "reported_pc")
        # Adding 0.0 turns -0.0 into 0.0: observations at TCA sort and print
        # alike whatever the sign of their zero.
        object.__setattr__(self, "time_to_tca", float(self.time_to_tca) + 0.0)
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
        assumed_covariance: bool = False,
        notes: tuple[str, ...] = (),
    ) -> PcObservation:
        """The observation of a keepout result.

        The result's :attr:`~keepout.pc.PcResult.kind` is kept (a maximum-Pc
        method gives an ``"upper_bound"``), and so is the upper end of its
        confidence interval (Monte Carlo) in ``ci_high``. The source is
        :attr:`~keepout.pc.PcComputation.label` of the method that gave the
        value, and :attr:`method` is that method.

        Parameters
        ----------
        result
            keepout's result of a Pc method, from :func:`~keepout.pc.compute_pc` or a
            :class:`~keepout.pc.PcComputation`.
        time_to_tca
            Seconds from the message's creation to TCA.
        message_id
            Identifier of the message.
        reported_pc
            The Pc the message's originator reported.
        assumed_covariance
            Whether the message gives an object's covariance as assumed.
        notes
            Diagnostics of the message, after the result's.
        """
        return cls(
            time_to_tca=time_to_tca,
            source=f"keepout ({result.method})",
            pc=result.value,
            kind=result.kind,
            ci_high=None if result.interval is None else result.interval[1],
            message_id=message_id,
            reported_pc=reported_pc,
            diagnostics=(*result.diagnostics, *notes),
            result=result,
            assumed_covariance=assumed_covariance,
            method=result.method,
        )
