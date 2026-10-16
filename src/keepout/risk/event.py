"""A conjunction event: the CDMs of one encounter of two objects."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Final, TypeAlias

from astropy.time import Time

from keepout._types import require_non_negative
from keepout.cdm.model import Cdm
from keepout.conjunction import Conjunction
from keepout.covariance import Policy
from keepout.errors import EventError, HardBodyRadiusError, KeepoutError
from keepout.pc.api import MethodChoice, compute_pc
from keepout.pc.result import PcResult
from keepout.risk.observation import PcObservation

__all__ = [
    "DEFAULT_TCA_TOLERANCE",
    "ConjunctionEvent",
    "PcComputation",
    "group_events",
    "time_to_tca",
]

DEFAULT_TCA_TOLERANCE: Final = 600.0
"""Largest spread [s] of the TCAs of one event's CDMs. Successive CDMs of an
encounter move its TCA by milliseconds to seconds; two encounters of the same
pair are at least a large fraction of an orbit apart."""

PcComputation: TypeAlias = Callable[[Conjunction], PcResult]
"""A Pc computation with its own options, e.g.
``lambda c: compute_pc(c, "montecarlo", samples=100_000, seed=1)``."""


def time_to_tca(cdm: Cdm) -> float:
    """Seconds from the CDM's creation to its TCA (negative after TCA)."""
    return _seconds(cdm.tca, cdm.creation_date)


def _seconds(later: Time, earlier: Time) -> float:
    return float((later - earlier).to_value("s"))


def _pair(cdm: Cdm) -> tuple[str, str]:
    return cdm.object1.designator, cdm.object2.designator


def _reported(cdm: Cdm) -> tuple[float | None, tuple[str, ...]]:
    """The originator's Pc, if it is a probability; else a note saying why not."""
    reported = cdm.collision_probability
    if reported is None or (math.isfinite(reported) and 0.0 <= reported <= 1.0):
        return reported, ()
    return None, (f"reported COLLISION_PROBABILITY {reported} is not in [0, 1]",)


@dataclass(frozen=True, eq=False, slots=True)
class ConjunctionEvent:
    """The CDMs of one encounter, from the oldest to the latest.

    Parameters
    ----------
    event_id
        Identifier of the event.
    cdms
        The CDMs, in any order; kept sorted by ``CREATION_DATE`` (then
        ``MESSAGE_ID``).
    tca_tolerance
        Largest spread [s] of the CDMs' TCAs.

    Raises
    ------
    ValueError
        If ``event_id`` is empty or there are no CDMs.
    EventError
        If a ``MESSAGE_ID`` repeats, the CDMs are about different objects
        (``OBJECT1`` and ``OBJECT2`` by designator, in that order), their TCAs
        spread over more than ``tca_tolerance``, or they give different
        ``CONJUNCTION_ID`` values.
    """

    event_id: str
    cdms: tuple[Cdm, ...]
    tca_tolerance: float = DEFAULT_TCA_TOLERANCE

    def __post_init__(self) -> None:
        if not self.cdms:
            raise ValueError("an event needs at least one CDM")
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        require_non_negative(self.tca_tolerance, "tca_tolerance")
        first = self.cdms[0]
        ordered = tuple(
            sorted(
                self.cdms,
                key=lambda cdm: (
                    _seconds(cdm.creation_date, first.creation_date),
                    cdm.message_id,
                ),
            )
        )
        object.__setattr__(self, "cdms", ordered)
        counts = Counter(cdm.message_id for cdm in ordered)
        repeated = sorted(i for i, n in counts.items() if n > 1)
        if repeated:
            raise EventError(f"MESSAGE_ID repeated: {', '.join(repeated)}")
        pairs = {_pair(cdm) for cdm in ordered}
        if len(pairs) > 1:
            listed = ", ".join(f"{a}/{b}" for a, b in sorted(pairs))
            raise EventError(f"CDMs of one event must share both objects, got {listed}")
        offsets = [_seconds(cdm.tca, first.tca) for cdm in ordered]
        spread = max(offsets) - min(offsets)
        if spread > self.tca_tolerance:
            raise EventError(
                f"TCAs spread over {spread:.3g} s, more than tca_tolerance "
                f"({self.tca_tolerance:g} s)"
            )
        conjunction_ids = {
            cdm.conjunction_id for cdm in ordered if cdm.conjunction_id is not None
        }
        if len(conjunction_ids) > 1:
            raise EventError(
                "CDMs of one event give different CONJUNCTION_ID values: "
                f"{', '.join(sorted(conjunction_ids))}"
            )

    @classmethod
    def from_cdms(
        cls,
        cdms: Iterable[Cdm],
        *,
        event_id: str | None = None,
        tca_tolerance: float = DEFAULT_TCA_TOLERANCE,
    ) -> ConjunctionEvent:
        """The event of ``cdms``, named by ``event_id`` or after its CDMs.

        Parameters
        ----------
        cdms
            The CDMs, in any order.
        event_id
            Identifier; if ``None``, the CDMs' ``CONJUNCTION_ID`` (CDM 2.0),
            else ``"<OBJECT1>/<OBJECT2>/<TCA of the oldest CDM, UTC ISO>"``.
        tca_tolerance
            Largest spread [s] of the CDMs' TCAs.

        Raises
        ------
        ValueError
            As :class:`ConjunctionEvent`.
        """
        messages = tuple(cdms)
        if event_id is None and messages:
            given = sorted({i for m in messages if (i := m.conjunction_id) is not None})
            if given:
                event_id = given[0]
            else:
                oldest = min(
                    messages,
                    key=lambda m: _seconds(m.creation_date, messages[0].creation_date),
                )
                primary, secondary = _pair(oldest)
                event_id = f"{primary}/{secondary}/{oldest.tca.utc.isot}"
        return cls(event_id or "", messages, tca_tolerance)

    @property
    def primary_id(self) -> str:
        """``OBJECT_DESIGNATOR`` of ``OBJECT1``."""
        return _pair(self.cdms[0])[0]

    @property
    def secondary_id(self) -> str:
        """``OBJECT_DESIGNATOR`` of ``OBJECT2``."""
        return _pair(self.cdms[0])[1]

    @property
    def latest(self) -> Cdm:
        """The most recently created CDM."""
        return self.cdms[-1]

    @property
    def assumed_covariance(self) -> bool:
        """Whether a CDM of the event has an assumed covariance
        (:attr:`~keepout.cdm.Cdm.assumed_covariance`): its Pc rests on a model.
        """
        return any(cdm.assumed_covariance for cdm in self.cdms)

    def observations(
        self,
        *,
        method: MethodChoice | PcComputation = "auto",
        policy: Policy = "repair",
        hard_body_radius: float | None = None,
    ) -> tuple[PcObservation, ...]:
        """Keepout's Pc of every CDM, oldest first.

        Each CDM is computed as :func:`~keepout.pc.api.compute_pc` would (in
        ``"auto"`` mode a slow encounter can take up to a minute per CDM;
        ``method="foster"`` takes milliseconds). A CDM whose Pc cannot be
        computed (a :class:`~keepout.errors.KeepoutError`: a covariance or
        geometry problem, no convergence) gives an observation with the
        error instead of stopping the others. The originator's Pc is kept in
        ``reported_pc``.

        Parameters
        ----------
        method
            A method name, or ``"auto"``, or a :data:`PcComputation` for a
            method with options.
        policy
            Covariance policy; see :func:`keepout.covariance.make_covariance`.
        hard_body_radius
            Combined radius [m]; if ``None``, the sum of both objects'
            ``HBR`` in each CDM.

        Raises
        ------
        HardBodyRadiusError
            If ``hard_body_radius`` is ``None`` and a CDM lacks ``HBR`` for an
            object (checked before any Pc is computed).
        """
        if hard_body_radius is None:
            missing = [
                cdm.message_id
                for cdm in self.cdms
                if cdm.object1.hbr is None or cdm.object2.hbr is None
            ]
            if missing:
                raise HardBodyRadiusError(
                    f"no HBR for both objects in {', '.join(missing)}: pass "
                    "hard_body_radius= (combined, in metres)"
                )
        if callable(method):
            compute, source = method, "keepout"
        else:
            choice = method

            def compute(conjunction: Conjunction) -> PcResult:
                return compute_pc(conjunction, choice)

            source = f"keepout ({method})"
        return tuple(
            _observe(cdm, compute, source, policy, hard_body_radius)
            for cdm in self.cdms
        )


def _observe(
    cdm: Cdm,
    compute: PcComputation,
    source: str,
    policy: Policy,
    hard_body_radius: float | None,
) -> PcObservation:
    """The observation of one CDM (see :meth:`ConjunctionEvent.observations`)."""
    reported, notes = _reported(cdm)
    when = time_to_tca(cdm)
    assumed = cdm.assumed_covariance
    try:
        conjunction = cdm.conjunction(hard_body_radius=hard_body_radius, policy=policy)
        result = compute(conjunction)
    except KeepoutError as exc:
        return PcObservation(
            time_to_tca=when,
            source=source,
            error=f"{type(exc).__name__}: {exc}",
            message_id=cdm.message_id,
            reported_pc=reported,
            diagnostics=notes,
            assumed_covariance=assumed,
        )
    observation = PcObservation.from_result(
        result,
        time_to_tca=when,
        message_id=cdm.message_id,
        reported_pc=reported,
        assumed_covariance=assumed,
    )
    if notes:
        observation = replace(
            observation, diagnostics=(*observation.diagnostics, *notes)
        )
    return observation


def group_events(
    cdms: Iterable[Cdm], *, tca_tolerance: float = DEFAULT_TCA_TOLERANCE
) -> list[ConjunctionEvent]:
    """Group CDMs into events: same objects, TCAs within ``tca_tolerance``.

    CDMs of one pair of objects are taken in TCA order; a CDM whose TCA is
    more than ``tca_tolerance`` after the first TCA of the current event
    starts a new event. Events are named as in
    :meth:`ConjunctionEvent.from_cdms`.

    Parameters
    ----------
    cdms
        CDMs about any objects, in any order.
    tca_tolerance
        Largest spread [s] of the TCAs of one event.

    Returns
    -------
    list[ConjunctionEvent]
        The events, by their first TCA (then by objects).

    Raises
    ------
    ValueError
        If ``tca_tolerance`` is invalid.
    EventError
        If a group is not a valid event (e.g. a repeated ``MESSAGE_ID``, or
        different ``CONJUNCTION_ID`` values within ``tca_tolerance``).
    """
    require_non_negative(tca_tolerance, "tca_tolerance")
    messages = list(cdms)
    if not messages:
        return []
    reference = messages[0].tca
    by_pair: dict[tuple[str, str], list[Cdm]] = {}
    for cdm in messages:
        by_pair.setdefault(_pair(cdm), []).append(cdm)
    groups: list[tuple[float, tuple[str, str], list[Cdm]]] = []
    for pair, members in by_pair.items():
        timed = sorted(
            ((_seconds(cdm.tca, reference), cdm) for cdm in members),
            key=lambda item: item[0],
        )
        start = timed[0][0]
        current: list[Cdm] = []
        for offset, cdm in timed:
            if current and offset - start > tca_tolerance:
                groups.append((start, pair, current))
                current, start = [], offset
            current.append(cdm)
        groups.append((start, pair, current))
    groups.sort(key=lambda group: (group[0], group[1]))
    return [
        ConjunctionEvent.from_cdms(members, tca_tolerance=tca_tolerance)
        for _, _, members in groups
    ]
