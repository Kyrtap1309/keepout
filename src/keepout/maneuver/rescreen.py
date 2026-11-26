"""What a burn does to the rest of the catalog: the approaches it brings.

A burn that clears one conjunction moves the primary along its orbit for days
afterwards, into approaches it did not have. :func:`rescreen` screens the
primary against a catalog twice over a window from the burn, without the burn
and with it (:class:`~keepout.propagation.ManeuveredPropagator`), and reports
the approaches that only the burn has, and the ones it brings closer. An
approach of a secondary is the same one, moved, while the baseline has an
approach of that secondary within the event's TCA tolerance
(:attr:`~keepout.events.ConjunctionEvent.tca_tolerance`, 10 minutes by
default).

A burn can also bring the event's own pair back. The planner looks for the
new TCA only near the old one, so in a slow encounter the two objects can
meet again hours away, at a Pc the plan never saw:
:attr:`RescreenResult.event_returns` holds those new or closer approaches of
the event's secondary, away from the event's TCA, at or above the plan's
target.

The answer "nothing new" must never stand for "nothing looked at". The primary
must be screened, in both runs: it is found in the catalog by its catalog
number, and a primary that is not there, or that screening leaves out (a stale
element set, say), is a :class:`~keepout.errors.ManeuverError`, as is a window
that does not hold the event's TCA and the tolerance either side, since a burn
moves that TCA and a range is read within the tolerance. SGP4 must also carry
the primary through the whole window: screening propagates only the objects of
pairs its filter keeps, so on its own it would not see a primary fail.

The event's secondary need not be screened: it may be missing from the
catalog given (or left out by screening), and the other objects still
deserve a check.
:attr:`RescreenResult.secondary_screened` says whether it was, and
:attr:`RescreenResult.skipped_objects` how many catalog objects were left out.

Element sets have no covariance, so the Pc of an approach reported here rests
on an **assumed** one (:mod:`keepout.synthetic.covariance_model`): the Pc of
``compute`` (``"auto"`` by default, as the synthetic operator's) on the
model's covariance of both objects at its TCA, with the plan's combined
hard-body radius, on the basis of the plan's thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import astropy.units as u
import numpy as np

from keepout._defaults import SCREENING_DURATION
from keepout._text import before_tca
from keepout._validation import require_non_negative
from keepout.errors import ManeuverError, PropagationError
from keepout.omm import Omm, deduplicate
from keepout.pc import AUTO, PcComputation
from keepout.propagation import (
    NO_ERROR,
    ManeuveredPropagator,
    Propagator,
    Sgp4Propagator,
    describe_error,
)
from keepout.risk import RiskLevel, Thresholds
from keepout.screening import (
    DEFAULT_VOLUME,
    CloseApproach,
    ScreeningResult,
    ScreeningRole,
    ScreeningVolume,
    SkippedObject,
    SkipReason,
    VolumeRule,
)
from keepout.screening.pipeline import prepare_catalog
from keepout.screening.tca import SEARCH_REACH
from keepout.synthetic.operator import DEFAULT_MODEL
from keepout.units import DAY, elapsed_seconds

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from astropy.time import Time

    from keepout.events import ConjunctionEvent
    from keepout.maneuver.planner import ManeuverContext, ManeuverOption
    from keepout.synthetic.covariance_model import SyntheticCovarianceModel

__all__ = ["NewApproach", "RescreenResult", "rescreen"]

_PRIMARY_STEP: Final = 60.0
"""Time between the checks of the primary's states over the window [s]."""


@dataclass(frozen=True, eq=False, slots=True, kw_only=True)
class NewApproach:
    """A close approach the burn brings, or brings closer.

    Parameters
    ----------
    approach
        The approach, with the burn.
    pc
        Its Pc on assumed covariances, on the basis of the plan's thresholds.
    level
        Its risk level.
    """

    approach: CloseApproach
    pc: float
    level: RiskLevel


@dataclass(frozen=True, eq=False, slots=True, kw_only=True)
class RescreenResult:
    """What a burn does to the primary's approaches with the catalog.

    Parameters
    ----------
    option
        The burn.
    new_approaches
        The approaches only the burn has, in chronological order; not the
        event's own when it was screened (``event_range`` has it).
    closer_approaches
        The approaches matched to the baseline whose range at TCA fell, by any
        amount, in chronological order: each against the baseline's approach
        of that secondary nearest in time. The event's own approach, that
        secondary's within the TCA tolerance of the event, is not among them
        when it was screened: ``event_range`` has it. Without
        ``secondary_screened`` it cannot be told apart, so it may appear
        here.
    event_returns
        The new and closer approaches of the event's secondary further than
        the TCA tolerance from the event's TCA whose Pc is at or above the plan's
        ``target_pc``, in chronological order: the burn brings the pair back
        elsewhere above the bar it was planned to clear. The event's own
        target is the bar for its own pair, whatever the risk level;
        ``conflict`` stays for red.
    event_range
        Closest range to the event's secondary after the burn within the
        event's TCA tolerance of its TCA [m]. With ``secondary_screened``, ``None``
        means the burn leaves it outside the screening volume there. Without
        it, ``None`` means it was **not screened**, never that it is outside
        the volume. An approach of that secondary further from the TCA is a
        new approach.
    window
        Start and end of the screening; the start is the burn.
    catalog_objects
        Number of catalog element sets screened with the burn.
    secondary_screened
        Whether both screenings looked at the event's secondary: ``False`` if
        its designator is not a catalog number, the catalog has no element set
        of it, or screening left it out. Nothing is then known of the event
        after the burn, whatever ``new_approaches`` and ``conflict`` say.
    skipped_objects
        Number of catalog objects the screening with the burn left out, the
        secondary among them if it was; an older element set of an object is
        not one, and the primary cannot be, as that raises. The re-screen says
        nothing of these objects.
    """

    option: ManeuverOption
    new_approaches: tuple[NewApproach, ...]
    closer_approaches: tuple[NewApproach, ...]
    event_returns: tuple[NewApproach, ...]
    event_range: float | None
    window: tuple[Time, Time]
    catalog_objects: int
    secondary_screened: bool
    skipped_objects: int

    @property
    def conflict(self) -> bool:
        """Whether a new or a closer approach is at :attr:`RiskLevel.RED`."""
        approaches = (*self.new_approaches, *self.closer_approaches)
        return any(item.level is RiskLevel.RED for item in approaches)

    @property
    def clean(self) -> bool:
        """Whether the burn brings no red approach and does not bring the
        event's pair back at or above the target (no :attr:`event_returns`)."""
        return not self.conflict and not self.event_returns


def _catalog_number(designator: str) -> int | None:
    """The catalog number an ``OBJECT_DESIGNATOR`` holds, in decimal digits;
    ``None`` if it holds anything else."""
    return int(designator) if designator.isascii() and designator.isdigit() else None


def _primary_number(event: ConjunctionEvent, newest: Mapping[int, Omm]) -> int:
    """The catalog number of the event's primary.

    Raises
    ------
    ManeuverError
        If its designator is not a catalog number, as a re-screen then has no
        object to find, or the catalog has no element set of it.
    """
    designator = event.latest.object1.object_id
    number = _catalog_number(designator)
    if number is None:
        raise ManeuverError(
            f"event {event.event_id}: the primary {designator} cannot be found in "
            "a catalog: its OBJECT_DESIGNATOR is not a catalog number"
        )
    if number not in newest:
        raise ManeuverError(
            f"event {event.event_id}: the primary {number} has no element set in "
            "the catalog"
        )
    return number


def _require_primary_carried(
    event: ConjunctionEvent, primary: Omm, start: Time, duration: float
) -> None:
    """A check that SGP4 has a state of the primary across the window, every
    :data:`_PRIMARY_STEP` and at its end. Elements SGP4 cannot be initialised
    with are left to screening, which says so.

    Raises
    ------
    ManeuverError
        If SGP4 reports an error at one of those times.
    """
    propagator, _, _ = Sgp4Propagator.from_omms([primary])
    if len(propagator) == 0:
        return
    offsets = np.append(np.arange(0.0, duration, _PRIMARY_STEP), duration)
    errors = propagator.propagate(start, offsets).errors[0]
    failing = np.flatnonzero(errors != NO_ERROR)
    if failing.size > 0:
        first = int(failing[0])
        when = (start + float(offsets[first]) * u.s).utc.isot
        raise ManeuverError(
            f"event {event.event_id}: the primary {primary.catalog_number} was not "
            f"screened ({SkipReason.PROPAGATION_FAILED}: "
            f"{describe_error(int(errors[first]))}, at {when})"
        )


def _left_out(
    result: ScreeningResult, number: int, role: ScreeningRole
) -> SkippedObject | None:
    """What screening says of an object it did not screen, if so. An older
    element set is left out as a duplicate of the one screened, which leaves
    the object in."""
    for skipped in result.skipped:
        if (
            skipped.catalog_number == number
            and skipped.role is role
            and skipped.reason is not SkipReason.DUPLICATE
        ):
            return skipped
    return None


def _require_primary_screened(
    result: ScreeningResult, event: ConjunctionEvent, primary: int
) -> None:
    """A check that screening looked at the primary.

    Raises
    ------
    ManeuverError
        If it left the primary out.
    """
    skipped = _left_out(result, primary, ScreeningRole.PRIMARY)
    if skipped is not None:
        raise ManeuverError(
            f"event {event.event_id}: the primary {primary} was not screened "
            f"({skipped.reason}: {skipped.detail})"
        )


def _screened_secondary(
    number: int | None,
    newest: Mapping[int, Omm],
    results: Iterable[ScreeningResult],
) -> int | None:
    """The catalog number of the event's secondary, ``number``, if every
    screening looked at it."""
    if number is None or number not in newest:
        return None
    left_out = (_left_out(result, number, ScreeningRole.CATALOG) for result in results)
    return None if any(item is not None for item in left_out) else number


def _skipped_objects(result: ScreeningResult) -> int:
    """How many catalog objects a screening left out."""
    return sum(
        skipped.role is ScreeningRole.CATALOG
        and skipped.reason is not SkipReason.DUPLICATE
        for skipped in result.skipped
    )


def _within_tolerance(one: Time, other: Time, tolerance: float) -> bool:
    """Whether two TCAs are of one approach, ``tolerance`` [s] apart at most."""
    return abs(elapsed_seconds(one, other)) <= tolerance


def _matched(
    approach: CloseApproach, baseline: ScreeningResult, tolerance: float
) -> CloseApproach | None:
    """The baseline's approach of the same secondary nearest in time, within
    the tolerance; ``None`` if there is none, so that this one is new."""
    matches = [
        known
        for known in baseline.approaches
        if known.secondary_number == approach.secondary_number
        and _within_tolerance(known.tca, approach.tca, tolerance)
    ]
    return min(
        matches,
        key=lambda known: abs(elapsed_seconds(known.tca, approach.tca)),
        default=None,
    )


@dataclass(frozen=True, slots=True)
class _Assessor:
    """What gives an approach of a re-screen its Pc and level."""

    combined_hard_body_radius: float
    elements: Mapping[int, Omm]
    """The element set screened of each object."""
    model: SyntheticCovarianceModel
    compute: PcComputation
    thresholds: Thresholds

    def __call__(self, approach: CloseApproach) -> NewApproach:
        """An approach with its Pc on the model's covariances, as of the burn."""
        tca, model, elements = approach.tca, self.model, self.elements
        conjunction = approach.to_conjunction(
            combined_hard_body_radius=self.combined_hard_body_radius,
            primary_covariance=model.covariance(elements[approach.primary_number], tca),
            secondary_covariance=model.covariance(
                elements[approach.secondary_number], tca
            ),
        )
        pc = self.thresholds.value_of_result(self.compute(conjunction))
        return NewApproach(approach=approach, pc=pc, level=self.thresholds.level(pc))


def _classified(
    after: ScreeningResult,
    baseline: ScreeningResult,
    is_the_event: Callable[[CloseApproach], bool],
    assessed: _Assessor,
    tolerance: float,
) -> tuple[tuple[NewApproach, ...], tuple[NewApproach, ...]]:
    """The approaches only ``after`` has, and those it has closer than
    ``baseline``, but the event's own (``event_range`` has it)."""
    matched = [
        (found, _matched(found, baseline, tolerance))
        for found in after.approaches
        if not is_the_event(found)
    ]
    new = tuple(assessed(found) for found, known in matched if known is None)
    closer = tuple(
        assessed(found)
        for found, known in matched
        if known is not None and found.range_at_tca < known.range_at_tca
    )
    return new, closer


def rescreen(
    plan: ManeuverContext,
    option: ManeuverOption,
    catalog: Iterable[Omm],
    *,
    duration: float = SCREENING_DURATION,
    volume: ScreeningVolume | VolumeRule = DEFAULT_VOLUME,
    model: SyntheticCovarianceModel = DEFAULT_MODEL,
    compute: PcComputation = AUTO,
    workers: int | None = 1,
) -> RescreenResult:
    """The approaches a burn brings, from a screening with and without it.

    See the module's description for the method. The window starts at the
    burn, and each object's element set is the one screening takes: the
    newest whose epoch is close enough to the window.

    Parameters
    ----------
    plan
        The event's :class:`~keepout.maneuver.ManeuverPlan` or
        :class:`~keepout.maneuver.ManeuverContext`; its latest CDM's
        ``OBJECT_DESIGNATOR`` values
        are the catalog numbers of the primary and the secondary.
    option
        The burn, ``burn_lead`` before the event's TCA. The window must hold
        that TCA and the event's TCA tolerance either side, so the baseline
        of a plan, with no lead, cannot be re-screened.
    catalog
        Element sets to screen against; must hold the event's primary. Without
        its secondary the rest is still screened
        (:attr:`RescreenResult.secondary_screened`).
    duration
        Length of the window [s], finite and >= 0.
    volume
        The screening volume, as for :func:`keepout.screening.screen`.
    model
        The covariance model of both objects of a new approach.
    compute
        The Pc method of a new approach, from :func:`~keepout.pc.computation`.
    workers
        Number of processes to screen in, or ``None`` for the default (see
        :func:`~keepout.screening.screen`: a script calling with more than 1
        needs an ``if __name__ == "__main__":`` guard).

    Returns
    -------
    RescreenResult
        The new and the closer approaches, those that bring the event's pair
        back above the plan's target, the range to the event's secondary
        after the burn and whether it was screened, the window, and how many
        catalog objects were screened and left out.

    Raises
    ------
    ManeuverError
        Checked before anything is screened:

        - the primary's designator is not a catalog number;
        - the catalog has no element set of the primary;
        - the window does not hold the event's TCA and its TCA tolerance
          either side;
        - SGP4 reports an error for the primary anywhere in the window
          (checked every minute and at the window's end).

        Found while screening:

        - screening leaves the primary out (stale or unsupported elements,
          no volume, SGP4 failing inside the window);
        - the burn cannot be integrated.
    ValueError
        If ``duration`` or ``workers`` is out of range, or ``option.delta_v``
        is not finite.
    TypeError
        If the volume rule returns something other than a
        :class:`~keepout.screening.volume.ScreeningVolume`.
    KeepoutError
        If the Pc of a new or a closer approach fails.
    """
    require_non_negative(duration, "duration")
    event = plan.event
    elements = tuple(catalog)
    newest = deduplicate(elements).by_number
    number = _primary_number(event, newest)
    lead = option.burn_lead
    tolerance = event.tca_tolerance
    burn = f"burn of {option.delta_v:g} m/s {before_tca(lead)}"
    if not tolerance <= lead <= duration - tolerance:
        raise ManeuverError(
            f"event {event.event_id}: a window of {duration / DAY:g} days from a "
            f"{burn} does not hold the event's TCA and {tolerance:g} s "
            "either side of it"
        )

    tca = plan.conjunction.tca
    start: Time = tca - lead * u.s
    horizon = duration + SEARCH_REACH

    def burned(base: Propagator) -> Propagator:
        return ManeuveredPropagator(base, start, option.delta_v, horizon)

    # Both screenings share the catalog's work: its element sets, SGP4 and
    # radius bounds over the window.
    prepared = prepare_catalog(elements, start, duration, workers=workers)
    screened = prepared.by_number
    # The set screened; one screening leaves out says why.
    primaries = [screened.get(number, newest[number])]
    _require_primary_carried(event, primaries[0], start, duration)
    baseline = prepared.screen(primaries, volume)
    _require_primary_screened(baseline, event, number)
    try:
        after = prepared.screen(primaries, volume, adjust_primaries=burned)
    except PropagationError as exc:
        raise ManeuverError(f"event {event.event_id}: {burn}: {exc}") from exc
    _require_primary_screened(after, event, number)
    pair = _catalog_number(event.latest.object2.object_id)
    secondary = _screened_secondary(pair, newest, (baseline, after))

    def is_the_event(approach: CloseApproach) -> bool:
        return approach.secondary_number == secondary and _within_tolerance(
            approach.tca, tca, tolerance
        )

    assessed = _Assessor(
        plan.conjunction.combined_hard_body_radius,
        screened,
        model,
        compute,
        plan.thresholds,
    )
    new, closer = _classified(after, baseline, is_the_event, assessed, tolerance)
    # By its designator, not only when screened: an approach of the secondary
    # in one screening alone is still that pair.
    returns = tuple(
        item
        for item in sorted(
            (*new, *closer), key=lambda item: elapsed_seconds(tca, item.approach.tca)
        )
        if item.approach.secondary_number == pair
        and not _within_tolerance(item.approach.tca, tca, tolerance)
        and item.pc >= plan.target_pc
    )
    return RescreenResult(
        option=option,
        new_approaches=new,
        closer_approaches=closer,
        event_returns=returns,
        event_range=min(
            (
                approach.range_at_tca
                for approach in after.approaches
                if is_the_event(approach)
            ),
            default=None,
        ),
        window=(after.start, after.end),
        catalog_objects=after.catalog_count,
        secondary_screened=secondary is not None,
        skipped_objects=_skipped_objects(after),
    )
