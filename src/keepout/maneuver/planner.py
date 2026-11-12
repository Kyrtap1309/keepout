"""The Pc grid of an event over Δv and burn time, and the minimum Δv per direction.

:func:`plan_maneuver` works from the event's latest CDM. Each cell adds the
change a burn makes (:func:`~keepout.maneuver.burn_effects`) to the primary's
state at TCA, finds the new closest approach, and takes Foster's Pc there with
the CDM's covariances unchanged. Foster costs a fraction of a millisecond per
cell where the event's own method can cost a minute, so that method only
confirms the cells that are chosen as options.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, NamedTuple

import astropy.units as u
import numpy as np

from keepout._defaults import DECISION_POINT, MIN_BURN_LEAD, TARGET_PC
from keepout._text import before_tca, plural
from keepout._types import FloatArray, read_only
from keepout._validation import (
    require_at_least,
    require_non_negative,
    require_open_probability,
    require_positive,
)
from keepout.cdm import Cdm, CdmObject, Maneuverable
from keepout.conjunction import Conjunction
from keepout.errors import (
    KeepoutError,
    ManeuverError,
    PropagationError,
)
from keepout.events import ConjunctionEvent, time_to_tca
from keepout.maneuver._differential import (
    BurnEffects,
    burn_effects,
    closest_approach_shift,
)
from keepout.objects import ObjectState
from keepout.orbits import orbital_period, perigee_radius
from keepout.pc import AUTO, Method, PcComputation, PcResult, computation
from keepout.propagation.forces import EARTH_RADIUS, ForceModel, earth_pole
from keepout.risk import DEFAULT_THRESHOLDS, Thresholds
from keepout.state import StateVector
from keepout.units import KM

if TYPE_CHECKING:
    from astropy.time import Time

    from keepout.covariance import Policy

__all__ = [
    "DEFAULT_BURN_GRID",
    "MAX_CONFIRMATIONS",
    "MIN_BURN_LEAD",
    "TARGET_PC",
    "BurnDirection",
    "BurnGrid",
    "ManeuverContext",
    "ManeuverOption",
    "ManeuverPlan",
    "evaluate_burn",
    "plan_maneuver",
    "prepare_maneuver",
]

MAX_CONFIRMATIONS: Final = 8
"""Most runs of the event's method on the candidates of one direction."""

_CELL_PC: Final = computation("foster")
"""The Pc of a cell of the grid."""

_SHIFT_LIMIT: Final = 1.0 / 20.0
"""How far from TCA the expansion about it is trusted, in orbital periods."""

_SLOW_ENCOUNTER: Final = 100.0
"""Relative speed [m/s] below which the new TCA is worth a note."""

_DRAG_PERIGEE: Final = 450 * KM
"""Perigee altitude [m] below which a plan without drag says what that costs."""


class BurnDirection(enum.StrEnum):
    """Which way a burn changes the velocity: along it or against it."""

    PROGRADE = "prograde"
    """Along the velocity: a positive Δv, which raises the semi-major axis."""
    RETROGRADE = "retrograde"
    """Against the velocity: a negative Δv, which lowers the semi-major
    axis."""

    @classmethod
    def of(cls, delta_v: float) -> BurnDirection:
        """The direction of a signed Δv along the velocity.

        Parameters
        ----------
        delta_v
            Signed Δv along the velocity [m/s].

        Raises
        ------
        ValueError
            If ``delta_v`` is 0, which is no burn.
        """
        if delta_v == 0.0:
            raise ValueError("a Δv of 0 is no burn and has no direction")
        return cls.PROGRADE if delta_v > 0.0 else cls.RETROGRADE

    @property
    def sign(self) -> float:
        """The sign of the Δv of a burn this way."""
        return 1.0 if self is BurnDirection.PROGRADE else -1.0


@dataclass(frozen=True, slots=True, kw_only=True)
class BurnGrid:
    """The burns of a plan: when, and how large.

    Parameters
    ----------
    earliest_lead
        Earliest burn before TCA [s]; a burn before the CDM's creation is left
        out as well.
    latest_lead
        Latest burn before TCA [s], >= 0.
    steps_per_orbit
        Burn times per orbital period of the primary.
    smallest_delta_v
        Smallest Δv [m/s] of each direction.
    largest_delta_v
        Largest Δv [m/s] of each direction.
    delta_v_per_direction
        Number of Δv of each direction, geometrically spaced.

    Raises
    ------
    ValueError
        If ``earliest_lead`` is not finite and positive or is below
        ``latest_lead``, ``latest_lead`` is negative, a Δv is not finite and
        positive, ``smallest_delta_v`` is above ``largest_delta_v``, or a count
        is below 1.
    """

    earliest_lead: float = DECISION_POINT
    latest_lead: float = MIN_BURN_LEAD
    steps_per_orbit: int = 8
    smallest_delta_v: float = 1e-3
    largest_delta_v: float = 1.0
    delta_v_per_direction: int = 31

    def __post_init__(self) -> None:
        require_positive(self.earliest_lead, "earliest_lead")
        require_non_negative(self.latest_lead, "latest_lead")
        if self.earliest_lead < self.latest_lead:
            raise ValueError(
                f"earliest_lead ({self.earliest_lead}) must not be below "
                f"latest_lead ({self.latest_lead})"
            )
        require_at_least(self.steps_per_orbit, 1, "steps_per_orbit")
        require_positive(self.smallest_delta_v, "smallest_delta_v")
        require_positive(self.largest_delta_v, "largest_delta_v")
        if self.largest_delta_v < self.smallest_delta_v:
            raise ValueError(
                f"largest_delta_v ({self.largest_delta_v}) must not be below "
                f"smallest_delta_v ({self.smallest_delta_v})"
            )
        require_at_least(self.delta_v_per_direction, 1, "delta_v_per_direction")

    @property
    def delta_v(self) -> FloatArray:
        """Signed Δv along the velocity [m/s], ascending, without 0."""
        magnitudes = np.geomspace(
            self.smallest_delta_v, self.largest_delta_v, self.delta_v_per_direction
        )
        changes: FloatArray = np.concatenate((-magnitudes[::-1], magnitudes))
        return changes


DEFAULT_BURN_GRID: Final = BurnGrid()
"""The default grid: 31 Δv per direction, 8 burn times per orbit, from 2 days
to 6 hours before TCA."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ManeuverOption:
    """One burn and what it does.

    Parameters
    ----------
    burn_lead
        Time of the burn before TCA [s]; 0 for the baseline, which has no burn.
    delta_v
        Signed Δv along the velocity [m/s].
    pc
        Pc after the burn, on the basis of the thresholds.
    range_at_tca
        Range at the new closest approach [m].
    method
        The method ``pc`` comes from.
    meets_target
        Whether ``pc`` is below the target.
    notes
        What the burn's own Pc says that the baseline's does not, and whether
        Monte Carlo stood in for Hall's method; empty for the baseline.
    """

    burn_lead: float
    delta_v: float
    pc: float
    range_at_tca: float
    method: Method
    meets_target: bool
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "notes", tuple(self.notes))

    @property
    def direction(self) -> BurnDirection:
        """The burn's direction.

        Raises
        ------
        ValueError
            For the baseline, which has no burn.
        """
        return BurnDirection.of(self.delta_v)


@dataclass(frozen=True, eq=False, slots=True, kw_only=True)
class ManeuverContext:
    """An event ready for burns: its conjunction, the target and the baseline.

    :func:`prepare_maneuver` makes one; :func:`evaluate_burn` and
    :func:`~keepout.maneuver.rescreen` take one, or a :class:`ManeuverPlan`.

    Parameters
    ----------
    event
        The event.
    conjunction
        The conjunction of its latest CDM.
    target_pc
        The Pc a burn must stay below.
    thresholds
        Whose :attr:`~keepout.risk.Thresholds.basis` turns a result into the
        Pc compared with ``target_pc``; :func:`evaluate_burn` and
        :func:`~keepout.maneuver.rescreen` use them too.
    baseline
        The event without a burn: the Pc of ``conjunction``, ``delta_v`` 0.
    confirmation
        The computation of ``baseline.method`` that confirms a burn, with the
        options that keep it reproducible (see
        :meth:`~keepout.pc.PcComputation.resolved`).
    notes
        What a reader must know.
    """

    event: ConjunctionEvent
    conjunction: Conjunction
    target_pc: float
    thresholds: Thresholds
    baseline: ManeuverOption
    confirmation: PcComputation
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "notes", tuple(self.notes))

    @property
    def assumed_covariance(self) -> bool:
        """Whether the latest CDM gives an assumed covariance: every Pc then
        rests on a model."""
        return self.event.latest.assumed_covariance


@dataclass(frozen=True, eq=False, slots=True, kw_only=True)
class ManeuverPlan(ManeuverContext):
    """The burns of an event: the Pc grid and the minimum Δv per direction.

    A :class:`ManeuverContext` with the grid of :func:`plan_maneuver`. The
    arrays are read-only copies of those given.

    It takes every parameter of :class:`ManeuverContext` too, as keywords:
    ``event``, ``conjunction``, ``target_pc``, ``thresholds``, ``baseline``,
    ``confirmation`` and ``notes``, which keep their meaning there. The
    parameters below are its own.

    Parameters
    ----------
    burn_leads
        Times of the burns before TCA [s], shape ``(B,)``, descending.
    delta_v
        Signed Δv along the velocity [m/s], shape ``(D,)``, ascending, without
        0.
    pc
        Foster's Pc of each cell, shape ``(B, D)``. NaN where it or the burn
        failed, or where the cell has no closest approach it can trust.
    range_at_tca
        Range at each cell's new closest approach [m], shape ``(B, D)``; NaN
        where the cell has none.
    prograde
        The smallest positive burn below the target, confirmed by
        ``baseline.method``; ``None`` if there is none.
    retrograde
        The same for negative burns.

    Raises
    ------
    ValueError
        If ``pc`` or ``range_at_tca`` does not have the shape ``(B, D)``.
    """

    burn_leads: FloatArray
    delta_v: FloatArray
    pc: FloatArray
    range_at_tca: FloatArray
    prograde: ManeuverOption | None
    retrograde: ManeuverOption | None

    def __post_init__(self) -> None:
        ManeuverContext.__post_init__(self)
        leads = np.array(self.burn_leads, dtype=np.float64)
        changes = np.array(self.delta_v, dtype=np.float64)
        pc = np.array(self.pc, dtype=np.float64)
        ranges = np.array(self.range_at_tca, dtype=np.float64)
        shape = (*leads.shape, *changes.shape)
        if pc.shape != shape or ranges.shape != shape:
            raise ValueError(
                f"pc and range_at_tca must have shape {shape}, got {pc.shape} "
                f"and {ranges.shape}"
            )
        read_only(leads, changes, pc, ranges)
        object.__setattr__(self, "burn_leads", leads)
        object.__setattr__(self, "delta_v", changes)
        object.__setattr__(self, "pc", pc)
        object.__setattr__(self, "range_at_tca", ranges)

    @property
    def options(self) -> tuple[ManeuverOption, ...]:
        """The options found, prograde first."""
        return tuple(
            item for item in (self.prograde, self.retrograde) if item is not None
        )

    def option(self, direction: BurnDirection) -> ManeuverOption | None:
        """The option of a direction; ``None`` if it has none.

        Parameters
        ----------
        direction
            Prograde or retrograde.
        """
        return self.prograde if direction is BurnDirection.PROGRADE else self.retrograde


class _Cells(NamedTuple):
    """Both objects at the closest approach of each cell, shape ``(B, D, ...)``."""

    shift: FloatArray
    """Time of the closest approach after TCA [s]; NaN without one."""
    primary_r: FloatArray
    primary_v: FloatArray
    secondary_r: FloatArray
    secondary_v: FloatArray
    ranges: FloatArray
    """Range at each closest approach [m]."""


class _Forces(NamedTuple):
    """The forces on each object, each with its own drag term."""

    primary: ForceModel
    secondary: ForceModel


def _period(event: ConjunctionEvent, conjunction: Conjunction) -> float:
    """The primary's orbital period [s].

    Raises
    ------
    ManeuverError
        If its orbit is not bound.
    """
    period = orbital_period(conjunction.primary.state)
    if period is None:
        raise ManeuverError(
            f"event {event.event_id}: the primary has no orbital period to step "
            "the burns by: its orbit is not bound"
        )
    return period


def _force(section: CdmObject, pole: FloatArray, event: ConjunctionEvent) -> ForceModel:
    """The forces on one object of a CDM.

    Raises
    ------
    ManeuverError
        If its ``CD_AREA_OVER_MASS`` is negative or not finite.
    """
    try:
        return ForceModel(pole, drag_term=section.cd_area_over_mass or None)
    except ValueError as exc:
        raise ManeuverError(
            f"event {event.event_id}: CD_AREA_OVER_MASS of {section.object_id}: {exc}"
        ) from exc


def _forces(event: ConjunctionEvent) -> _Forces:
    """The forces on both objects of the latest CDM.

    Raises
    ------
    ManeuverError
        If a ``CD_AREA_OVER_MASS`` is negative or not finite.
    """
    cdm = event.latest
    pole = earth_pole(cdm.tca)
    return _Forces(_force(cdm.object1, pole, event), _force(cdm.object2, pole, event))


def _effects(
    event: ConjunctionEvent,
    state: StateVector,
    leads: FloatArray,
    delta_v: FloatArray,
    force: ForceModel,
) -> BurnEffects:
    """:func:`burn_effects` for an event; a cell that falls below the Earth's
    surface or that the solver fails on is NaN.

    Raises
    ------
    ManeuverError
        If the orbit without a burn falls below the Earth's surface or the
        solver fails on it: nothing is known of the grid then.
    """
    try:
        return burn_effects(state, leads, delta_v, force)
    except PropagationError as exc:
        raise ManeuverError(f"event {event.event_id}: {exc}") from exc


def _closest_approaches(
    conjunction: Conjunction, effects: BurnEffects, forces: _Forces, limit: float
) -> _Cells:
    """Both objects at the new closest approach of each cell.

    A state moves to its cell's closest approach by the second-order Taylor
    step that :func:`closest_approach_shift` assumes. Without a minimum to go
    to, a cell is NaN, which is never read as safe.
    """
    primary, secondary = conjunction.primary.state, conjunction.secondary.state
    primary_r, primary_v = primary.r + effects.dr, primary.v + effects.dv
    flat_acceleration = forces.primary.acceleration(
        primary_r.reshape(-1, 3), primary_v.reshape(-1, 3)
    )
    primary_a = flat_acceleration.reshape(primary_r.shape)
    secondary_a = forces.secondary.acceleration(secondary.r[None], secondary.v[None])[0]
    shift = closest_approach_shift(
        secondary.r - primary_r,
        secondary.v - primary_v,
        secondary_a - primary_a,
        limit=limit,
    )
    t = shift[..., None]
    moved_primary = primary_r + (primary_v + 0.5 * primary_a * t) * t
    moved_secondary = secondary.r + (secondary.v + 0.5 * secondary_a * t) * t
    return _Cells(
        shift,
        moved_primary,
        primary_v + primary_a * t,
        moved_secondary,
        secondary.v + secondary_a * t,
        np.linalg.norm(moved_secondary - moved_primary, axis=-1),
    )


def _moved(item: ObjectState, r: FloatArray, v: FloatArray, epoch: Time) -> ObjectState:
    """``item`` at another state, with the same covariance."""
    covariance = item.covariance
    return ObjectState(
        item.object_id,
        StateVector(epoch, r, v, "GCRF"),
        None if covariance is None else covariance.at(epoch),
        item.hard_body_radius,
    )


def _cell_conjunction(
    conjunction: Conjunction,
    cells: _Cells,
    index: tuple[int, int],
    epoch: Time | None = None,
) -> Conjunction:
    """The conjunction of one cell that has a closest approach; ``epoch``, its
    time, if already known."""
    if epoch is None:
        epoch = conjunction.tca + float(cells.shift[index]) * u.s
    return Conjunction(
        _moved(
            conjunction.primary, cells.primary_r[index], cells.primary_v[index], epoch
        ),
        _moved(
            conjunction.secondary,
            cells.secondary_r[index],
            cells.secondary_v[index],
            epoch,
        ),
        conjunction.combined_hard_body_radius,
    )


def _pc_grid(conjunction: Conjunction, cells: _Cells) -> tuple[FloatArray, int]:
    """Foster's Pc of each cell, and how many failed where a cell had a minimum."""
    pc = np.full(cells.shift.shape, np.nan)
    failed = 0
    found = np.isfinite(cells.shift)
    epochs = conjunction.tca + cells.shift[found] * u.s
    for (row, column), epoch in zip(np.argwhere(found).tolist(), epochs, strict=True):
        try:
            result = _CELL_PC(
                _cell_conjunction(conjunction, cells, (row, column), epoch)
            )
        except KeepoutError:
            failed += 1
        else:
            pc[row, column] = result.value
    return pc, failed


def _option(
    result: PcResult,
    *,
    burn_lead: float,
    delta_v: float,
    range_at_tca: float,
    target_pc: float,
    thresholds: Thresholds,
    notes: tuple[str, ...] = (),
) -> ManeuverOption:
    """The option of a method's result, on the basis of the thresholds."""
    value = thresholds.value_of_result(result)
    return ManeuverOption(
        burn_lead=burn_lead,
        delta_v=delta_v,
        pc=value,
        range_at_tca=range_at_tca,
        method=result.method,
        meets_target=value < target_pc,
        notes=notes,
    )


def _own_notes(result: PcResult, baseline: tuple[str, ...]) -> tuple[str, ...]:
    """The notes of a burn: its result's diagnostics not in ``baseline``."""
    return tuple(note for note in result.diagnostics if note not in baseline)


@dataclass(frozen=True, eq=False, slots=True)
class _Search:
    """What the search for the options of a plan works on."""

    conjunction: Conjunction
    cells: _Cells
    leads: FloatArray
    delta_v: FloatArray
    pc: FloatArray
    target_pc: float
    confirmation: PcComputation
    thresholds: Thresholds
    baseline_notes: tuple[str, ...]

    def candidates(self, direction: BurnDirection) -> list[tuple[int, int]]:
        """The cells of a direction below the target, by ``(|Δv|, burn lead)``."""
        below = self.pc < self.target_pc
        rows, columns = np.nonzero(below & (np.sign(self.delta_v) == direction.sign))
        order = np.lexsort((self.leads[rows], np.abs(self.delta_v[columns])))
        return [(int(rows[i]), int(columns[i])) for i in order]

    def confirm(self, index: tuple[int, int]) -> ManeuverOption:
        """The option of one cell, by the event's confirmation.

        Raises
        ------
        KeepoutError
            If the confirmation fails there.
        """
        result = self.confirmation(
            _cell_conjunction(self.conjunction, self.cells, index)
        )
        return _option(
            result,
            burn_lead=float(self.leads[index[0]]),
            delta_v=float(self.delta_v[index[1]]),
            range_at_tca=float(self.cells.ranges[index]),
            target_pc=self.target_pc,
            thresholds=self.thresholds,
            notes=_own_notes(result, self.baseline_notes),
        )

    def option(
        self, direction: BurnDirection
    ) -> tuple[ManeuverOption | None, tuple[str, ...]]:
        """The smallest confirmed burn of a direction, and its notes: the
        burn's own, and how candidates fell through."""
        candidates = self.candidates(direction)
        refused = failed = 0
        found = None
        notes: list[str] = []
        for index in candidates[:MAX_CONFIRMATIONS]:
            try:
                option = self.confirm(index)
            except KeepoutError:
                failed += 1
                continue
            if option.meets_target:
                found = option
                notes += [f"{direction}: {note}" for note in option.notes]
                break
            refused += 1
        if refused or failed:
            note = (
                f"{direction}: of {plural(len(candidates), 'cell')} below the target "
                f"on Foster's Pc, {self.confirmation.method} refused {refused} and "
                f"failed on {failed}"
            )
            untested = len(candidates) - MAX_CONFIRMATIONS
            if found is None and untested > 0:
                tries = plural(MAX_CONFIRMATIONS, "try", "tries")
                note += (
                    f"; the search stopped after {tries} with "
                    f"{plural(untested, 'candidate')} left untested"
                )
            notes.append(note)
        return found, tuple(notes)


def _drag_off_note(state: StateVector) -> str:
    """The note of a plan without drag; a low perigee makes it a warning. The
    orbit is bound: the plan has checked."""
    note = "drag is off: the CDM gives the primary no positive CD_AREA_OVER_MASS"
    perigee = (perigee_radius(state) or np.inf) - EARTH_RADIUS
    if perigee >= _DRAG_PERIGEE:
        return note
    return (
        f"{note}, and with a perigee at {perigee / KM:.0f} km drag changes this "
        "grid: on one orbit each (CD·A/m = 0.02, 25 h) it moved log10 Pc by up to "
        "0.18 at 400 km and 1.29 at 300 km, measurements and no bound"
    )


def _notes(baseline: PcResult, cdm: Cdm, conjunction: Conjunction) -> tuple[str, ...]:
    """The notes of a plan, before those of its grid and options."""
    notes = list(baseline.diagnostics)
    if cdm.assumed_covariance:
        notes.append(
            "every Pc rests on an assumed covariance (COVARIANCE_METHOD = DEFAULT)"
        )
    if not cdm.object1.cd_area_over_mass:
        notes.append(_drag_off_note(conjunction.primary.state))
    if conjunction.relative_speed < _SLOW_ENCOUNTER:
        notes.append(
            f"relative speed {conjunction.relative_speed:.3g} m/s is below "
            f"{_SLOW_ENCOUNTER:g} m/s: the planner looks for the new TCA only within "
            f"1/{round(1.0 / _SHIFT_LIMIT)} of an orbit of TCA, so after a burn the "
            "pair can meet "
            "again hours away, at a Pc no cell shows; only a re-screen shows "
            "it (inside that window the new TCA was measured up to 28 m off at "
            "10 m/s)"
        )
    return tuple(notes)


def _grid_notes(
    pc: FloatArray, *, no_minimum: int, failed: int, failed_burns: int
) -> tuple[str, ...]:
    """The note of the cells without a Pc, if any; ``failed`` counts the
    ``failed_burns`` too."""
    if not no_minimum and not failed:
        return ()
    what = "the burn or Foster's Pc" if failed_burns else "Foster's Pc"
    return (
        (
            f"{no_minimum + failed} of {pc.size} cells have no Pc and are never "
            f"options: {no_minimum} without a closest approach to trust, "
            f"{failed} where {what} failed"
        ),
    )


def _burn_leads(event: ConjunctionEvent, grid: BurnGrid, period: float) -> FloatArray:
    """The times of the burns [s], descending, from the CDM's creation at the
    latest (a burn before it does not exist)."""
    earliest = min(grid.earliest_lead, time_to_tca(event.latest))
    step = period / grid.steps_per_orbit
    steps = int((earliest - grid.latest_lead) // step)
    return earliest - step * np.arange(steps + 1, dtype=np.float64)


def _prepare(
    event: ConjunctionEvent,
    *,
    target_pc: float,
    compute: PcComputation,
    policy: Policy,
    combined_hard_body_radius: float | None,
    thresholds: Thresholds,
    grid: BurnGrid | None = None,
) -> tuple[ManeuverContext, PcResult]:
    """:func:`prepare_maneuver`, with the baseline's own result; with a
    ``grid``, also the check that its burn window is still open.

    Raises
    ------
    ValueError
        If ``target_pc`` is not in (0, 1).
    ManeuverError
        As :func:`prepare_maneuver`, or if the CDM comes after the window of
        ``grid`` has closed.
    """
    require_open_probability(target_pc, "target_pc")
    cdm = event.latest
    if cdm.object1.maneuverable == Maneuverable.NO:
        raise ManeuverError(
            f"event {event.event_id}: the primary {cdm.object1.object_id} cannot "
            "manoeuvre (MANEUVERABLE = NO)"
        )
    if grid is not None and time_to_tca(cdm) < grid.latest_lead:
        raise ManeuverError(
            f"event {event.event_id}: the latest CDM was created "
            f"{before_tca(time_to_tca(cdm))}, after the burn window closed "
            f"{before_tca(grid.latest_lead)}"
        )
    conjunction = cdm.conjunction(
        combined_hard_body_radius=combined_hard_body_radius, policy=policy
    )
    unknown = conjunction.unknown_covariance
    if unknown:
        labels = ", ".join(conjunction.label(role) for role in unknown)
        raise ManeuverError(
            f"event {event.event_id}: no burn can be planned without a covariance "
            f"of {labels}"
        )
    # Both check the event before its Pc, which can take a minute, is computed.
    _period(event, conjunction)
    _forces(event)
    result = compute(conjunction)
    context = ManeuverContext(
        event=event,
        conjunction=conjunction,
        target_pc=target_pc,
        thresholds=thresholds,
        baseline=_option(
            result,
            burn_lead=0.0,
            delta_v=0.0,
            range_at_tca=conjunction.relative_motion.distance,
            target_pc=target_pc,
            thresholds=thresholds,
        ),
        confirmation=compute.resolved(result),
        notes=_notes(result, cdm, conjunction),
    )
    return context, result


def prepare_maneuver(
    event: ConjunctionEvent,
    *,
    target_pc: float = TARGET_PC,
    compute: PcComputation = AUTO,
    policy: Policy = "repair",
    combined_hard_body_radius: float | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> ManeuverContext:
    """An event ready for burns of one's own (:func:`evaluate_burn`), without
    the grid of :func:`plan_maneuver`: its baseline in a fraction of the time.

    Parameters
    ----------
    event
        The event; its latest CDM is planned from.
    target_pc
        The Pc a burn must stay below, in (0, 1).
    compute
        The Pc method of the baseline, from :func:`~keepout.pc.computation`.
        An explicit method confirms burns with its own options; the method
        ``"auto"`` chose confirms them with defaults, or with ``"auto"``'s
        seed and sample budget if it is the Monte Carlo fallback. Where
        ``"auto"`` chose Hall's method and it does not converge on a burn,
        Monte Carlo confirms that burn as ``"auto"`` would.
    policy
        Covariance policy; see :func:`keepout.covariance.make_covariance`.
    combined_hard_body_radius
        Combined radius [m]; if ``None``, the sum of both objects' ``HBR``.
    thresholds
        Whose :attr:`~keepout.risk.Thresholds.basis` turns a result into the
        Pc compared with ``target_pc``.

    Raises
    ------
    ValueError
        If ``target_pc`` is not in (0, 1).
    ManeuverError
        If the primary has ``MANEUVERABLE = NO``, a covariance is unknown, the
        primary's orbit is not bound, or a ``CD_AREA_OVER_MASS`` is negative or
        not finite.
    KeepoutError
        If the CDM's conjunction or its Pc cannot be computed, e.g.
        :class:`~keepout.errors.HardBodyRadiusError` for a missing radius.
    """
    context, _ = _prepare(
        event,
        target_pc=target_pc,
        compute=compute,
        policy=policy,
        combined_hard_body_radius=combined_hard_body_radius,
        thresholds=thresholds,
    )
    return context


def plan_maneuver(
    event: ConjunctionEvent,
    *,
    target_pc: float = TARGET_PC,
    grid: BurnGrid = DEFAULT_BURN_GRID,
    compute: PcComputation = AUTO,
    policy: Policy = "repair",
    combined_hard_body_radius: float | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> ManeuverPlan:
    """The Pc of along-track burns of the primary, from the latest CDM.

    The baseline is that of :func:`prepare_maneuver`. Every cell of the grid
    uses Foster's Pc. A cell is an option in a direction when it is below
    ``target_pc``: cells are tried from the smallest Δv and, among equals,
    from the latest burn, and the first that the event's method also puts
    below ``target_pc`` is taken. The method gets at most
    :data:`MAX_CONFIRMATIONS` tries per direction. Comparisons are on the
    basis of ``thresholds``.

    Takes tens of seconds on the default grid, and a run of the event's method
    (up to a minute for a slow encounter) per try.

    Parameters
    ----------
    event
        The event; its latest CDM is planned from.
    target_pc
        The Pc a burn must stay below, in (0, 1).
    grid
        The burns tried: their times before TCA and their Δv each way. By
        default :data:`DEFAULT_BURN_GRID`: 8 burn times per orbit from 2 days
        to 6 hours before TCA, 31 Δv each way from 1 mm/s to 1 m/s.
    compute
        The Pc method of the baseline, as for :func:`prepare_maneuver`.
    policy
        Covariance policy; see :func:`keepout.covariance.make_covariance`.
    combined_hard_body_radius
        Combined radius [m]; if ``None``, the sum of both objects' ``HBR``.
    thresholds
        Whose :attr:`~keepout.risk.Thresholds.basis` turns a result into the
        Pc compared with ``target_pc``.

    Returns
    -------
    ManeuverPlan
        The plan; a direction without an option has ``None``.

    Raises
    ------
    ValueError
        If ``target_pc`` is not in (0, 1).
    ManeuverError
        As :func:`prepare_maneuver`, and if the CDM comes after the burn
        window has closed, or the orbit without a burn falls below the
        Earth's surface or fails to integrate (a cell that does so is only
        left without a Pc).
    KeepoutError
        If the CDM's conjunction or its Pc cannot be computed, e.g.
        :class:`~keepout.errors.HardBodyRadiusError` for a missing radius.

    Examples
    --------
    ```python
    from keepout import cdm
    from keepout.events import ConjunctionEvent
    from keepout.maneuver import plan_maneuver
    from keepout.pc import computation

    event = ConjunctionEvent.from_cdms([cdm.load("tests/cdm/data/tracss_v2.kvn")])
    plan = plan_maneuver(  # Foster's Pc for every confirmation, for speed
        event, combined_hard_body_radius=20.0, compute=computation("foster")
    )
    print(f"{plan.baseline.pc:.2e}", plan.pc.shape)  # no burn; (burn times, Δv)
    # 4.50e-02 (72, 62)
    for option in plan.options:
        print(option.direction, f"{option.delta_v:+.4f} m/s", f"{option.pc:.1e}")
    # prograde +0.0040 m/s 9.6e-06
    # retrograde -0.0063 m/s 8.4e-06
    ```
    """
    context, result = _prepare(
        event,
        target_pc=target_pc,
        compute=compute,
        policy=policy,
        combined_hard_body_radius=combined_hard_body_radius,
        thresholds=thresholds,
        grid=grid,
    )
    conjunction = context.conjunction
    period = _period(event, conjunction)
    forces = _forces(event)
    leads = _burn_leads(event, grid, period)
    delta_v = grid.delta_v
    effects = _effects(event, conjunction.primary.state, leads, delta_v, forces.primary)
    cells = _closest_approaches(conjunction, effects, forces, _SHIFT_LIMIT * period)
    pc, failed = _pc_grid(conjunction, cells)
    search = _Search(
        conjunction=conjunction,
        cells=cells,
        leads=leads,
        delta_v=delta_v,
        pc=pc,
        target_pc=target_pc,
        confirmation=context.confirmation,
        thresholds=thresholds,
        baseline_notes=result.diagnostics,
    )
    prograde, prograde_notes = search.option(BurnDirection.PROGRADE)
    retrograde, retrograde_notes = search.option(BurnDirection.RETROGRADE)
    failed_burns = int(np.count_nonzero(effects.failed))
    no_minimum = int(np.count_nonzero(np.isnan(cells.shift))) - failed_burns
    return ManeuverPlan(
        event=event,
        conjunction=conjunction,
        target_pc=target_pc,
        thresholds=thresholds,
        baseline=context.baseline,
        confirmation=context.confirmation,
        burn_leads=leads,
        delta_v=delta_v,
        pc=pc,
        range_at_tca=cells.ranges,
        prograde=prograde,
        retrograde=retrograde,
        notes=(
            *context.notes,
            *_grid_notes(
                pc,
                no_minimum=no_minimum,
                failed=failed + failed_burns,
                failed_burns=failed_burns,
            ),
            *prograde_notes,
            *retrograde_notes,
        ),
    )


def evaluate_burn(
    plan: ManeuverContext,
    *,
    burn_lead: float,
    delta_v: float,
) -> ManeuverOption:
    """One burn of an event, on or off a plan's grid, by the confirmation of
    ``plan`` and on the basis of
    its thresholds.

    Parameters
    ----------
    plan
        The event's :class:`ManeuverContext` or :class:`ManeuverPlan`.
    burn_lead
        Time of the burn before TCA [s], >= 0.
    delta_v
        Signed Δv along the velocity [m/s].

    Returns
    -------
    ManeuverOption
        The burn, whether it meets the target of ``plan`` or not; its notes
        are its Pc's diagnostics that ``plan`` does not already give.

    Raises
    ------
    ValueError
        If ``burn_lead`` is negative, or either is not finite.
    ManeuverError
        If the burn comes before the latest CDM, falls below the Earth's
        surface or fails to integrate, or has no closest approach to trust.
    KeepoutError
        If the confirmation fails on the burn.
    """
    event, conjunction = plan.event, plan.conjunction
    cdm = event.latest
    if burn_lead > time_to_tca(cdm):
        raise ManeuverError(
            f"event {event.event_id}: a burn {before_tca(burn_lead)} comes before "
            f"the latest CDM, created {before_tca(time_to_tca(cdm))}"
        )
    forces = _forces(event)
    effects = _effects(
        event,
        conjunction.primary.state,
        np.array([burn_lead]),
        np.array([delta_v]),
        forces.primary,
    )
    if effects.failed[0, 0]:
        raise ManeuverError(
            f"event {event.event_id}: a burn of {delta_v:g} m/s "
            f"{before_tca(burn_lead)} falls below the Earth's surface or fails "
            "to integrate"
        )
    limit = _SHIFT_LIMIT * _period(event, conjunction)
    cells = _closest_approaches(conjunction, effects, forces, limit)
    if np.isnan(cells.shift[0, 0]):
        raise ManeuverError(
            f"event {event.event_id}: a burn of {delta_v:g} m/s "
            f"{before_tca(burn_lead)} has no closest approach to trust"
        )
    result = plan.confirmation(_cell_conjunction(conjunction, cells, (0, 0)))
    return _option(
        result,
        burn_lead=burn_lead,
        delta_v=delta_v,
        range_at_tca=float(cells.ranges[0, 0]),
        target_pc=plan.target_pc,
        thresholds=plan.thresholds,
        notes=_own_notes(result, plan.notes),
    )
