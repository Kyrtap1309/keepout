"""The outcome of planning an event: its burns, their re-screens and the verdict.

:func:`plan_event` plans an event on the burn grid, or evaluates one burn of
its own. It re-screens each burn against a catalog, if one is given.
:meth:`ManeuverOutcome.of` makes the same value from burns found otherwise.
A :class:`ManeuverOutcome` holds what is shown of an event's burns, and
:attr:`ManeuverOutcome.verdict` says whether a burn can be recommended.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from keepout._defaults import TARGET_PC
from keepout._types import FloatArray, read_only
from keepout.maneuver.planner import (
    DEFAULT_BURN_GRID,
    BurnDirection,
    BurnGrid,
    ManeuverContext,
    ManeuverOption,
    ManeuverPlan,
    evaluate_burn,
    plan_maneuver,
    prepare_maneuver,
)
from keepout.maneuver.rescreen import BurnVerdict, RescreenResult, judge, rescreen
from keepout.pc import AUTO, PcComputation
from keepout.risk import DEFAULT_THRESHOLDS, Thresholds

if TYPE_CHECKING:
    from collections.abc import Sequence

    from keepout.covariance import Policy
    from keepout.events import ConjunctionEvent
    from keepout.omm import Omm

__all__ = ["ManeuverGrid", "ManeuverOutcome", "PlannedBurn", "plan_event"]


@dataclass(frozen=True, slots=True)
class PlannedBurn:
    """A burn of an outcome, and its re-screen if there was a catalog.

    Parameters
    ----------
    option
        The burn.
    rescreen
        Its re-screen against the catalog; ``None`` without one.
    """

    option: ManeuverOption
    rescreen: RescreenResult | None

    @property
    def direction(self) -> BurnDirection:
        """The burn's direction along the velocity."""
        return self.option.direction


@dataclass(frozen=True, eq=False, slots=True, kw_only=True)
class ManeuverGrid:
    """The Pc grid of a :class:`~keepout.maneuver.ManeuverPlan`, without its
    event. The arrays are read-only copies of those given.

    Parameters
    ----------
    burn_leads
        Times of the burns before TCA [s], shape ``(B,)``.
    delta_v
        Signed Δv along the velocity [m/s], shape ``(D,)``.
    pc
        Foster's Pc of each cell, shape ``(B, D)``; NaN where it has none.
    range_at_tca
        Range at each cell's new closest approach [m], shape ``(B, D)``; NaN
        where the cell has none.

    Raises
    ------
    ValueError
        If ``pc`` or ``range_at_tca`` does not have the shape ``(B, D)``.
    """

    burn_leads: FloatArray
    delta_v: FloatArray
    pc: FloatArray
    range_at_tca: FloatArray

    def __post_init__(self) -> None:
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

    @classmethod
    def of(cls, plan: ManeuverPlan) -> ManeuverGrid:
        """The grid of a plan.

        Parameters
        ----------
        plan
            The plan, from :func:`~keepout.maneuver.plan_maneuver`.
        """
        return cls(
            burn_leads=plan.burn_leads,
            delta_v=plan.delta_v,
            pc=plan.pc,
            range_at_tca=plan.range_at_tca,
        )


@dataclass(frozen=True, eq=False, slots=True, kw_only=True)
class ManeuverOutcome:
    """The outcome of planning an event: the baseline, the burns and their
    re-screens.

    Parameters
    ----------
    event
        The event planned.
    target_pc
        The Pc a burn must stay below.
    baseline
        The event without a burn.
    notes
        What a reader must know of the outcome.
    burns
        The burns, prograde first; each with its re-screen if there was a
        catalog.
    grid
        The Pc grid the burns were found on; ``None`` for burns of one's own.
    catalog_label
        What the burns were re-screened against, for people; ``None`` if they
        were not.
    """

    event: ConjunctionEvent
    target_pc: float
    baseline: ManeuverOption
    notes: tuple[str, ...]
    burns: tuple[PlannedBurn, ...]
    grid: ManeuverGrid | None
    catalog_label: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "notes", tuple(self.notes))
        object.__setattr__(self, "burns", tuple(self.burns))

    @property
    def assumed_covariance(self) -> bool:
        """Whether the latest CDM gives an assumed covariance: every Pc then
        rests on a model."""
        return self.event.latest.assumed_covariance

    @property
    def verdict(self) -> BurnVerdict:
        """Whether a burn can be recommended, by :func:`~keepout.maneuver.judge`."""
        return judge([(burn.option, burn.rescreen) for burn in self.burns])

    @classmethod
    def of(
        cls,
        context: ManeuverContext,
        burns: Sequence[PlannedBurn],
        catalog_label: str | None,
        *,
        grid: ManeuverGrid | None,
    ) -> ManeuverOutcome:
        """The outcome of burns of an event.

        Parameters
        ----------
        context
            The event ready for burns: a
            :class:`~keepout.maneuver.ManeuverContext` or a
            :class:`~keepout.maneuver.ManeuverPlan`.
        burns
            The burns, each with its re-screen if there was a catalog.
        catalog_label
            What the burns were re-screened against; ``None`` if they were
            not.
        grid
            The Pc grid the burns were found on (:meth:`ManeuverGrid.of` a
            plan); ``None`` for burns of one's own.
        """
        return cls(
            event=context.event,
            target_pc=context.target_pc,
            baseline=context.baseline,
            notes=context.notes,
            burns=tuple(burns),
            grid=grid,
            catalog_label=catalog_label,
        )


def plan_event(
    event: ConjunctionEvent,
    *,
    target_pc: float = TARGET_PC,
    compute: PcComputation = AUTO,
    policy: Policy = "repair",
    combined_hard_body_radius: float | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    burn: tuple[float, float] | None = None,
    grid: BurnGrid = DEFAULT_BURN_GRID,
    catalog: Sequence[Omm] | None = None,
    catalog_label: str | None = None,
    workers: int | None = 1,
) -> ManeuverOutcome:
    """The outcome of an event's burns, each re-screened against a catalog
    if one is given.

    Without ``burn``, the burns are those of
    :func:`~keepout.maneuver.plan_maneuver` on ``grid``. With it, the one
    burn is evaluated by :func:`~keepout.maneuver.evaluate_burn`, and the
    grid is not planned.

    Parameters
    ----------
    event
        The event; its latest CDM is planned from.
    target_pc, compute, policy, combined_hard_body_radius, thresholds
        As for :func:`~keepout.maneuver.plan_maneuver`.
    burn
        One burn to evaluate instead of the grid: its time before TCA [s]
        and its signed Δv along the velocity [m/s].
    grid
        The burns tried without ``burn``.
    catalog
        The element sets to re-screen each burn against
        (:func:`~keepout.maneuver.rescreen`); ``None``: no re-screen.
    catalog_label
        What ``catalog`` is, for people; given with ``catalog`` and only then.
    workers
        Processes of each re-screen, as for
        :func:`~keepout.maneuver.rescreen`.

    Returns
    -------
    ManeuverOutcome
        The outcome; with the plan's grid unless ``burn`` is given.

    Raises
    ------
    ValueError
        If only one of ``catalog`` and ``catalog_label`` is given. Also as
        :func:`~keepout.maneuver.plan_maneuver`,
        :func:`~keepout.maneuver.evaluate_burn` and
        :func:`~keepout.maneuver.rescreen`.
    ManeuverError
        As those functions.
    KeepoutError
        As those functions.

    Examples
    --------
    One burn 12 hours before TCA, 1 cm/s prograde, without a re-screen:

    ```python
    from keepout import cdm
    from keepout.events import ConjunctionEvent
    from keepout.maneuver import plan_event
    from keepout.pc import computation

    event = ConjunctionEvent.from_cdms([cdm.load("tests/cdm/data/tracss_v2.kvn")])
    outcome = plan_event(
        event,
        combined_hard_body_radius=20.0,
        compute=computation("foster"),
        burn=(12 * 3600.0, 0.01),  # [s before TCA], [m/s]
    )
    (burn,) = outcome.burns
    print(burn.direction, f"{burn.option.pc:.2e}", burn.option.meets_target)
    # prograde 3.19e-10 True
    ```

    Pass ``catalog=`` (e.g. from :func:`keepout.omm.load`) and
    ``catalog_label=`` to screen each burn against it too.
    """
    if (catalog is None) != (catalog_label is None):
        raise ValueError("give catalog and catalog_label together, or neither")
    context: ManeuverContext
    if burn is None:
        context = plan = plan_maneuver(
            event,
            target_pc=target_pc,
            grid=grid,
            compute=compute,
            policy=policy,
            combined_hard_body_radius=combined_hard_body_radius,
            thresholds=thresholds,
        )
        options = plan.options
        found = ManeuverGrid.of(plan)
    else:
        context = prepare_maneuver(
            event,
            target_pc=target_pc,
            compute=compute,
            policy=policy,
            combined_hard_body_radius=combined_hard_body_radius,
            thresholds=thresholds,
        )
        lead, delta_v = burn
        options = (evaluate_burn(context, burn_lead=lead, delta_v=delta_v),)
        found = None
    burns = [
        PlannedBurn(
            option,
            None
            if catalog is None
            else rescreen(context, option, catalog, workers=workers),
        )
        for option in options
    ]
    return ManeuverOutcome.of(context, burns, catalog_label, grid=found)
