"""The outcome of planning an event: its burns, their re-screens and the verdict.

:func:`plan_event` plans an event with an operator's settings and re-screens
each burn against a catalog, if one is given; :meth:`ManeuverOutcome.of`
makes the same value from burns of one's own. A :class:`ManeuverOutcome`
holds what is shown of a plan, and :attr:`ManeuverOutcome.verdict` says
whether a burn can be recommended.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from keepout._types import FloatArray, read_only
from keepout.events import ConjunctionEvent
from keepout.maneuver.planner import (
    DEFAULT_BURN_GRID,
    BurnDirection,
    BurnGrid,
    ManeuverContext,
    ManeuverOption,
    ManeuverPlan,
    plan_maneuver,
)
from keepout.maneuver.rescreen import BurnVerdict, RescreenResult, judge, rescreen
from keepout.omm import Omm

# The settings sit above the library's manoeuvres: only their type is needed.
if TYPE_CHECKING:
    from keepout.settings import Settings

__all__ = ["ManeuverGrid", "ManeuverOutcome", "PlannedBurn", "plan_event"]


@dataclass(frozen=True, slots=True)
class PlannedBurn:
    """A burn of a plan, and its re-screen if there was a catalog.

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
            The plan.
        """
        return cls(
            burn_leads=plan.burn_leads,
            delta_v=plan.delta_v,
            pc=plan.pc,
            range_at_tca=plan.range_at_tca,
        )


@dataclass(frozen=True, eq=False, slots=True, kw_only=True)
class ManeuverOutcome:
    """What planning an event found: the baseline, the burns and their
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
        What a reader must know of the plan.
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
        plan: ManeuverContext,
        burns: Sequence[PlannedBurn],
        catalog_label: str | None,
    ) -> ManeuverOutcome:
        """The outcome of burns of a plan or a context.

        Parameters
        ----------
        plan
            The event's :class:`~keepout.maneuver.ManeuverPlan`, whose grid
            the outcome keeps, or :class:`~keepout.maneuver.ManeuverContext`.
        burns
            The burns, each with its re-screen if there was a catalog.
        catalog_label
            What the burns were re-screened against; ``None`` if they were
            not.
        """
        return cls(
            event=plan.event,
            target_pc=plan.target_pc,
            baseline=plan.baseline,
            notes=plan.notes,
            burns=tuple(burns),
            grid=ManeuverGrid.of(plan) if isinstance(plan, ManeuverPlan) else None,
            catalog_label=catalog_label,
        )


def plan_event(
    event: ConjunctionEvent,
    settings: Settings,
    *,
    catalog: Sequence[Omm] | None = None,
    catalog_label: str | None = None,
    grid: BurnGrid = DEFAULT_BURN_GRID,
    workers: int | None = 1,
) -> ManeuverOutcome:
    """The burns of an event, by :func:`~keepout.maneuver.plan_maneuver` with
    an operator's settings, each re-screened against a catalog if one is
    given.

    Parameters
    ----------
    event
        The event; its latest CDM is planned from.
    settings
        The target, Pc method, covariance policy, hard-body radius and
        thresholds of the plan.
    catalog
        The element sets to re-screen each burn against
        (:func:`~keepout.maneuver.rescreen`); ``None``: no re-screen.
    catalog_label
        What ``catalog`` is, for people; given with ``catalog`` and only then.
    grid
        The burns tried.
    workers
        Processes of each re-screen, as for
        :func:`~keepout.maneuver.rescreen`.

    Returns
    -------
    ManeuverOutcome
        The outcome, with the plan's grid.

    Raises
    ------
    ValueError
        If only one of ``catalog`` and ``catalog_label`` is given, and as
        :func:`~keepout.maneuver.plan_maneuver` and
        :func:`~keepout.maneuver.rescreen`.
    ManeuverError
        As :func:`~keepout.maneuver.plan_maneuver` and
        :func:`~keepout.maneuver.rescreen`.
    KeepoutError
        As :func:`~keepout.maneuver.plan_maneuver` and
        :func:`~keepout.maneuver.rescreen`.
    """
    if (catalog is None) != (catalog_label is None):
        raise ValueError("give catalog and catalog_label together, or neither")
    plan = plan_maneuver(
        event,
        target_pc=settings.target_pc,
        grid=grid,
        compute=settings.compute,
        policy=settings.policy,
        combined_hard_body_radius=settings.combined_hard_body_radius,
        thresholds=settings.thresholds,
    )
    burns = [
        PlannedBurn(
            option,
            None
            if catalog is None
            else rescreen(plan, option, catalog, workers=workers),
        )
        for option in plan.options
    ]
    return ManeuverOutcome.of(plan, burns, catalog_label)
