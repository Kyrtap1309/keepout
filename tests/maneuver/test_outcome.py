"""The outcome of planning an event: the burns, their re-screens and verdict."""

import math
from typing import Final

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time

from keepout._types import FloatArray
from keepout.cdm import CovarianceMethod, Maneuverable, ObjectMetadata, make_cdm
from keepout.conjunction import Conjunction
from keepout.covariance import Covariance
from keepout.events import ConjunctionEvent
from keepout.maneuver import (
    BurnGrid,
    ManeuverGrid,
    ManeuverOutcome,
    ManeuverPlan,
    PlannedBurn,
    evaluate_burn,
    plan_event,
    plan_maneuver,
    prepare_maneuver,
)
from keepout.objects import ObjectState
from keepout.orbits import EARTH_MU
from keepout.propagation.forces import EARTH_RADIUS
from keepout.settings import Settings
from keepout.state import StateVector
from keepout.units import HOUR, KM

TCA: Final = Time("2026-10-06T00:00:00", scale="utc")
RADIUS: Final = EARTH_RADIUS + 630 * KM
SMALL: Final = BurnGrid(steps_per_orbit=1, delta_v_per_direction=6)
"""About 70 cells at 25 h, which take a fraction of a second."""


def _item(object_id: str, r: FloatArray, angle: float) -> ObjectState:
    speed = math.sqrt(EARTH_MU / RADIUS)
    v = speed * np.array([0.0, math.cos(angle), math.sin(angle)])
    covariance = np.diag([100.0**2] * 3 + [0.01**2] * 3)
    return ObjectState(
        object_id,
        StateVector(TCA, r, v, "GCRF"),
        Covariance(covariance, "GCRF", TCA),
        10.0,
    )


def leo_event() -> ConjunctionEvent:
    """A fast crossing in LEO, 150 m apart radially, 25 h before TCA, as the
    planner's tests make it: Pc about 5e-3."""
    conjunction = Conjunction(
        _item("1001", np.array([RADIUS, 0.0, 0.0]), 0.9),
        _item("2002", np.array([RADIUS + 150.0, 0.0, 0.0]), -0.9),
        20.0,
    )
    method = CovarianceMethod.CALCULATED
    cdm = make_cdm(
        conjunction,
        ObjectMetadata(
            "PRI",
            "2020-001A",
            maneuverable=Maneuverable.YES,
            covariance_method=method,
        ),
        ObjectMetadata("SEC", "2019-001A", covariance_method=method),
        created=TCA - 25 * HOUR * u.s,
    )
    return ConjunctionEvent.from_cdms([cdm], event_id="EVENT-7")


@pytest.fixture(scope="module", name="event")
def the_event() -> ConjunctionEvent:
    return leo_event()


@pytest.fixture(scope="module", name="plan")
def small_plan(event: ConjunctionEvent) -> ManeuverPlan:
    return plan_maneuver(event, grid=SMALL)


def test_plan_event_matches_plan_maneuver(
    event: ConjunctionEvent, plan: ManeuverPlan
) -> None:
    outcome = plan_event(event, Settings(), grid=SMALL)
    assert outcome.event is event
    assert outcome.target_pc == plan.target_pc
    assert outcome.baseline == plan.baseline
    assert outcome.notes == plan.notes
    assert [b.option for b in outcome.burns] == list(plan.options)
    assert [b.direction for b in outcome.burns] == [o.direction for o in plan.options]
    assert outcome.grid is not None
    assert np.array_equal(outcome.grid.pc, plan.pc, equal_nan=True)
    assert np.array_equal(outcome.grid.burn_leads, plan.burn_leads)
    assert np.array_equal(outcome.grid.delta_v, plan.delta_v)
    assert np.array_equal(outcome.grid.range_at_tca, plan.range_at_tca, equal_nan=True)
    assert all(b.rescreen is None for b in outcome.burns)
    assert outcome.catalog_label is None
    assert outcome.assumed_covariance is event.latest.assumed_covariance
    assert outcome.verdict.recommendable


def test_the_grid_is_a_read_only_copy(plan: ManeuverPlan) -> None:
    grid = ManeuverGrid.of(plan)
    for array in (grid.burn_leads, grid.delta_v, grid.pc, grid.range_at_tca):
        assert not array.flags.writeable
    with pytest.raises(ValueError, match="shape"):
        ManeuverGrid(
            burn_leads=grid.burn_leads,
            delta_v=grid.delta_v,
            pc=grid.pc[:, :-1],
            range_at_tca=grid.range_at_tca,
        )


def test_outcome_of_a_context_has_no_grid(
    event: ConjunctionEvent, plan: ManeuverPlan
) -> None:
    context = prepare_maneuver(event)
    option = plan.options[0]
    burn = evaluate_burn(context, burn_lead=option.burn_lead, delta_v=option.delta_v)
    outcome = ManeuverOutcome.of(context, [PlannedBurn(burn, None)], None)
    assert outcome.grid is None
    assert outcome.burns == (PlannedBurn(burn, None),)
    assert outcome.baseline == context.baseline


def test_verdict_is_judge_of_the_burns(plan: ManeuverPlan) -> None:
    outcome = ManeuverOutcome.of(plan, [], None)
    assert outcome.verdict.burns == 0
    assert not outcome.verdict.recommendable


def test_catalog_needs_its_label(event: ConjunctionEvent) -> None:
    with pytest.raises(ValueError, match="catalog"):
        plan_event(event, Settings(), catalog=[])
    with pytest.raises(ValueError, match="catalog"):
        plan_event(event, Settings(), catalog_label="nothing")
