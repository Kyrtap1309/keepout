"""The outcome of planning an event: the burns, their re-screens and verdict."""

import numpy as np
import pytest
from builders import SMALL_GRID, leo_crossing

from keepout.events import ConjunctionEvent
from keepout.maneuver import (
    ManeuverGrid,
    ManeuverOutcome,
    ManeuverPlan,
    PlannedBurn,
    evaluate_burn,
    plan_event,
    plan_maneuver,
    prepare_maneuver,
)


@pytest.fixture(scope="module", name="event")
def the_event() -> ConjunctionEvent:
    return leo_crossing()


@pytest.fixture(scope="module", name="plan")
def small_plan(event: ConjunctionEvent) -> ManeuverPlan:
    return plan_maneuver(event, grid=SMALL_GRID)


def test_plan_event_matches_plan_maneuver(
    event: ConjunctionEvent, plan: ManeuverPlan
) -> None:
    outcome = plan_event(event, grid=SMALL_GRID)
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
    assert outcome.burns
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
    outcome = ManeuverOutcome.of(context, [PlannedBurn(burn, None)], None, grid=None)
    assert outcome.grid is None
    assert outcome.burns == (PlannedBurn(burn, None),)
    assert outcome.baseline == context.baseline


def test_plan_event_of_one_burn_has_no_grid(
    event: ConjunctionEvent, plan: ManeuverPlan
) -> None:
    option = plan.options[0]
    outcome = plan_event(
        event, burn=(option.burn_lead, option.delta_v), target_pc=plan.target_pc
    )
    context = prepare_maneuver(event)
    burn = evaluate_burn(context, burn_lead=option.burn_lead, delta_v=option.delta_v)
    assert outcome.grid is None
    assert outcome.burns == (PlannedBurn(burn, None),)
    assert outcome.baseline == context.baseline
    assert outcome.notes == context.notes


def test_verdict_is_judge_of_the_burns(plan: ManeuverPlan) -> None:
    outcome = ManeuverOutcome.of(plan, [], None, grid=ManeuverGrid.of(plan))
    assert outcome.verdict.burns == 0
    assert not outcome.verdict.recommendable


def test_catalog_needs_its_label(event: ConjunctionEvent) -> None:
    with pytest.raises(ValueError, match="catalog"):
        plan_event(event, catalog=[])
    with pytest.raises(ValueError, match="catalog"):
        plan_event(event, catalog_label="nothing")


def test_a_baseline_meeting_the_target_needs_no_burn(event: ConjunctionEvent) -> None:
    plain = plan_event(event, grid=SMALL_GRID)
    target = plain.baseline.pc * 2.0
    outcome = plan_event(event, grid=SMALL_GRID, target_pc=target)
    assert outcome.baseline.meets_target
    assert outcome.verdict.no_burn_needed
    assert not outcome.verdict.recommendable
    assert outcome.grid is not None


def test_a_baseline_missing_the_target_needs_a_burn(plan: ManeuverPlan) -> None:
    outcome = ManeuverOutcome.of(plan, [], None, grid=ManeuverGrid.of(plan))
    assert not outcome.baseline.meets_target
    assert not outcome.verdict.no_burn_needed
