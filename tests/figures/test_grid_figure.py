import dataclasses

import numpy as np
import pytest
from builders import SMALL_GRID, leo_crossing

from keepout.figures import LOWEST_LOG_PC, GridFigure, grid, grid_figure
from keepout.maneuver import (
    BurnDirection,
    ManeuverGrid,
    ManeuverOutcome,
    ManeuverPlan,
    PlannedBurn,
    plan_maneuver,
)
from keepout.units import HOUR


@pytest.fixture(scope="module", name="plan")
def small_plan() -> ManeuverPlan:
    return plan_maneuver(leo_crossing(), grid=SMALL_GRID)


@pytest.fixture(scope="module", name="figure")
def small_figure(plan: ManeuverPlan) -> GridFigure:
    return grid_figure(plan)


def test_grid_figure_on_small_grid(plan: ManeuverPlan, figure: GridFigure) -> None:
    assert [panel.direction for panel in figure.directions] == [
        BurnDirection.RETROGRADE,
        BurnDirection.PROGRADE,
    ]
    assert figure.target_pc == plan.target_pc
    assert figure.target_log_pc == pytest.approx(np.log10(plan.target_pc))
    for panel in figure.directions:
        assert panel.log_pc.shape == (
            len(panel.hours_before_tca) - 1,
            len(panel.delta_v) - 1,
        )
        option = plan.option(panel.direction)
        assert option is not None
        assert panel.minimum == (option.burn_lead / HOUR, abs(option.delta_v))


def test_cells_are_the_plans(plan: ManeuverPlan, figure: GridFigure) -> None:
    hours = plan.burn_leads[::-1] / HOUR
    for panel in figure.directions:
        sign = 1.0 if panel.direction is BurnDirection.PROGRADE else -1.0
        columns = np.flatnonzero(np.sign(plan.delta_v) == sign)
        columns = columns[np.argsort(np.abs(plan.delta_v[columns]))]
        magnitudes = np.abs(plan.delta_v[columns])
        # Each cell's centre lies between its edges.
        assert np.all(panel.hours_before_tca[:-1] < hours)
        assert np.all(hours < panel.hours_before_tca[1:])
        assert np.all(panel.delta_v[:-1] < magnitudes)
        assert np.all(magnitudes < panel.delta_v[1:])
        np.testing.assert_array_equal(panel.hour_centres, hours)
        np.testing.assert_array_equal(panel.delta_v_centres, magnitudes)
        with np.errstate(divide="ignore"):
            expected = np.log10(plan.pc[::-1][:, columns])
        expected[np.isneginf(expected)] = LOWEST_LOG_PC - 1.0
        np.testing.assert_array_equal(panel.log_pc, expected)


def test_log_pc_is_not_clipped_and_keeps_missing_cells(plan: ManeuverPlan) -> None:
    pc = plan.pc.copy()
    pc[0, 0] = 0.0
    pc[0, -1] = 1e-20
    pc[1, -1] = np.nan
    changed = dataclasses.replace(plan, pc=pc)
    figure = grid_figure(changed)
    retrograde, prograde = figure.directions
    # Row 0 of the plan is the earliest burn: the last row of a panel. A Pc
    # of 0 lies just below the colour scale; a low Pc keeps its value.
    assert retrograde.log_pc[-1, -1] == LOWEST_LOG_PC - 1.0
    assert prograde.log_pc[-1, -1] == pytest.approx(-20.0)
    assert np.isnan(prograde.log_pc[-2, -1])


def test_the_colour_scale_tops_at_the_highest_pc_rounded_up(
    plan: ManeuverPlan,
) -> None:
    pc = np.full_like(plan.pc, 1e-9)
    pc[0, 0] = 3e-4
    pc[1, 1] = np.nan
    assert grid_figure(dataclasses.replace(plan, pc=pc)).top_log_pc == -3.0
    low = dataclasses.replace(plan, pc=np.zeros_like(plan.pc))
    assert grid_figure(low).top_log_pc == LOWEST_LOG_PC + 1.0
    # Without any Pc the scale tops at 0.1, as the report draws it.
    unknown = dataclasses.replace(plan, pc=np.full_like(plan.pc, np.nan))
    assert grid_figure(unknown).top_log_pc == -1.0


def test_a_direction_without_burns_is_an_error(plan: ManeuverPlan) -> None:
    one_sided = dataclasses.replace(plan, delta_v=np.abs(plan.delta_v))
    with pytest.raises(ValueError, match="retrograde"):
        grid_figure(one_sided)


def test_edges_lie_midway_and_beyond_the_ends() -> None:
    edges = grid._edges(np.array([1.0, 2.0, 4.0]), geometric=False)
    assert edges.tolist() == [0.5, 1.5, 3.0, 5.0]
    geometric = grid._edges(np.array([1e-2, 1e-1, 1.0]), geometric=True)
    assert np.allclose(np.log10(geometric), [-2.5, -1.5, -0.5, 0.5])
    assert grid._edges(np.array([7.0]), geometric=False).tolist() == [6.5, 7.5]


def test_a_direction_without_option_has_no_minimum(plan: ManeuverPlan) -> None:
    changed = dataclasses.replace(plan, prograde=None)
    assert grid_figure(changed).directions[1].minimum is None


def test_a_single_burn_time_gets_an_hour_wide_row(plan: ManeuverPlan) -> None:
    changed = dataclasses.replace(
        plan,
        burn_leads=plan.burn_leads[:1],
        pc=plan.pc[:1],
        range_at_tca=plan.range_at_tca[:1],
    )
    hours = plan.burn_leads[0] / HOUR
    for panel in grid_figure(changed).directions:
        np.testing.assert_allclose(panel.hours_before_tca, [hours - 0.5, hours + 0.5])
        assert panel.log_pc.shape == (1, len(panel.delta_v) - 1)


def _outcome(plan: ManeuverPlan, *, grid: bool = True) -> ManeuverOutcome:
    """The outcome of the plan's burns, unscreened, as the store caches it."""
    burns = [PlannedBurn(option, None) for option in plan.options]
    return ManeuverOutcome.of(
        plan, burns, None, grid=ManeuverGrid.of(plan) if grid else None
    )


def test_an_outcome_draws_as_its_plan(plan: ManeuverPlan, figure: GridFigure) -> None:
    drawn = grid_figure(_outcome(plan))
    assert drawn.target_pc == figure.target_pc
    assert drawn.top_log_pc == figure.top_log_pc
    for mine, theirs in zip(drawn.directions, figure.directions, strict=True):
        assert mine.direction is theirs.direction
        assert mine.minimum == theirs.minimum
        np.testing.assert_array_equal(mine.log_pc, theirs.log_pc)
        np.testing.assert_array_equal(mine.delta_v, theirs.delta_v)
        np.testing.assert_array_equal(mine.hours_before_tca, theirs.hours_before_tca)


def test_an_outcome_without_a_burn_in_a_direction_has_no_minimum(
    plan: ManeuverPlan,
) -> None:
    changed = dataclasses.replace(plan, prograde=None)
    assert grid_figure(_outcome(changed)).directions[1].minimum is None


def test_an_outcome_without_a_grid_is_an_error(plan: ManeuverPlan) -> None:
    with pytest.raises(ValueError, match="no grid"):
        grid_figure(_outcome(plan, grid=False))
