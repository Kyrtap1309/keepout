import numpy as np
import pytest

from keepout.figures import TrendFigure, trend_figure
from keepout.risk import PC_FLOOR, PcObservation, RiskLevel, Thresholds, assess
from keepout.units import HOUR


def observations(*pairs: tuple[float, float]) -> list[PcObservation]:
    return [
        PcObservation(hours * HOUR, "operator", pc=pc, message_id=f"M{hours:g}")
        for hours, pc in pairs
    ]


def figure(*pairs: tuple[float, float]) -> TrendFigure:
    return trend_figure(assess(observations(*pairs)), Thresholds())


def test_trend_figure_floors_and_orders() -> None:
    result = figure((12.0, 1e-6), (48.0, 1e-3), (24.0, 0.0))
    assert result.hours_to_tca.tolist() == [48.0, 24.0, 12.0]
    assert result.pc.tolist() == [1e-3, PC_FLOOR, 1e-6]
    assert result.pc[1] == PC_FLOOR
    assert result.levels == (RiskLevel.RED, RiskLevel.GREEN, RiskLevel.YELLOW)
    assert result.upper_bound == (False, False, False)
    assert result.below_floor == (False, False, False)
    assert result.labels == ("M48", "M24", "M12")
    assert result.thresholds == ((1e-7, RiskLevel.YELLOW), (1e-4, RiskLevel.RED))


def test_trend_line_joins_the_points() -> None:
    result = figure((48.0, 1e-3), (24.0, 0.0), (12.0, 1e-6))
    assert result.fit is not None
    hours, pc = result.fit
    np.testing.assert_array_equal(hours, result.hours_to_tca)
    np.testing.assert_array_equal(pc, result.pc)


def test_one_point_has_no_trend_line() -> None:
    assert figure((24.0, 1e-5)).fit is None


def test_points_are_on_the_assessments_basis() -> None:
    thresholds = Thresholds(basis="estimate")
    items = [
        PcObservation(24 * HOUR, "keepout (monte_carlo)", pc=1e-5, ci_high=2e-4),
        PcObservation(12 * HOUR, "keepout (maximum)", pc=1e-3, kind="upper_bound"),
    ]
    result = trend_figure(assess(items, thresholds=thresholds), thresholds)
    assert result.pc.tolist() == [1e-5, 1e-3]
    assert result.levels == (RiskLevel.YELLOW, RiskLevel.RED)
    assert result.upper_bound == (False, True)
    assert result.labels == ("", "")
    conservative = trend_figure(assess(items), Thresholds())
    assert conservative.pc.tolist() == [2e-4, 1e-3]


def test_points_below_the_floor_are_marked() -> None:
    items = [
        PcObservation(24 * HOUR, "ESA", pc=1e-30, kind="below_floor"),
        PcObservation(12 * HOUR, "ESA", pc=1e-5),
    ]
    result = trend_figure(assess(items), Thresholds())
    assert result.below_floor == (True, False)
    assert result.upper_bound == (False, False)


def test_observations_without_pc_or_after_tca_are_left_out() -> None:
    items = [
        *observations((24.0, 1e-5)),
        PcObservation(12 * HOUR, "operator", error="no covariance"),
        PcObservation(-1 * HOUR, "operator", pc=1e-2),
    ]
    result = trend_figure(assess(items), Thresholds())
    assert result.hours_to_tca.tolist() == [24.0]


def test_arrays_are_read_only() -> None:
    result = figure((48.0, 1e-3), (24.0, 1e-5))
    with pytest.raises(ValueError, match="read-only"):
        result.pc[0] = 1.0


def test_other_thresholds_are_an_error() -> None:
    assessment = assess(observations((24.0, 1e-5)))
    with pytest.raises(ValueError, match="thresholds"):
        trend_figure(assessment, Thresholds(green_below=1e-6))
