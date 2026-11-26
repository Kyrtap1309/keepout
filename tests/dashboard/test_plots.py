"""The dashboard's Plotly figures of :mod:`keepout.figures`: the same things
the reports draw with matplotlib."""

import math
from typing import Any

import numpy as np

from keepout._types import FloatArray
from keepout.dashboard import plots
from keepout.figures import (
    LOWEST_LOG_PC,
    RISK_COLOURS,
    EncounterFigure,
    GeometryFigure,
    GridFigure,
    GridPanel,
    TrendFigure,
)
from keepout.maneuver import BurnDirection
from keepout.risk import RiskLevel


def _traces(figure: dict[str, Any]) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = figure["data"]
    return traces


def _trend() -> TrendFigure:
    hours = np.array([48.0, 36.0, 24.0])
    pc = np.array([1e-9, 3e-6, 2e-4])
    return TrendFigure(
        hours_to_tca=hours,
        pc=pc,
        levels=(RiskLevel.GREEN, RiskLevel.YELLOW, RiskLevel.RED),
        upper_bound=(False, False, False),
        below_floor=(True, False, False),
        labels=("A", "B", "C"),
        thresholds=((1e-7, RiskLevel.YELLOW), (1e-4, RiskLevel.RED)),
        fit=(hours, pc),
    )


def test_trend_colours_points_by_level_hollow_below_floor() -> None:
    figure = plots.trend(_trend()).to_dict()
    points = [trace for trace in _traces(figure) if trace["mode"] == "markers"]
    assert [trace["name"] for trace in points] == [
        "green (below floor)",
        "yellow",
        "red",
    ]
    assert [trace["marker"]["symbol"] for trace in points] == [
        "circle-open",
        "circle",
        "circle",
    ]
    assert [trace["marker"]["color"] for trace in points] == [
        RISK_COLOURS[RiskLevel.GREEN],
        RISK_COLOURS[RiskLevel.YELLOW],
        RISK_COLOURS[RiskLevel.RED],
    ]
    assert [trace["text"] for trace in points] == [["A"], ["B"], ["C"]]
    lines = {shape["y0"] for shape in figure["layout"]["shapes"]}
    assert lines == {1e-4, 1e-7}
    notes = {note["text"]: note["y"] for note in figure["layout"]["annotations"]}
    # On its line: a log axis places annotations at log10 of the value.
    assert notes == {"red from 1e-04": -4.0, "green below 1e-07": -7.0}
    assert figure["layout"]["yaxis"]["type"] == "log"
    assert figure["layout"]["xaxis"]["autorange"] == "reversed"


def test_trend_marks_the_decision_point() -> None:
    plain = plots.trend(_trend()).to_dict()
    assert sorted(shape["type"] for shape in plain["layout"]["shapes"]) == [
        "line",
        "line",
    ]
    figure = plots.trend(_trend(), decision_point_hours=120.0).to_dict()
    vertical = [s for s in figure["layout"]["shapes"] if s["x0"] == s["x1"]]
    assert [s["x0"] for s in vertical] == [120.0]
    assert vertical[0]["line"]["dash"] == "dot"
    notes = {note["text"] for note in figure["layout"]["annotations"]}
    assert "decision point" in notes


def _decision_anchor(hours: list[float], decision: float) -> str:
    """The horizontal anchor of the decision point's label: ``left`` puts
    the text to the right of the line on the screen."""
    pc = np.full(len(hours), 1e-5)
    figure = TrendFigure(
        hours_to_tca=np.array(hours),
        pc=pc,
        levels=(RiskLevel.YELLOW,) * len(hours),
        upper_bound=(False,) * len(hours),
        below_floor=(False,) * len(hours),
        labels=("",) * len(hours),
        thresholds=((1e-7, RiskLevel.YELLOW), (1e-4, RiskLevel.RED)),
        fit=None,
    )
    drawn = plots.trend(figure, decision_point_hours=decision).to_dict()
    (note,) = [
        n for n in drawn["layout"]["annotations"] if n["text"] == "decision point"
    ]
    anchor: str = note["xanchor"]
    return anchor


def test_decision_point_label_stays_over_the_points() -> None:
    # The x axis is reversed: an anchor on the left puts the text on the
    # screen's right of the line, nearer TCA.
    assert _decision_anchor([40.0, 20.0, 5.0], 48.0) == "left"
    assert _decision_anchor([110.0, 98.0, 74.0], 48.0) == "right"
    assert _decision_anchor([100.0, 30.0], 48.0) == "right"
    assert _decision_anchor([60.0, 10.0], 48.0) == "left"
    assert _decision_anchor([], 48.0) == "left"


def test_trend_of_one_point_has_no_line() -> None:
    figure = TrendFigure(
        hours_to_tca=np.array([10.0]),
        pc=np.array([1e-5]),
        levels=(RiskLevel.YELLOW,),
        upper_bound=(True,),
        below_floor=(False,),
        labels=("",),
        thresholds=((1e-7, RiskLevel.YELLOW), (1e-4, RiskLevel.RED)),
        fit=None,
    )
    traces = _traces(plots.trend(figure).to_dict())
    assert [trace["mode"] for trace in traces] == ["markers"]
    assert traces[0]["name"] == "yellow (max-Pc)"


def test_encounter_draws_the_ellipses_circle_and_miss() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 5)
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    figure = EncounterFigure(
        miss=(100.0, -50.0),
        ellipses=tuple(np.array([100.0, -50.0]) + k * 30.0 * unit for k in (1, 2, 3)),
        hbr_circle=10.0 * unit,
    )
    drawn = plots.encounter(figure).to_dict()
    names = [trace["name"] for trace in _traces(drawn)]
    assert names == ["1 sigma", "2 sigma", "3 sigma", "hard-body radius", "miss"]
    assert _traces(drawn)[-1]["x"] == [100.0]
    assert drawn["layout"]["yaxis"]["scaleanchor"] == "x"


def test_encounter_without_covariance_has_no_ellipse() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 5)
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    figure = EncounterFigure(
        miss=(100.0, -50.0),
        ellipses=tuple(np.tile([100.0, -50.0], (5, 1)) for _ in range(3)),
        hbr_circle=10.0 * unit,
    )
    drawn = plots.encounter(figure, covariance=False).to_dict()
    names = [trace["name"] for trace in _traces(drawn)]
    assert names == ["hard-body radius", "miss"]


def test_geometry_without_covariance_has_no_ellipsoid() -> None:
    grid = np.zeros((3, 3))
    figure = GeometryFigure(
        path=np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        at_tca=(0.0, 0.0, 0.0),
        velocity=(0.0, 1.0, 0.0),
        ellipsoid=(grid, grid, grid),
        hbr_sphere=(grid, grid, grid),
    )
    drawn = plots.geometry(figure, covariance=False).to_dict()
    names = [trace["name"] for trace in _traces(drawn)]
    assert names == ["secondary's path", "secondary at TCA", "hard-body sphere"]


def test_geometry_draws_path_ellipsoid_and_sphere() -> None:
    grid = np.zeros((3, 3))
    figure = GeometryFigure(
        path=np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        at_tca=(0.0, 0.0, 0.0),
        velocity=(0.0, 1.0, 0.0),
        ellipsoid=(grid, grid, grid),
        hbr_sphere=(grid, grid, grid),
    )
    drawn = plots.geometry(figure).to_dict()
    assert [trace["type"] for trace in _traces(drawn)] == [
        "scatter3d",
        "scatter3d",
        "surface",
        "surface",
    ]
    assert drawn["layout"]["scene"]["xaxis"]["title"]["text"] == "R [m]"


def _panel(
    direction: BurnDirection,
    log_pc: FloatArray,
    minimum: tuple[float, float] | None = (2.0, 0.03),
) -> GridPanel:
    return GridPanel(
        direction=direction,
        hours_before_tca=np.array([0.5, 1.5, 2.5, 3.5]),
        delta_v=np.array([0.005, 0.015, 0.05]),
        hour_centres=np.array([1.0, 2.0, 3.0]),
        delta_v_centres=np.array([0.01, 0.03]),
        log_pc=log_pc,
        minimum=minimum,
    )


def test_grid_draws_both_panels_contour_and_minimum() -> None:
    values = np.array([[-3.0, -6.0], [-4.0, -14.0], [math.nan, -8.0]])
    figure = GridFigure(
        directions=(
            _panel(BurnDirection.RETROGRADE, values),
            _panel(BurnDirection.PROGRADE, values),
        ),
        target_pc=1e-5,
        target_log_pc=-5.0,
        top_log_pc=-3.0,
    )
    drawn = plots.grid(figure).to_dict()
    traces = _traces(drawn)
    heatmaps = [trace for trace in traces if trace["type"] == "heatmap"]
    assert len(heatmaps) == 2
    assert all(trace["zmin"] == LOWEST_LOG_PC for trace in heatmaps)
    assert all(trace["zmax"] == -3.0 for trace in heatmaps)
    # One colour bar for both panels.
    assert [trace["showscale"] for trace in heatmaps] == [False, True]
    contours = [trace for trace in traces if trace["type"] == "contour"]
    assert len(contours) == 2
    assert all(trace["contours"]["start"] == -5.0 for trace in contours)
    marks = [
        trace for trace in traces if trace.get("name") == "smallest burn meeting it"
    ]
    assert [trace["x"] for trace in marks] == [[0.03], [0.03]]
    names = [trace["name"] for trace in traces if trace.get("showlegend")]
    assert names == ["target Pc 1e-05", "smallest burn meeting it", "no Pc"]
    assert drawn["layout"]["xaxis"]["autorange"] == "reversed"
    assert drawn["layout"]["xaxis"]["type"] == "log"


def test_grid_without_a_crossing_has_no_contour() -> None:
    values = np.array([[-3.0, -4.0], [-3.5, -4.5], [-3.2, -4.2]])
    figure = GridFigure(
        directions=(
            _panel(BurnDirection.RETROGRADE, values),
            _panel(BurnDirection.PROGRADE, values),
        ),
        target_pc=1e-5,
        target_log_pc=-5.0,
        top_log_pc=-3.0,
    )
    traces = _traces(plots.grid(figure).to_dict())
    assert [trace for trace in traces if trace["type"] == "contour"] == []
    assert "no Pc" not in [trace.get("name") for trace in traces]


def test_grid_legend_names_what_only_the_second_panel_draws() -> None:
    # No retrograde burn meets the target: the contour and the smallest
    # burn are drawn on the prograde panel alone, and the legend names them.
    above = np.array([[-3.0, -4.0], [-3.5, -4.5], [-3.2, -4.2]])
    crossing = np.array([[-3.0, -6.0], [-4.0, -7.0], [-4.5, -8.0]])
    figure = GridFigure(
        directions=(
            _panel(BurnDirection.RETROGRADE, above, minimum=None),
            _panel(BurnDirection.PROGRADE, crossing),
        ),
        target_pc=1e-5,
        target_log_pc=-5.0,
        top_log_pc=-3.0,
    )
    traces = _traces(plots.grid(figure).to_dict())
    names = [trace["name"] for trace in traces if trace.get("showlegend")]
    assert names == ["target Pc 1e-05", "smallest burn meeting it"]
    assert [trace["xaxis"] for trace in traces if trace.get("showlegend")] == [
        "x2",
        "x2",
    ]
