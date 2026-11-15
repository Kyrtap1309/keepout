# make_subplots as keepout.dashboard.plots calls it: a row of panels.
from collections.abc import Sequence

from plotly.graph_objects import Figure

def make_subplots(
    rows: int = 1,
    cols: int = 1,
    *,
    shared_yaxes: bool = False,
    horizontal_spacing: float | None = None,
    subplot_titles: Sequence[str] | None = None,
) -> Figure: ...
