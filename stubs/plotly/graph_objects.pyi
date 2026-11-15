# The figure and the traces keepout.dashboard.plots draws. Plotly takes a
# trace's or layout's nested properties as dicts ({"color": ..., "width":
# ...}): a mapping of property names to values, as Plotly validates them at
# run time. to_dict gives the figure as Plotly's JSON-like dict.
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Self, TypeAlias

import numpy as np
import numpy.typing as npt
from plotly.basedatatypes import BaseFigure

_Values: TypeAlias = (
    npt.NDArray[np.float64] | Sequence[float] | Sequence[float | None] | Sequence[str]
)
_Grid: TypeAlias = npt.NDArray[np.float64] | Sequence[Sequence[float]]
_Properties: TypeAlias = Mapping[str, object]
_Colourscale: TypeAlias = Sequence[tuple[float, str]]
_Mode: TypeAlias = Literal["lines", "markers", "lines+markers"]

class BaseTraceType: ...

class Scatter(BaseTraceType):
    def __init__(
        self,
        *,
        x: _Values,
        y: _Values,
        mode: _Mode,
        name: str,
        line: _Properties | None = None,
        marker: _Properties | None = None,
        text: Sequence[str] | None = None,
        hovertemplate: str | None = None,
        hoverinfo: Literal["skip"] | None = None,
        fill: Literal["toself"] | None = None,
        showlegend: bool | None = None,
        legendgroup: str | None = None,
    ) -> None: ...

class Scatter3d(BaseTraceType):
    def __init__(
        self,
        *,
        x: _Values,
        y: _Values,
        z: _Values,
        mode: _Mode,
        name: str,
        line: _Properties | None = None,
        marker: _Properties | None = None,
    ) -> None: ...

class Surface(BaseTraceType):
    def __init__(
        self,
        *,
        x: _Grid,
        y: _Grid,
        z: _Grid,
        name: str,
        opacity: float | None = None,
        showscale: bool | None = None,
        showlegend: bool | None = None,
        colorscale: _Colourscale | None = None,
    ) -> None: ...

class Heatmap(BaseTraceType):
    def __init__(
        self,
        *,
        x: _Values,
        y: _Values,
        z: _Grid,
        zmin: float | None = None,
        zmax: float | None = None,
        colorscale: _Colourscale | None = None,
        showscale: bool | None = None,
        colorbar: _Properties | None = None,
        name: str | None = None,
        hovertemplate: str | None = None,
    ) -> None: ...

class Contour(BaseTraceType):
    def __init__(
        self,
        *,
        x: _Values,
        y: _Values,
        z: _Grid,
        contours: _Properties | None = None,
        line: _Properties | None = None,
        showscale: bool | None = None,
        name: str | None = None,
        showlegend: bool | None = None,
        legendgroup: str | None = None,
        hoverinfo: Literal["skip"] | None = None,
    ) -> None: ...

class Figure(BaseFigure):
    def __init__(self) -> None: ...
    def add_trace(
        self, trace: BaseTraceType, row: int | None = None, col: int | None = None
    ) -> Self: ...
    def add_hline(
        self,
        y: float,
        *,
        line: _Properties | None = None,
        annotation_text: str | None = None,
        annotation_position: str | None = None,
    ) -> Self: ...
    def add_vline(
        self,
        x: float,
        *,
        line: _Properties | None = None,
        annotation_text: str | None = None,
        annotation_position: str | None = None,
    ) -> Self: ...
    def update_layout(self, **kwargs: object) -> Self: ...
    def update_xaxes(
        self, *, row: int | None = None, col: int | None = None, **kwargs: object
    ) -> Self: ...
    def update_yaxes(
        self, *, row: int | None = None, col: int | None = None, **kwargs: object
    ) -> Self: ...
    def to_dict(self) -> dict[str, Any]: ...
