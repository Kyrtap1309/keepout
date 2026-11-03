# The ERFA functions keepout calls. They are ufuncs: scalar arguments give
# NumPy scalars, array arguments arrays broadcast together. The scalar
# overloads come first; a scalar is array-like too, hence the ignores (only
# needed with the NumPy stubs of Python 3.12 and later).
from typing import TypeAlias, overload

import numpy as np
import numpy.typing as npt

_Float: TypeAlias = float | np.floating
_Int: TypeAlias = int | np.integer
_FloatArray: TypeAlias = npt.NDArray[np.float64]
_IntArray: TypeAlias = npt.NDArray[np.int32]

class ErfaWarning(Warning): ...

@overload
def jd2cal(  # type: ignore[overload-overlap, unused-ignore]
    dj1: _Float, dj2: _Float, /
) -> tuple[np.int32, np.int32, np.int32, np.float64]: ...
@overload
def jd2cal(
    dj1: npt.ArrayLike, dj2: npt.ArrayLike, /
) -> tuple[_IntArray, _IntArray, _IntArray, _FloatArray]: ...
@overload
def cal2jd(  # type: ignore[overload-overlap, unused-ignore]
    iy: _Int, im: _Int, id: _Int, /
) -> tuple[np.float64, np.float64]: ...
@overload
def cal2jd(
    iy: npt.ArrayLike, im: npt.ArrayLike, id: npt.ArrayLike, /
) -> tuple[_FloatArray, _FloatArray]: ...
@overload
def dat(  # type: ignore[overload-overlap, unused-ignore]
    iy: _Int, im: _Int, id: _Int, fd: _Float, /
) -> np.float64: ...
@overload
def dat(
    iy: npt.ArrayLike, im: npt.ArrayLike, id: npt.ArrayLike, fd: npt.ArrayLike, /
) -> _FloatArray: ...
def taiutc(
    tai1: npt.ArrayLike, tai2: npt.ArrayLike, /
) -> tuple[_FloatArray, _FloatArray]: ...
def gmst82(dj1: npt.ArrayLike, dj2: npt.ArrayLike, /) -> _FloatArray: ...
def era00(dj1: npt.ArrayLike, dj2: npt.ArrayLike, /) -> _FloatArray: ...
def c2i06a(date1: npt.ArrayLike, date2: npt.ArrayLike, /) -> _FloatArray: ...
def pnm06a(date1: npt.ArrayLike, date2: npt.ArrayLike, /) -> _FloatArray: ...
def bp06(
    date1: npt.ArrayLike, date2: npt.ArrayLike, /
) -> tuple[_FloatArray, _FloatArray, _FloatArray]: ...
