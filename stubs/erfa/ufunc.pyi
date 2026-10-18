# The raw ERFA ufuncs keepout calls: they return ERFA's status last
# instead of warning or raising.
import numpy as np
import numpy.typing as npt

# Called on one epoch only.
def d2dtf(
    scale: bytes, ndp: int, d1: npt.ArrayLike, d2: npt.ArrayLike, /
) -> tuple[np.int32, np.int32, np.int32, np.void, np.int32]: ...
