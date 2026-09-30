"""Fault injection into the batched peak-overlap iteration of the 3D-Nc rate.

The seam is :func:`keepout.pc._rate._iterate_overlap`; the code under test
(``_peak_overlap``) keeps its own error isolation and fallback logic.
"""

import importlib
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from keepout._types import FloatArray
from keepout.errors import GeometryError

rate_module = importlib.import_module("keepout.pc._rate")

Mask = npt.NDArray[np.bool_]
Iteration = Callable[..., tuple[Any, Mask]]
Failure = Callable[[Iteration, tuple[Any, ...], Mask], tuple[Any, Mask]]


def never_converges(*args: Any) -> tuple[Any, Mask]:
    """The iteration fails everywhere: the mean-centred expansion is kept."""
    times, mean_centred = args[2], args[4]
    return mean_centred, np.zeros(times.size, dtype=bool)


def not_converging(real: Iteration, args: tuple[Any, ...], failing: Mask) -> Any:
    """The real iteration, reported as not converged at the failing times."""
    gaussians, converged = real(*args)
    return gaussians, converged & ~failing


def raising(real: Iteration, args: tuple[Any, ...], failing: Mask) -> Any:
    """A batch holding a failing time raises, as a non-elliptic iterate does."""
    raise GeometryError("orbit is unbound")


def failing_where(when: Callable[[FloatArray], Mask], failure: Failure) -> Iteration:
    """``_iterate_overlap`` that fails through ``failure`` at the times ``when``."""
    real = rate_module._iterate_overlap

    def patched(*args: Any) -> tuple[Any, Mask]:
        failing = when(args[2])
        return failure(real, args, failing) if failing.any() else real(*args)

    return patched
