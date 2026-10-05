"""Fault injection and spies at the seams of the 3D-Nc rate.

The seams are :func:`keepout.pc._rate._iterate_overlap`, the batched
peak-overlap iteration (the code under test, ``_peak_overlap``, keeps its own
error isolation and fallback logic), and the automatic window search
:func:`keepout.pc._timeline.locate_auto` as :mod:`keepout.pc._rate` calls it.
"""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from keepout._types import BoolArray, FloatArray
from keepout.errors import GeometryError
from keepout.pc import _rate as rate_module
from keepout.pc import _timeline as timeline_module
from keepout.pc._orbit import OrbitUncertainty
from keepout.pc._rate import Gaussians
from keepout.pc._timeline import Rate, RateScan, Screen

Arguments = tuple[OrbitUncertainty, OrbitUncertainty, FloatArray, float, Gaussians]
"""The arguments of ``_iterate_overlap``: primary, secondary, times, floor and
the mean-centred expansion."""

Iteration = Callable[
    [OrbitUncertainty, OrbitUncertainty, FloatArray, float, Gaussians],
    tuple[Gaussians, BoolArray],
]
Failure = Callable[[Iteration, Arguments, BoolArray], tuple[Gaussians, BoolArray]]


def never_converges(
    primary: OrbitUncertainty,
    secondary: OrbitUncertainty,
    times: FloatArray,
    floor: float,
    mean_centred: Gaussians,
) -> tuple[Gaussians, BoolArray]:
    """The iteration fails everywhere: the mean-centred expansion is kept."""
    return mean_centred, np.zeros(times.size, dtype=bool)


def not_converging(
    real: Iteration, args: Arguments, failing: BoolArray
) -> tuple[Gaussians, BoolArray]:
    """The real iteration, reported as not converged at the failing times."""
    gaussians, converged = real(*args)
    return gaussians, converged & ~failing


def raising(
    real: Iteration, args: Arguments, failing: BoolArray
) -> tuple[Gaussians, BoolArray]:
    """A batch holding a failing time raises, as a non-elliptic iterate does."""
    raise GeometryError("orbit is unbound")


def failing_where(
    when: Callable[[FloatArray], BoolArray], failure: Failure
) -> Iteration:
    """``_iterate_overlap`` that fails through ``failure`` at the times ``when``."""
    real = rate_module._iterate_overlap

    def patched(
        primary: OrbitUncertainty,
        secondary: OrbitUncertainty,
        times: FloatArray,
        floor: float,
        mean_centred: Gaussians,
    ) -> tuple[Gaussians, BoolArray]:
        args = (primary, secondary, times, floor, mean_centred)
        failing = when(times)
        return failure(real, args, failing) if failing.any() else real(*args)

    return patched


class WindowSearch(NamedTuple):
    """The scalar arguments of ``locate_auto``, all in s from TCA."""

    half_width: float
    """Initial widening of the window."""
    limit: float
    """Largest widening."""
    centre: tuple[float, float]
    """Stretch the window always covers."""


LocateAuto = Callable[[Rate, float, float, Screen, tuple[float, float]], RateScan]
"""``locate_auto``: rate, initial half-width, limit, screen and centre."""


def searching_with(change: Callable[[WindowSearch], WindowSearch]) -> LocateAuto:
    """``locate_auto`` run with its window arguments passed through ``change``.

    ``change`` may record them (a spy), replace them, or raise to stop the
    search before any scan.
    """
    real = timeline_module.locate_auto

    def patched(
        rate: Rate,
        half_width: float,
        limit: float,
        screen: Screen,
        centre: tuple[float, float] = (0.0, 0.0),
    ) -> RateScan:
        search = change(WindowSearch(half_width, limit, centre))
        return real(rate, search.half_width, search.limit, screen, search.centre)

    return patched
