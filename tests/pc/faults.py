"""Fault injection and spies at the seams of the Pc methods.

Every patch of a private symbol of :mod:`keepout.pc` is made here, by a named
seam taking pytest's ``monkeypatch``, so tests do not name private symbols and
a rename in ``src`` is fixed in one place.

The main seams are :func:`keepout.pc._overlap._iterate_overlap`, the batched
peak-overlap iteration (the code under test, ``peak_overlap``, keeps its own
error isolation and fallback logic), the automatic window search
:func:`keepout.pc._timeline.locate_auto` as :mod:`keepout.pc._encounters`
calls it,
the rate of a batch of times (``CollisionRate._rates``), the Clenshaw-Curtis
orders of the time integral, the Monte Carlo batch size and the methods
:func:`keepout.pc.compute_pc` falls back between.
"""

import importlib
from collections.abc import Callable
from typing import NamedTuple, Self, cast

import numpy as np
import pytest
from typing_extensions import override

from keepout._types import BoolArray, FloatArray, IndexArray
from keepout.equinoctial import Orbits
from keepout.errors import GeometryError
from keepout.pc import _auto as auto_module
from keepout.pc import _encounters as encounters_module
from keepout.pc import _overlap as overlap_module
from keepout.pc import _quadrature as quadrature_module
from keepout.pc import _rate as rate_module
from keepout.pc._orbit import OrbitUncertainty
from keepout.pc._overlap import Gaussians
from keepout.pc._quadrature import RateIntegral
from keepout.pc._timeline import Rate, RateScan, Screen
from keepout.pc._timeline import locate_auto as real_locate_auto
from keepout.pc.montecarlo import _SamplePairs
from keepout.pc.result import PcResult

# The package exports the functions hall3d and montecarlo under their
# modules' names.
hall3d_module = importlib.import_module("keepout.pc.hall3d")
montecarlo_module = importlib.import_module("keepout.pc.montecarlo")

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
    real = overlap_module._iterate_overlap

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


def never_converging_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    """The peak-overlap iteration fails at every time."""
    monkeypatch.setattr(overlap_module, "_iterate_overlap", never_converges)


def overlap_failing_where(
    monkeypatch: pytest.MonkeyPatch,
    when: Callable[[FloatArray], BoolArray],
    failure: Failure,
) -> None:
    """The peak-overlap iteration fails through ``failure`` at the times ``when``."""
    monkeypatch.setattr(
        overlap_module, "_iterate_overlap", failing_where(when, failure)
    )


def overlap_failing_while_integrating(monkeypatch: pytest.MonkeyPatch) -> None:
    """The peak-overlap iteration fails at Hall's quadrature nodes only.

    It converges during the scan, and fails everywhere while
    :func:`~keepout.pc._quadrature.integrate_segments` integrates the rate.
    """
    real_integral = quadrature_module.integrate_segments
    real_iteration = overlap_module._iterate_overlap
    integrating = False

    def integral(rate: Rate, found: RateScan, *, rtol: float) -> RateIntegral:
        nonlocal integrating
        integrating = True
        try:
            return real_integral(rate, found, rtol=rtol)
        finally:
            integrating = False

    def iteration(
        primary: OrbitUncertainty,
        secondary: OrbitUncertainty,
        times: FloatArray,
        floor: float,
        mean_centred: Gaussians,
    ) -> tuple[Gaussians, BoolArray]:
        run = never_converges if integrating else real_iteration
        return run(primary, secondary, times, floor, mean_centred)

    monkeypatch.setattr(hall3d_module, "integrate_segments", integral)
    monkeypatch.setattr(overlap_module, "_iterate_overlap", iteration)


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
    real = real_locate_auto

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


def window_search(
    monkeypatch: pytest.MonkeyPatch, change: Callable[[WindowSearch], WindowSearch]
) -> None:
    """The automatic window search runs with its arguments passed through
    ``change`` (see :func:`searching_with`)."""
    monkeypatch.setattr(encounters_module, "locate_auto", searching_with(change))


def straight_line_closest_at(monkeypatch: pytest.MonkeyPatch, time: float) -> None:
    """The automatic window takes the straight-line closest approach at ``time``."""
    monkeypatch.setattr(encounters_module, "straight_line_closest_time", lambda _: time)


def rate_returning(monkeypatch: pytest.MonkeyPatch, value: float) -> None:
    """The collision rate is ``value`` wherever it is computed (not screened out)."""

    def constant(_: object, gaussians: Gaussians) -> FloatArray:
        return np.full(gaussians.x1.shape[0], value)

    monkeypatch.setattr(rate_module.CollisionRate, "_rates", constant)


def rate_raising(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    """Computing the collision rate at any time raises ``error``."""

    def broken(*_: object) -> FloatArray:
        raise error

    monkeypatch.setattr(rate_module.CollisionRate, "_rates", broken)


def rate_raising_for(
    monkeypatch: pytest.MonkeyPatch,
    when: Callable[[Gaussians], BoolArray],
    error: Exception,
) -> None:
    """Computing the collision rate raises ``error`` for any batch holding a
    Gaussian ``when`` selects, and is computed as usual otherwise."""
    real = rate_module.CollisionRate._rates

    def broken(self: rate_module.CollisionRate, gaussians: Gaussians) -> FloatArray:
        if np.any(when(gaussians)):
            raise error
        return real(self, gaussians)

    monkeypatch.setattr(rate_module.CollisionRate, "_rates", broken)


def time_integral_orders(
    monkeypatch: pytest.MonkeyPatch, *, start: int | None = None, largest: int
) -> None:
    """The time integral's Clenshaw-Curtis orders: the first is ``start``
    (unchanged if ``None``) and the finest ``largest``."""
    if start is not None:
        monkeypatch.setattr(quadrature_module, "CC_START_ORDER", start)
    monkeypatch.setattr(quadrature_module, "CC_MAX_ORDER", largest)


def monte_carlo_batches(monkeypatch: pytest.MonkeyPatch, size: int) -> None:
    """Monte Carlo runs its trials in batches of ``size``."""
    monkeypatch.setattr(montecarlo_module, "MC_BATCH", size)


Runner = Callable[..., PcResult]
"""A fake for :func:`~keepout.pc.hall3d.hall3d_over_scan` or
:func:`~keepout.pc.montecarlo.montecarlo_over_scan`: conjunction, encounters,
options."""


def auto_hall_integration(monkeypatch: pytest.MonkeyPatch, fake: Runner) -> None:
    """``compute_pc`` integrates Hall's rate over its scan with ``fake``."""
    monkeypatch.setattr(auto_module, "hall3d_over_scan", fake)


def auto_monte_carlo(monkeypatch: pytest.MonkeyPatch, fake: Runner) -> None:
    """``compute_pc``'s Monte Carlo fallback samples Hall's scan with ``fake``."""
    monkeypatch.setattr(auto_module, "montecarlo_over_scan", fake)


def auto_monte_carlo_budget(monkeypatch: pytest.MonkeyPatch, trials: int) -> None:
    """``compute_pc``'s Monte Carlo fallback runs at most ``trials`` trials."""
    monkeypatch.setattr(auto_module, "AUTO_MC_SAMPLES", trials)


class _MisleadingLine(_SamplePairs):
    """Relative motion ``r = (1, t - 0.1, 0)`` with a misleading acceleration.

    The acceleration ``(0, 20, 0)`` makes Newton's slope wrong, so from the
    interval's midpoint 0 (distance 1.005) it jumps to 2 and then creeps back
    by ~1/20 of the way per step: after every step it is still far from the
    closest approach at 0.1. It has no orbits: it overrides all it uses.
    """

    __slots__ = ()

    @override
    def subset(self, rows: IndexArray) -> Self:
        return self

    @override
    def relative(
        self, t: float | FloatArray, rows: slice | IndexArray = slice(None)
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        t = np.asarray(t, dtype=np.float64)
        one = np.ones_like(t)
        r = np.stack([one, t - 0.1, 0.0 * t], axis=1)
        v = np.stack([0.0 * t, one, 0.0 * t], axis=1)
        a = np.stack([0.0 * t, 20.0 * one, 0.0 * t], axis=1)
        return r, v, a


ClosestBelow = Callable[[IndexArray, float, float, float], BoolArray]
"""Whether each trial in ``rows`` gets within ``radius`` in ``[start, end]``."""


def misleading_newton_search() -> ClosestBelow:
    """Monte Carlo's closest-approach search for one trial whose relative
    acceleration misleads Newton's method (see :class:`_MisleadingLine`):
    it passes the closest approach at t = 0.1 (distance 1) but never settles
    there."""
    return _MisleadingLine(cast(Orbits, None), cast(Orbits, None)).closest_below
