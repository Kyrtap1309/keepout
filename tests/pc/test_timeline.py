import math
from collections.abc import Callable

import numpy as np
import pytest

from keepout.errors import PcConvergenceError
from keepout.pc._timeline import (
    DETECTION_LEVEL,
    DETECTION_SIGMAS,
    MAX_REFINED_POINTS,
    MIN_SCAN_POINTS,
    Candidate,
    Screen,
    integrate_segments,
    locate,
)

# A 1 s wide encounter whose crossing is 0.1 s: refining a 10000 s window to
# crossing / 8 everywhere would take 800 000 points.
TAU, CROSSING, WINDOW = 1.0, 0.1, (-5000.0, 5000.0)


def bump(centre: float, tau: float = TAU) -> Callable[[float], float]:
    return lambda t: math.exp(-0.5 * ((t - centre) / tau) ** 2)


def whole_window(width: float = TAU) -> Screen:
    return lambda window: [Candidate(*window, CROSSING, width)]


@pytest.mark.parametrize("phase", np.linspace(0.0, 1.0, 7).tolist())
def test_an_encounter_down_to_the_detection_level_is_found(phase: float) -> None:
    # A second encounter DETECTION_LEVEL as high as the first, placed at
    # every phase of the detection grid (the worst case is midway between
    # two nodes), must still get its own segment.
    weak_centre = 1234.5 + phase * DETECTION_SIGMAS * TAU
    strong, weak = bump(-800.0), bump(weak_centre)
    found = locate(
        lambda t: strong(t) + DETECTION_LEVEL * weak(t), WINDOW, whole_window()
    )
    spans = [(found.times[lo], found.times[hi]) for lo, hi in found.segments]
    assert len(spans) == 2
    assert any(a <= weak_centre <= b for a, b in spans)


@pytest.mark.parametrize(
    "centre", np.random.default_rng(3).uniform(-4900.0, 4900.0, 20).tolist()
)
def test_an_encounter_anywhere_in_a_long_window_is_found_and_integrated(
    centre: float,
) -> None:
    rate = bump(centre)
    found = locate(rate, WINDOW, whole_window())
    assert len(found.segments) == 1
    assert found.times.size < 5000
    lo, hi = found.segments[0]
    assert np.max(np.diff(found.times[lo : hi + 1])) <= CROSSING / 8 * (1 + 1e-9)
    nc = integrate_segments(rate, found).nc
    assert nc == pytest.approx(TAU * math.sqrt(2 * math.pi), rel=1e-9)


def test_refinement_beyond_its_budget_is_coarser_not_an_error() -> None:
    # 400 s wide: crossing / 8 would need 3e5 points in the encounter.
    rate = bump(123.4, tau=40.0)
    found = locate(rate, WINDOW, lambda w: [Candidate(*w, 0.01, 40.0)])
    assert found.times.size <= MIN_SCAN_POINTS + MAX_REFINED_POINTS
    nc = integrate_segments(rate, found).nc
    assert nc == pytest.approx(40.0 * math.sqrt(2 * math.pi), rel=1e-9)


def test_detection_beyond_its_budget_raises_naming_the_window() -> None:
    with pytest.raises(PcConvergenceError, match=r"window \[-5000, 5000\] s"):
        locate(bump(0.0, 0.01), WINDOW, whole_window(0.01))


def test_rate_where_the_screen_ruled_it_negligible_raises() -> None:
    def screen(window: tuple[float, float]) -> list[Candidate]:
        return [Candidate(-10.0, 10.0, CROSSING, TAU)]

    with pytest.raises(PcConvergenceError, match="screen ruled it negligible"):
        locate(bump(10.0), WINDOW, screen)
