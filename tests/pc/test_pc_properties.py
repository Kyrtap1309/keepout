"""Invariants of 2-D Pc that any correct method satisfies."""

import dataclasses
import math

import numpy as np
import pytest
from builders import TINY, rotate, rotations
from builders import approx as relative_approx
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from pc_builders import PlaneCase, conjunctions, plane_cases
from reference_pc import principal_axes_pc
from scipy import special

from keepout._types import FloatArray
from keepout.conjunction import Conjunction
from keepout.pc import compute_pc, elrod, foster
from keepout.pc._plane import SIGMA_FLOOR
from keepout.pc.result import Method, PcResult

METHODS: tuple[Method, ...] = ("foster", "elrod")
EACH_METHOD = pytest.mark.parametrize("method", METHODS)

# Geometric invariants hold far tighter than the 1e-6 CARA acceptance
# criterion: only the quadrature tolerance (1e-10) and round-off separate the
# two sides.
RTOL = 1e-8


def approx(expected: float, rtol: float = RTOL) -> object:
    return relative_approx(expected, rel=rtol)


def pc(conjunction: Conjunction, method: Method = "foster") -> float:
    return compute_pc(conjunction, method).value


def on_plane(
    case: PlaneCase, method: Method = "foster", scale: float = 1.0
) -> PcResult:
    """The method run on the exact plane of ``case``."""
    kernel = foster if method == "foster" else elrod
    return kernel(case.plane(scale), scale * case.radius)


def unfloored(case: PlaneCase) -> bool:
    """Neither plane sigma is raised to keepout's floor (the oracle has none)."""
    return case.sigma_v > 1e-3 * case.radius


@settings(max_examples=50)
@given(plane_cases(), st.sampled_from(METHODS))
def test_both_methods_match_the_independent_oracle(
    case: PlaneCase, method: Method
) -> None:
    assume(unfloored(case))
    expected = principal_axes_pc(
        case.miss_u, case.miss_v, case.sigma_u, case.sigma_v, case.radius
    )
    assume(expected > 1e-250)
    assert on_plane(case, method).value == approx(expected, rtol=1e-7)


def interval_mass(low: float, high: float, sigma: float) -> float:
    """Mass of ``N(0, sigma^2)`` on ``[low, high]``, accurate in either tail."""
    if low > 0.0:
        return float(special.ndtr(-low / sigma) - special.ndtr(-high / sigma))
    return float(special.ndtr(high / sigma) - special.ndtr(low / sigma))


@settings(max_examples=50)
@given(plane_cases(), st.sampled_from(METHODS))
def test_pc_is_bounded_by_the_peak_density_and_the_strips(
    case: PlaneCase, method: Method
) -> None:
    """Pc is at most pi R^2 times the peak plane density 1 / (2 pi sigma_u
    sigma_v), and at most the mass of either strip ``|u - miss_u| <= R`` and
    ``|v - miss_v| <= R``, which contain the disk."""
    floor = SIGMA_FLOOR * case.radius  # keepout raises smaller sigmas to it
    sigma_u, sigma_v = max(case.sigma_u, floor), max(case.sigma_v, floor)
    radius = case.radius
    bound = min(
        radius**2 / (2.0 * sigma_u * sigma_v),
        interval_mass(case.miss_u - radius, case.miss_u + radius, sigma_u),
        interval_mass(case.miss_v - radius, case.miss_v + radius, sigma_v),
    )
    value = on_plane(case, method).value
    assert 0.0 <= value <= bound * (1.0 + 1e-9) + TINY


@EACH_METHOD
@given(plane_cases(), st.floats(1e-3, 1e3))
def test_pc_does_not_depend_on_the_unit_of_length(
    method: Method, case: PlaneCase, scale: float
) -> None:
    assert on_plane(case, method, scale=scale).value == approx(
        on_plane(case, method).value
    )


@EACH_METHOD
@given(plane_cases())
def test_mirrored_miss_gives_the_same_pc(method: Method, case: PlaneCase) -> None:
    mirrored = dataclasses.replace(case, miss_u=-case.miss_u, miss_v=-case.miss_v)
    assert on_plane(mirrored, method).value == approx(on_plane(case, method).value)


@settings(max_examples=50)
@given(plane_cases(), st.floats(1.01, 3.0), st.sampled_from(METHODS))
def test_pc_grows_with_the_hard_body(
    case: PlaneCase, factor: float, method: Method
) -> None:
    smaller = on_plane(case, method).value
    assume(1e-12 < smaller < 1.0 - 1e-6)  # room to grow, above round-off
    larger = on_plane(
        dataclasses.replace(case, radius=case.radius * factor), method
    ).value
    assert larger > smaller


@settings(max_examples=50)
@given(plane_cases(), st.floats(1.01, 3.0), st.sampled_from(METHODS))
def test_pc_falls_as_the_miss_grows(
    case: PlaneCase, factor: float, method: Method
) -> None:
    # Anderson's theorem: a centred Gaussian's mass in a centred disk falls
    # as the disk moves out along any fixed direction.
    nearer = on_plane(case, method).value
    farther = on_plane(
        dataclasses.replace(
            case, miss_u=case.miss_u * factor, miss_v=case.miss_v * factor
        ),
        method,
    ).value
    assert farther <= nearer * (1.0 + 1e-9)


@given(plane_cases(), st.sampled_from(METHODS))
def test_small_hard_body_sees_the_density_at_the_miss(
    case: PlaneCase, method: Method
) -> None:
    # R -> 0: Pc -> pi R^2 f(miss), with a relative error of order (R/sigma_v)^2.
    small = dataclasses.replace(case, radius=1e-4 * case.sigma_v)
    density = math.exp(
        -0.5 * ((case.miss_u / case.sigma_u) ** 2 + (case.miss_v / case.sigma_v) ** 2)
    ) / (2.0 * math.pi * case.sigma_u * case.sigma_v)
    expected = math.pi * small.radius**2 * density
    assume(expected > 1e-250)
    assert on_plane(small, method).value == approx(expected, rtol=1e-6)


@EACH_METHOD
@given(conjunctions())
def test_pc_does_not_depend_on_which_object_is_primary(
    method: Method, conjunction: Conjunction
) -> None:
    swapped = Conjunction(
        conjunction.secondary, conjunction.primary, conjunction.hard_body_radius
    )
    assert pc(swapped, method) == approx(pc(conjunction, method))


@EACH_METHOD
@given(conjunctions(), rotations())
def test_pc_is_invariant_under_rotation_of_the_scene(
    method: Method, conjunction: Conjunction, rotation: FloatArray
) -> None:
    rotated = Conjunction(
        rotate(conjunction.primary, rotation),
        rotate(conjunction.secondary, rotation),
        conjunction.hard_body_radius,
    )
    assert pc(rotated, method) == approx(pc(conjunction, method))


@given(plane_cases())
def test_foster_and_elrod_agree(case: PlaneCase) -> None:
    by_elrod = on_plane(case, "elrod")
    assert not any("Elrod" in d for d in by_elrod.diagnostics)
    assert by_elrod.value == approx(on_plane(case, "foster").value, rtol=1e-7)


def test_plane_case_conjunction_has_the_same_plane() -> None:
    case = PlaneCase(
        miss_u=3.0, miss_v=-4.0, sigma_u=20.0, sigma_v=5.0, angle=0.3, radius=10.0
    )
    plane = case.conjunction().plane
    assert math.isclose(plane.miss_distance, 5.0, rel_tol=1e-9)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(plane.covariance), [25.0, 400.0], rtol=1e-9
    )
    assert math.isclose(pc(case.conjunction()), on_plane(case).value, rel_tol=1e-8)
