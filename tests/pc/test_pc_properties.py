"""Invariants of 2-D Pc that any correct method satisfies."""

import dataclasses
import math

import numpy as np
from builders import rotate, rotations
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from pc_builders import PlaneCase, conjunctions, plane_cases
from reference_pc import principal_axes_pc

from keepout._types import FloatArray
from keepout.conjunction import Conjunction
from keepout.pc import compute_pc, elrod, foster
from keepout.pc.result import Method, PcResult

METHODS: tuple[Method, ...] = ("foster", "elrod")

# Geometric invariants hold far tighter than the 1e-6 CARA acceptance
# criterion: only the quadrature tolerance (1e-10) and round-off separate the
# two sides. Below TINY a relative comparison means nothing.
RTOL = 1e-8
TINY = 1e-300


def close(a: float, b: float, rtol: float = RTOL) -> bool:
    return abs(a - b) <= rtol * max(abs(a), abs(b)) + TINY


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
    return case.sv > 1e-3 * case.radius


@settings(max_examples=50)
@given(plane_cases())
def test_both_methods_match_the_independent_oracle(case: PlaneCase) -> None:
    assume(unfloored(case))
    expected = principal_axes_pc(case.mu, case.mv, case.su, case.sv, case.radius)
    assume(expected > 1e-250)
    for method in METHODS:
        assert close(on_plane(case, method).value, expected, rtol=1e-7)


@given(plane_cases())
def test_pc_is_bounded_by_the_peak_density(case: PlaneCase) -> None:
    # Pc is at most pi R^2 times the largest plane density, 1 / (2 pi su sv).
    bound = case.radius**2 / (2.0 * max(case.su, 1e-4 * case.radius) * case.sv)
    for method in METHODS:
        value = on_plane(case, method).value
        assert 0.0 <= value <= bound * (1.0 + 1e-9)


@given(plane_cases(), st.floats(1e-3, 1e3))
def test_pc_does_not_depend_on_the_unit_of_length(
    case: PlaneCase, scale: float
) -> None:
    assert close(on_plane(case, scale=scale).value, on_plane(case).value)


@given(plane_cases())
def test_mirrored_miss_gives_the_same_pc(case: PlaneCase) -> None:
    mirrored = dataclasses.replace(case, mu=-case.mu, mv=-case.mv)
    assert close(on_plane(mirrored).value, on_plane(case).value)


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
        dataclasses.replace(case, mu=case.mu * factor, mv=case.mv * factor), method
    ).value
    assert farther <= nearer * (1.0 + 1e-9)


@given(plane_cases(), st.sampled_from(METHODS))
def test_small_hard_body_sees_the_density_at_the_miss(
    case: PlaneCase, method: Method
) -> None:
    # R -> 0: Pc -> pi R^2 f(miss), with a relative error of order (R/sv)^2.
    small = dataclasses.replace(case, radius=1e-4 * case.sv)
    density = math.exp(-0.5 * ((case.mu / case.su) ** 2 + (case.mv / case.sv) ** 2)) / (
        2.0 * math.pi * case.su * case.sv
    )
    expected = math.pi * small.radius**2 * density
    assume(expected > 1e-250)
    assert close(on_plane(small, method).value, expected, rtol=1e-6)


@given(conjunctions())
def test_pc_does_not_depend_on_which_object_is_primary(
    conjunction: Conjunction,
) -> None:
    swapped = Conjunction(
        conjunction.secondary, conjunction.primary, conjunction.hard_body_radius
    )
    assert close(pc(swapped), pc(conjunction))


@given(conjunctions(), rotations())
def test_pc_is_invariant_under_rotation_of_the_scene(
    conjunction: Conjunction, rotation: FloatArray
) -> None:
    rotated = Conjunction(
        rotate(conjunction.primary, rotation),
        rotate(conjunction.secondary, rotation),
        conjunction.hard_body_radius,
    )
    assert close(pc(rotated), pc(conjunction))


@given(plane_cases())
def test_foster_and_elrod_agree(case: PlaneCase) -> None:
    by_elrod = on_plane(case, "elrod")
    assert not any("Elrod" in d for d in by_elrod.diagnostics)
    assert close(by_elrod.value, on_plane(case, "foster").value, rtol=1e-7)


def test_plane_case_conjunction_has_the_same_plane() -> None:
    case = PlaneCase(mu=3.0, mv=-4.0, su=20.0, sv=5.0, angle=0.3, radius=10.0)
    plane = case.conjunction().plane
    assert math.isclose(plane.miss_distance, 5.0, rel_tol=1e-9)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(plane.covariance), [25.0, 400.0], rtol=1e-9
    )
    assert math.isclose(pc(case.conjunction()), on_plane(case).value, rel_tol=1e-8)
