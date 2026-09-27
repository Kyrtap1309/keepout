from hypothesis import given, settings
from hypothesis import strategies as st
from pc_builders import conjunctions, rotate, rotations

from keepout._types import FloatArray
from keepout.conjunction import Conjunction
from keepout.pc import compute_pc, elrod, foster

SMALL = 1e-10


def close(a: float, b: float) -> bool:
    """The spec's 2-D tolerance: 1e-6 relative above 1e-10, 1e-15 absolute below."""
    if max(a, b) >= SMALL:
        return abs(a - b) <= 1e-6 * max(a, b)
    return abs(a - b) <= 1e-15


@given(conjunctions())
def test_pc_is_a_probability(conjunction: Conjunction) -> None:
    for method in ("foster", "elrod"):
        value = compute_pc(conjunction, method).value
        assert 0.0 <= value <= 1.0


@given(conjunctions())
def test_pc_does_not_depend_on_which_object_is_primary(
    conjunction: Conjunction,
) -> None:
    swapped = Conjunction(
        conjunction.secondary, conjunction.primary, conjunction.hard_body_radius
    )
    assert close(compute_pc(swapped).value, compute_pc(conjunction).value)


@given(conjunctions(), rotations())
def test_pc_is_invariant_under_rotation_of_the_scene(
    conjunction: Conjunction, rotation: FloatArray
) -> None:
    rotated = Conjunction(
        rotate(conjunction.primary, rotation),
        rotate(conjunction.secondary, rotation),
        conjunction.hard_body_radius,
    )
    assert close(compute_pc(rotated).value, compute_pc(conjunction).value)


@given(conjunctions())
def test_foster_and_elrod_agree(conjunction: Conjunction) -> None:
    plane, radius = conjunction.plane, conjunction.hard_body_radius
    by_elrod = elrod(plane, radius)
    assert by_elrod.diagnostics == ()
    assert close(by_elrod.value, foster(plane, radius).value)


@settings(max_examples=50)
@given(conjunctions(), st.floats(1.01, 3.0))
def test_pc_grows_with_the_hard_body(conjunction: Conjunction, factor: float) -> None:
    plane, radius = conjunction.plane, conjunction.hard_body_radius
    smaller = foster(plane, radius).value
    larger = foster(plane, radius * factor).value
    assert larger >= smaller * (1.0 - 1e-9)
