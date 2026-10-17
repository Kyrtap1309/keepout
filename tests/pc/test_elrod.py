import pytest
from builders import TYPICAL_HEAD_ON, head_on
from reference_pc import isotropic_pc, principal_axes_pc

from keepout.pc.elrod import MAX_ORDER, START_ORDER, elrod

CARA_ORDER = 64


def test_adaptive_order_resolves_a_thin_covariance() -> None:
    # Narrow axis 0.3 m against a 15 m hard body: order 64 is not enough.
    conjunction = head_on(5.0, 100.0, 0.3, 15.0)
    adaptive = elrod(conjunction.plane, 15.0)
    fixed = elrod(conjunction.plane, 15.0, order=CARA_ORDER)
    # Miss along the wide (x) axis.
    reference = principal_axes_pc(5.0, 0.0, 100.0, 0.3, 15.0)
    assert adaptive.value == pytest.approx(reference, rel=1e-8)
    assert fixed.value != pytest.approx(reference, rel=1e-8)
    assert adaptive.details["order"] > START_ORDER
    assert fixed.details["order"] == CARA_ORDER


def test_adaptive_orders_are_nested() -> None:
    # Refining reuses the old terms as the even nodes of the next order, so
    # the adaptive value must be the fixed rule of its final order evaluated
    # from scratch, bit for bit: the angles 2j pi / (2n + 2) and j pi / (n + 1)
    # differ only by powers of two.
    plane = head_on(5.0, 100.0, 0.3, 15.0).plane
    adaptive = elrod(plane, 15.0)
    order = int(adaptive.details["order"])
    assert order > START_ORDER  # at least one refinement
    assert adaptive.value == elrod(plane, 15.0, order=order).value


def test_fixed_order_too_low_for_the_covariance_is_reported() -> None:
    # Covariance 0.1 m against a 20 m hard body: ~1300 nodes needed.
    conjunction = head_on(19.0, 0.1, 0.1, 20.0)
    result = elrod(conjunction.plane, 20.0, order=16)
    assert any("may be too low" in d for d in result.diagnostics)


def test_starting_order_follows_the_covariance() -> None:
    conjunction = head_on(19.0, 0.1, 0.1, 20.0)
    result = elrod(conjunction.plane, 20.0)
    # 2 pi R / sigma = 1257 nodes, rounded up to the nested order 2^11 - 1.
    assert result.details["order"] >= 2047
    assert result.value == pytest.approx(isotropic_pc(19.0, 0.1, 20.0), rel=1e-8)


def test_non_convergence_at_the_order_cap_is_reported() -> None:
    # 1 mm against 1 km is floored to 0.1 m; orders 32767 and 65535 still
    # differ by more than ELROD_RTOL, so the cap is reached and reported.
    conjunction = head_on(999.0, 1e-3, 1e-3, 1000.0)
    result = elrod(conjunction.plane, 1000.0)
    assert result.details["order"] == MAX_ORDER
    assert any("did not converge" in d for d in result.diagnostics)
    assert result.value == pytest.approx(isotropic_pc(999.0, 0.1, 1000.0), rel=1e-8)


@pytest.mark.parametrize("order", [0, -64])
def test_order_must_be_positive(order: int) -> None:
    with pytest.raises(ValueError, match="order must be at least 1"):
        elrod(TYPICAL_HEAD_ON.plane, 5.0, order=order)
