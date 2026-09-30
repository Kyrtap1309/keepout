import pytest
from builders import head_on
from pc_builders import ANISOTROPIC, PlaneCase
from reference_pc import isotropic_pc, principal_axes_pc

from keepout.pc.elrod import MAX_ORDER, START_ORDER, elrod


@pytest.mark.parametrize(
    ("miss", "sigma", "radius"),
    [(0.0, 10.0, 5.0), (30.0, 10.0, 5.0), (5.0, 1.0, 20.0), (100.0, 10.0, 5.0)],
)
def test_isotropic_covariance_matches_the_rice_integral(
    miss: float, sigma: float, radius: float
) -> None:
    conjunction = head_on(miss, sigma, sigma, radius)
    result = elrod(conjunction.plane, radius)
    assert result.method == "elrod"
    assert result.value == pytest.approx(isotropic_pc(miss, sigma, radius), rel=1e-9)
    assert result.diagnostics == ()


def test_adaptive_order_resolves_a_thin_covariance() -> None:
    # Narrow axis 0.3 m against a 15 m hard body: order 64 is not enough.
    conjunction = head_on(5.0, 100.0, 0.3, 15.0)
    adaptive = elrod(conjunction.plane, 15.0)
    fixed = elrod(conjunction.plane, 15.0, order=START_ORDER)
    # Miss along the wide (x) axis.
    reference = principal_axes_pc(5.0, 0.0, 100.0, 0.3, 15.0)
    assert adaptive.value == pytest.approx(reference, rel=1e-8)
    assert fixed.value != pytest.approx(reference, rel=1e-8)
    assert adaptive.details["order"] > START_ORDER
    assert fixed.details["order"] == START_ORDER


def test_fixed_order_too_low_for_the_covariance_is_reported() -> None:
    # Covariance 0.1 m against a 20 m hard body: ~1300 nodes needed.
    conjunction = head_on(19.0, 0.1, 0.1, 20.0)
    result = elrod(conjunction.plane, 20.0, order=16)
    assert any("may be too low" in d for d in result.diagnostics)


def test_starting_order_follows_the_covariance() -> None:
    conjunction = head_on(19.0, 0.1, 0.1, 20.0)
    result = elrod(conjunction.plane, 20.0)
    assert result.details["order"] >= 2048
    assert result.value == pytest.approx(isotropic_pc(19.0, 0.1, 20.0), rel=1e-8)


def test_non_convergence_at_the_order_cap_is_reported() -> None:
    # 1 mm against 1 km is floored to 0.1 m; orders 32768 and 65536 still
    # differ by ~1e-9, so the cap is reached and reported.
    conjunction = head_on(999.0, 1e-3, 1e-3, 1000.0)
    result = elrod(conjunction.plane, 1000.0)
    assert result.details["order"] == MAX_ORDER
    assert any("did not converge" in d for d in result.diagnostics)
    assert result.value == pytest.approx(isotropic_pc(999.0, 0.1, 1000.0), rel=1e-8)


@pytest.mark.parametrize("order", [0, -64])
def test_order_must_be_positive(order: int) -> None:
    conjunction = head_on(30.0, 10.0, 20.0, 5.0)
    with pytest.raises(ValueError, match="order must be >= 1"):
        elrod(conjunction.plane, 5.0, order=order)


@pytest.mark.parametrize("case", ANISOTROPIC, ids=str)
def test_correlated_off_axis_covariance_matches_the_oracle(case: PlaneCase) -> None:
    result = elrod(case.plane(), case.radius)
    expected = principal_axes_pc(
        case.miss_u, case.miss_v, case.sigma_u, case.sigma_v, case.radius
    )
    assert result.value == pytest.approx(expected, rel=1e-8)
    assert result.diagnostics == ()
