import math

import pytest
from builders import head_on
from pc_builders import ANISOTROPIC, PlaneCase
from reference_pc import isotropic_pc, principal_axes_pc

from keepout.pc.foster import MIN_RTOL, foster


@pytest.mark.parametrize(
    ("miss", "sigma", "radius"),
    [
        (0.0, 10.0, 5.0),
        (30.0, 10.0, 5.0),
        (5.0, 1.0, 20.0),  # covariance much smaller than the hard body
        (200.0, 500.0, 10.0),
        (100.0, 10.0, 5.0),  # Pc ~ 1e-23
    ],
)
def test_isotropic_covariance_matches_the_rice_integral(
    miss: float, sigma: float, radius: float
) -> None:
    conjunction = head_on(miss, sigma, sigma, radius)
    result = foster(conjunction.plane, radius)
    assert result.method == "foster"
    assert result.value == pytest.approx(isotropic_pc(miss, sigma, radius), rel=1e-9)


def test_zero_miss_isotropic_has_a_closed_form() -> None:
    conjunction = head_on(0.0, 10.0, 10.0, 5.0)
    expected = 1.0 - math.exp(-(5.0**2) / (2.0 * 10.0**2))
    assert foster(conjunction.plane, 5.0).value == pytest.approx(expected, rel=1e-10)


def test_far_miss_underflows_to_exactly_zero_without_integrating() -> None:
    # 5e3 sigma along the wide axis: the 12-sigma band misses the hard body.
    conjunction = head_on(1e5, 20.0, 10.0, 5.0)
    result = foster(conjunction.plane, 5.0)
    assert result.value == 0.0
    assert result.diagnostics == ()
    assert result.details == {"error_estimate": 0.0, "evaluations": 0.0}


def test_huge_hard_body_gives_certain_collision() -> None:
    conjunction = head_on(10.0, 1.0, 1.0, 1000.0)
    assert foster(conjunction.plane, 1000.0).value == pytest.approx(1.0, abs=1e-12)


def test_details_report_the_quadrature() -> None:
    conjunction = head_on(30.0, 10.0, 20.0, 5.0)
    result = foster(conjunction.plane, 5.0)
    assert result.details["evaluations"] > 0
    # QUADPACK's absolute error estimate must honour the relative tolerance.
    assert 0.0 <= result.details["error_estimate"] <= 1e-10 * result.value


def test_quadrature_trouble_is_reported() -> None:
    # A line-like covariance at the smallest tolerance QUADPACK accepts:
    # round-off stops it from reaching the tolerance, and it says so.
    conjunction = head_on(3.0, 1e-3, 50.0, 10.0)
    result = foster(conjunction.plane, 10.0, rtol=MIN_RTOL * 1.1)
    (message,) = result.diagnostics
    assert message.startswith("Foster quadrature: The occurrence of roundoff error")
    assert "\n" not in message


@pytest.mark.parametrize(
    ("radius", "rtol", "message"),
    [
        (0.0, 1e-10, "hard_body_radius must be finite and > 0"),
        (-1.0, 1e-10, "hard_body_radius must be finite and > 0"),
        (5.0, 0.0, "rtol must be >="),
        (5.0, 1e-16, "rtol must be >="),
        (5.0, float("nan"), "rtol must be >="),
    ],
)
def test_bad_arguments_are_rejected(radius: float, rtol: float, message: str) -> None:
    conjunction = head_on(30.0, 10.0, 20.0, 5.0)
    with pytest.raises(ValueError, match=message):
        foster(conjunction.plane, radius, rtol=rtol)


@pytest.mark.parametrize(
    ("miss", "sigma_narrow", "sigma_wide", "radius"),
    [
        (0.05, 0.002, 35.0, 10.0),  # chord mass steps within 1e-3 rad of the edge
        (0.01, 0.002, 35.0, 10.0),
        (3.0, 0.001, 50.0, 10.0),
    ],
)
def test_line_like_covariance_is_integrated_accurately(
    miss: float, sigma_narrow: float, sigma_wide: float, radius: float
) -> None:
    # Miss along the narrow axis: the chord mass is a sharp step in u.
    conjunction = head_on(miss, sigma_narrow, sigma_wide, radius)
    expected = principal_axes_pc(0.0, miss, sigma_wide, sigma_narrow, radius)
    result = foster(conjunction.plane, radius)
    assert result.value == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize("case", ANISOTROPIC, ids=str)
def test_correlated_off_axis_covariance_matches_the_oracle(case: PlaneCase) -> None:
    result = foster(case.plane(), case.radius)
    expected = principal_axes_pc(case.mu, case.mv, case.su, case.sv, case.radius)
    assert result.value == pytest.approx(expected, rel=1e-8)
