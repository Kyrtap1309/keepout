"""Both 2-D methods against the independent reference integrals."""

from collections.abc import Callable

import pytest
from builders import approx, head_on
from pc_builders import ANISOTROPIC, PlaneCase
from reference_pc import isotropic_pc, principal_axes_pc

from keepout.encounter import EncounterPlane
from keepout.pc import PcResult, elrod, foster

TwoD = Callable[[EncounterPlane, float], PcResult]

METHODS = pytest.mark.parametrize("method", [foster, elrod], ids=["foster", "elrod"])


@METHODS
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
    method: TwoD, miss: float, sigma: float, radius: float
) -> None:
    conjunction = head_on(miss, sigma, sigma, radius)
    result = method(conjunction.plane, radius)
    assert result.method == method.__name__
    assert result.value == approx(isotropic_pc(miss, sigma, radius), rel=1e-9)
    assert result.diagnostics == ()


@METHODS
@pytest.mark.parametrize("case", ANISOTROPIC, ids=str)
def test_correlated_off_axis_covariance_matches_the_oracle(
    method: TwoD, case: PlaneCase
) -> None:
    result = method(case.plane(), case.radius)
    expected = principal_axes_pc(
        case.miss_u, case.miss_v, case.sigma_u, case.sigma_v, case.radius
    )
    assert result.value == approx(expected, rel=1e-8)
    assert result.diagnostics == ()
