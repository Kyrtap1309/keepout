import dataclasses
import math
from collections.abc import Callable

import numpy as np
import pytest
from builders import TYPICAL_HEAD_ON

from keepout.encounter import EncounterPlane
from keepout.errors import CovarianceError, GeometryError
from keepout.pc import elrod, foster
from keepout.pc._plane import SIGMA_FLOOR, plane_gaussian, require_known_covariance
from keepout.pc.result import PcResult


def test_principal_axes_put_the_wide_axis_first() -> None:
    angle = math.radians(30.0)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    covariance = rotation @ np.diag([400.0, 25.0]) @ rotation.T
    miss = rotation @ np.array([3.0, 4.0])
    gaussian = plane_gaussian(miss, covariance, 10.0)
    assert (gaussian.sigma_u, gaussian.sigma_v) == pytest.approx((20.0, 5.0), rel=1e-12)
    assert (abs(gaussian.miss_u), abs(gaussian.miss_v)) == pytest.approx(
        (3.0, 4.0), rel=1e-12
    )
    assert gaussian.diagnostics == ()


def test_degenerate_axis_is_raised_to_the_floor() -> None:
    gaussian = plane_gaussian(np.array([1.0, 0.0]), np.diag([100.0, 0.0]), 20.0)
    assert gaussian.sigma_v == pytest.approx(SIGMA_FLOOR * 20.0)
    assert len(gaussian.diagnostics) == 1
    assert gaussian.diagnostics[0].startswith("encounter-plane covariance: narrow-axis")


def test_zero_covariance_is_rejected() -> None:
    with pytest.raises(CovarianceError, match="no positive eigenvalue"):
        plane_gaussian(np.array([1.0, 0.0]), np.zeros((2, 2)), 20.0)


def test_non_finite_inputs_are_rejected() -> None:
    with pytest.raises(GeometryError, match="miss vector"):
        plane_gaussian(np.array([math.nan, 0.0]), np.eye(2), 20.0)
    with pytest.raises(CovarianceError, match="must be finite"):
        plane_gaussian(np.array([1.0, 0.0]), np.diag([math.inf, 1.0]), 20.0)


@pytest.mark.parametrize("miss_v", [30.0, -30.0])
def test_chord_mass_keeps_precision_in_both_tails(miss_v: float) -> None:
    gaussian = plane_gaussian(np.array([miss_v, 0.0]), np.diag([1.0, 4.0]), 10.0)
    # P(|v| <= 10) for v ~ N(miss_v, 1) with miss_v = +-30: a tail 20 sigma out
    # (~2.8e-89), where a difference of erf values would cancel to zero. The
    # far edge, 40 sigma out, adds nothing representable.
    expected = 0.5 * math.erfc(20.0 / math.sqrt(2.0))
    assert expected > 1e-100
    assert gaussian.chord_mass(np.array([10.0]))[0] == pytest.approx(
        expected, rel=1e-12, abs=0.0
    )
    assert gaussian.chord_mass_at(10.0) == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_scalar_and_array_evaluations_agree() -> None:
    gaussian = plane_gaussian(np.array([3.0, 40.0]), np.diag([400.0, 25.0]), 30.0)
    half_chords = np.linspace(0.0, 30.0, 13)
    np.testing.assert_allclose(
        gaussian.chord_mass(half_chords),
        [gaussian.chord_mass_at(h) for h in half_chords],
        rtol=1e-14,
        atol=1e-300,
    )
    np.testing.assert_allclose(
        gaussian.density_u(half_chords),
        [gaussian.density_u_at(u) for u in half_chords],
        rtol=1e-14,
    )


def test_an_unknown_covariance_is_refused_by_name() -> None:
    require_known_covariance((), "foster")
    with pytest.raises(
        CovarianceError, match="foster needs both covariances; unknown: secondary"
    ):
        require_known_covariance(("secondary",), "foster")


@pytest.mark.parametrize("method", [foster, elrod])
def test_two_d_methods_refuse_a_plane_with_an_unknown_covariance(
    method: Callable[[EncounterPlane, float], PcResult],
) -> None:
    plane = dataclasses.replace(TYPICAL_HEAD_ON.plane, unknown_covariance=("primary",))
    with pytest.raises(CovarianceError, match="unknown: primary"):
        method(plane, 5.0)
