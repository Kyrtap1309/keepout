import math

import numpy as np
import pytest

from keepout.errors import CovarianceError
from keepout.pc._plane import SIGMA_FLOOR, plane_gaussian


def test_principal_axes_put_the_wide_axis_first() -> None:
    angle = math.radians(30.0)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    covariance = rotation @ np.diag([400.0, 25.0]) @ rotation.T
    miss = rotation @ np.array([3.0, 4.0])
    gaussian = plane_gaussian(miss, covariance, 10.0)
    assert (gaussian.su, gaussian.sv) == pytest.approx((20.0, 5.0), rel=1e-12)
    assert (abs(gaussian.mu), abs(gaussian.mv)) == pytest.approx((3.0, 4.0), rel=1e-12)
    assert gaussian.diagnostics == ()


def test_degenerate_axis_is_raised_to_the_floor() -> None:
    gaussian = plane_gaussian(np.array([1.0, 0.0]), np.diag([100.0, 0.0]), 20.0)
    assert gaussian.sv == pytest.approx(SIGMA_FLOOR * 20.0)
    assert len(gaussian.diagnostics) == 1
    assert gaussian.diagnostics[0].startswith("encounter-plane covariance: narrow-axis")


def test_zero_covariance_is_rejected() -> None:
    with pytest.raises(CovarianceError, match="no positive eigenvalue"):
        plane_gaussian(np.array([1.0, 0.0]), np.zeros((2, 2)), 20.0)


def test_band_misses_the_circle_far_from_the_mean() -> None:
    gaussian = plane_gaussian(np.array([0.0, 1000.0]), np.diag([1.0, 4.0]), 10.0)
    assert gaussian.band() is None


def test_chord_mass_keeps_precision_in_both_tails() -> None:
    for mv in (60.0, -60.0):
        gaussian = plane_gaussian(np.array([mv, 0.0]), np.diag([1.0, 4.0]), 10.0)
        mass = gaussian.chord_mass(np.array([10.0]))[0]
        # P(|v - mv| <= 10) for v ~ N(mv, 1) with mv = +-60: ~ upper tail at 50 sigma.
        expected = 0.5 * math.erfc(50.0 / math.sqrt(2.0))
        assert mass == pytest.approx(expected, rel=1e-12)
