import dataclasses
import math

import numpy as np
import pytest
from pc_builders import ANISOTROPIC, PlaneCase
from scipy import optimize, special

from keepout._types import FloatArray
from keepout.encounter import EncounterPlane, Role
from keepout.errors import CovarianceError
from keepout.pc import foster
from keepout.pc.maxpc import alfano, dilution, frisbee


def with_unknown(case: PlaneCase, *roles: Role) -> EncounterPlane:
    return dataclasses.replace(case.plane(), unknown_covariance=roles)


def pc_with(plane: EncounterPlane, covariance: FloatArray, radius: float) -> float:
    return foster(
        dataclasses.replace(plane, covariance=covariance, unknown_covariance=()),
        radius,
    ).value


def added_along_miss(plane: EncounterPlane, variance: float) -> FloatArray:
    """The plane covariance with ``variance`` added along the miss direction."""
    if variance == 0.0:
        return plane.covariance
    unit = plane.miss_vector / plane.miss_distance
    return plane.covariance + variance * np.outer(unit, unit)


@pytest.mark.parametrize("case", ANISOTROPIC[:4])
def test_frisbee_is_the_maximum_over_added_covariances(case: PlaneCase) -> None:
    plane = with_unknown(case, "secondary")
    result = frisbee(plane, case.radius)
    assert result.method == "frisbee"
    assert "upper bound" in result.diagnostics[0]
    rng = np.random.default_rng(7)
    scale = max(case.sigma_u, plane.miss_distance, 1.0)
    for _ in range(200):
        factor = rng.normal(size=(2, 2)) * scale * rng.uniform(0.0, 3.0)
        added = factor @ factor.T
        assert pc_with(plane, plane.covariance + added, case.radius) <= result.value * (
            1.0 + 1e-9
        )
    # ... and is reached, by the variance it reports along the miss.
    along = added_along_miss(plane, result.details["added_variance_m2"])
    assert result.value == pytest.approx(pc_with(plane, along, case.radius), rel=1e-12)


def test_frisbee_adds_the_small_body_optimum_along_the_miss() -> None:
    case = PlaneCase(400.0, 0.0, 20.0, 10.0, 0.0, 5.0)
    plane = with_unknown(case, "primary")
    result = frisbee(plane, case.radius)
    k2 = (400.0 / 20.0) ** 2
    analytic = 400.0**2 * (k2 - 1.0) / k2
    assert result.details["mahalanobis"] == pytest.approx(20.0)
    assert result.details["added_variance_m2"] == pytest.approx(analytic)
    pc_analytic = pc_with(plane, np.diag([20.0**2 + analytic, 10.0**2]), 5.0)
    assert result.details["pc_analytic"] == pytest.approx(pc_analytic, rel=1e-12)
    along = np.diag([20.0**2 + result.details["added_variance_m2"], 10.0**2])
    assert result.value == pytest.approx(pc_with(plane, along, 5.0), rel=1e-12)
    assert result.value >= result.details["pc_analytic"]


def test_frisbee_with_a_miss_inside_one_sigma_adds_nothing() -> None:
    case = PlaneCase(5.0, 0.0, 20.0, 10.0, 0.0, 5.0)
    result = frisbee(with_unknown(case, "secondary"), 5.0)
    assert result.details["added_variance_m2"] == 0.0
    assert "zero added covariance" in result.diagnostics[-1]


ALONG_MISS_RTOL = 1e-5
"""How far below the true maximum :func:`along_miss_grid_max` can fall.

Its nearest node is at most half a step (0.0015 decades) from the optimum,
where ``ln Pc`` curves by about 1/2 per ``(ln s)^2`` (small-body limit:
``-ln(w) / 2 - d^2 / (2 w)``, ``w`` the variance along the miss): the
node's Pc is lower by ~``0.5 * 0.5 * (0.0015 ln 10)^2 = 3e-6``. Frisbee's own
bounded search converges to 1e-9 decades and to within
:data:`~keepout.pc.maxpc.SEARCH_GAIN_RTOL` (1e-8) of its maximum."""


def along_miss_grid_max(plane: EncounterPlane, radius: float) -> float:
    """Independent dense sweep of the added variance along the miss."""
    return max(
        pc_with(plane, added_along_miss(plane, s), radius)
        for s in [0.0, *10.0 ** np.linspace(-4.0, 8.0, 4001)]
    )


@pytest.mark.parametrize(
    "case",
    [
        PlaneCase(15.0, 0.0, 1.0, 1.0, 0.0, 10.0),
        PlaneCase(12.0, 0.0, 0.5, 0.5, 0.0, 10.0),
        PlaneCase(10.5, 0.0, 0.1, 0.1, 0.0, 10.0),
    ],
)
def test_frisbee_is_an_upper_bound_for_a_large_hard_body(case: PlaneCase) -> None:
    plane = with_unknown(case, "secondary")
    result = frisbee(plane, case.radius)
    grid_max = along_miss_grid_max(plane, case.radius)
    assert grid_max * (1.0 - 1e-9) <= result.value
    assert result.value == pytest.approx(grid_max, rel=ALONG_MISS_RTOL)


@pytest.mark.parametrize(
    "case",
    [
        PlaneCase(15.0, 0.0, 1.0, 1.0, 0.0, 10.0),
        PlaneCase(12.0, 0.0, 0.5, 0.5, 0.0, 10.0),
        PlaneCase(10.5, 0.0, 0.1, 0.1, 0.0, 10.0),
    ],
)
def test_frisbee_for_a_large_hard_body_beats_the_analytic_variance(
    case: PlaneCase,
) -> None:
    result = frisbee(with_unknown(case, "secondary"), case.radius)
    assert result.value > result.details["pc_analytic"] * (1.0 + 1e-3)


def test_frisbee_with_a_zero_miss_adds_nothing() -> None:
    case = PlaneCase(0.0, 0.0, 20.0, 10.0, 0.0, 5.0)
    plane = with_unknown(case, "secondary")
    result = frisbee(plane, 5.0)
    assert result.details["added_variance_m2"] == 0.0
    assert result.value == pc_with(plane, plane.covariance, 5.0)


@pytest.mark.parametrize("roles", [(), ("primary", "secondary")])
def test_frisbee_needs_exactly_one_unknown_covariance(roles: tuple[Role, ...]) -> None:
    with pytest.raises(ValueError, match="exactly one unknown"):
        frisbee(with_unknown(ANISOTROPIC[0], *roles), 5.0)


SCALE_GRID_RTOL = 2e-3
"""How far below the true maximum the scale-factor grid of
``test_dilution_is_the_maximum_over_scale_factors`` can fall.

Its nearest node is at most half a step (0.025 decades) from the optimum,
where ``ln Pc`` curves by about 1 per ``(ln k)^2`` (small-body limit:
``-ln k - K^2 / (2 k)`` at ``k = K^2 / 2``): the node's Pc is lower by
~``0.5 * (0.025 ln 10)^2 = 1.7e-3``. Dilution's own bounded search converges
to 1e-9 decades and to within :data:`~keepout.pc.maxpc.SEARCH_GAIN_RTOL`."""


@pytest.mark.parametrize("case", ANISOTROPIC[:4])
def test_dilution_is_the_maximum_over_scale_factors(case: PlaneCase) -> None:
    plane = case.plane()
    result = dilution(plane, case.radius)
    assert result.method == "dilution"
    grid_max = max(
        pc_with(plane, 10.0**k * plane.covariance, case.radius)
        for k in np.linspace(-6.0, 6.0, 241)
    )
    assert grid_max * (1.0 - 1e-9) <= result.value
    assert result.value == pytest.approx(grid_max, rel=SCALE_GRID_RTOL)


@pytest.mark.parametrize("case", ANISOTROPIC[:4])
def test_dilution_reports_the_nominal_pc(case: PlaneCase) -> None:
    plane = case.plane()
    assert dilution(plane, case.radius).details["pc_nominal"] == (
        foster(plane, case.radius).value
    )


def test_a_large_covariance_is_diluted() -> None:
    wide = dilution(PlaneCase(200.0, 0.0, 5000.0, 3000.0, 0.0, 10.0).plane(), 10.0)
    assert wide.details["diluted"] == 1.0
    assert wide.details["scale_factor"] < 1.0
    assert wide.value > wide.details["pc_nominal"]


def test_a_small_covariance_is_not_diluted() -> None:
    tight = dilution(PlaneCase(200.0, 0.0, 20.0, 10.0, 0.0, 10.0).plane(), 10.0)
    assert tight.details["diluted"] == 0.0
    assert tight.details["scale_factor"] > 1.0


def test_dilution_with_the_miss_inside_the_hard_body_is_one() -> None:
    result = dilution(PlaneCase(3.0, 0.0, 50.0, 20.0, 0.0, 10.0).plane(), 10.0)
    assert result.value == 1.0
    assert result.details["scale_factor"] == 0.0
    assert "inside the hard body" in result.diagnostics[-1]


def test_dilution_at_the_optimal_scale_reports_scale_one() -> None:
    # K^2 = 2: the small-body optimum k = K^2 / 2 is the nominal covariance.
    plane = PlaneCase(math.sqrt(2.0) * 100.0, 0.0, 100.0, 50.0, 0.0, 0.1).plane()
    result = dilution(plane, 0.1)
    assert result.value == result.details["pc_nominal"] > 0.0
    assert result.details["scale_factor"] == 1.0
    assert result.details["diluted"] == 0.0


def test_dilution_needs_both_covariances() -> None:
    with pytest.raises(CovarianceError, match="unknown: secondary"):
        dilution(with_unknown(ANISOTROPIC[0], "secondary"), 5.0)


def line_pc(d: float, radius: float, sigma: float) -> float:
    s = math.sqrt(2.0) * sigma
    return 0.5 * float(special.erf((radius + d) / s) + special.erf((radius - d) / s))


@pytest.mark.parametrize(("d", "radius"), [(100.0, 10.0), (1000.0, 20.0), (15.0, 10.0)])
def test_alfano_maximum_matches_a_numerical_search(d: float, radius: float) -> None:
    plane = PlaneCase(d, 0.0, 1.0, 1.0, 0.0, radius).plane()
    result = alfano(plane, radius)
    search = optimize.minimize_scalar(
        lambda log_s: -line_pc(d, radius, 10.0**log_s),
        bounds=(-3.0, 6.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    assert result.value == pytest.approx(-search.fun, rel=1e-9)
    assert result.details["sigma_m"] == pytest.approx(10.0**search.x, rel=1e-4)


def test_alfano_bounds_every_covariance() -> None:
    plane = PlaneCase(100.0, 0.0, 1.0, 1.0, 0.0, 10.0).plane()
    bound = alfano(plane, 10.0).value
    rng = np.random.default_rng(3)
    for _ in range(300):
        factor = rng.normal(size=(2, 2)) * 10.0 ** rng.uniform(-1.0, 3.0)
        covariance = factor @ factor.T + 1e-6 * np.eye(2)
        assert pc_with(plane, covariance, 10.0) <= bound * (1.0 + 1e-9)


def test_alfano_far_miss_follows_the_small_body_limit() -> None:
    plane = PlaneCase(1e5, 0.0, 1.0, 1.0, 0.0, 1.0).plane()
    limit = 1.0 / 1e5 * math.sqrt(2.0 / (math.pi * math.e))
    assert alfano(plane, 1.0).value == pytest.approx(limit, rel=1e-8, abs=0.0)


def test_alfano_miss_inside_the_hard_body_is_one() -> None:
    result = alfano(PlaneCase(3.0, 0.0, 1.0, 1.0, 0.0, 10.0).plane(), 10.0)
    assert result.value == 1.0
    assert result.details["sigma_m"] == 0.0


def test_alfano_miss_on_the_hard_body_edge_is_one_half() -> None:
    result = alfano(PlaneCase(10.0, 0.0, 1.0, 1.0, 0.0, 10.0).plane(), 10.0)
    assert result.value == 0.5
    assert result.details["sigma_m"] == 0.0
    assert "inside the hard body" not in result.diagnostics[-1]
    assert "on the hard-body edge" in result.diagnostics[-1]
