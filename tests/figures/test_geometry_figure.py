import numpy as np
import pytest
from builders import TYPICAL_HEAD_ON, object_state

from keepout.conjunction import Conjunction
from keepout.figures import geometry_figure
from keepout.frames import rtn_rotation

# An inclined primary with r and v on all three axes: its RTN matrix is far
# from the identity and not symmetric, so a transposed or missing rotation
# fails the tests.
PRIMARY_R = np.array([4000e3, -3000e3, 4500e3])
PRIMARY_V = np.array([2500.0, 6000.0, 1500.0])
ROTATION = rtn_rotation(PRIMARY_R, PRIMARY_V)
RELATIVE_POSITION = np.array([30.0, -40.0, 50.0])
RELATIVE_VELOCITY = np.array([100.0, -14000.0, 300.0])


def _conjunction(rtn_covariance: np.ndarray) -> Conjunction:
    """A conjunction with the given combined position covariance in the
    primary's RTN, all on the primary."""
    primary = object_state(
        PRIMARY_R,
        PRIMARY_V,
        ROTATION.T @ rtn_covariance @ ROTATION,
        object_id="PRI",
    )
    secondary = object_state(
        PRIMARY_R + RELATIVE_POSITION,
        PRIMARY_V + RELATIVE_VELOCITY,
        np.zeros((3, 3)),
        object_id="SEC",
    )
    return Conjunction(primary, secondary, 5.0)


CONJUNCTION = _conjunction(np.diag([30.0**2, 20.0**2, 10.0**2]))


def test_the_rotation_is_not_trivial() -> None:
    assert np.abs(ROTATION - np.eye(3)).max() > 0.5
    assert np.abs(ROTATION - ROTATION.T).max() > 0.5


def test_path_is_linear_through_tca() -> None:
    span, points = 2.0, 41
    figure = geometry_figure(CONJUNCTION, 10.0, span=span, points=points)
    velocity = ROTATION @ RELATIVE_VELOCITY
    assert figure.path.shape == (points, 3)
    np.testing.assert_allclose(figure.velocity, velocity, rtol=1e-12)
    np.testing.assert_allclose(
        figure.path[-1] - figure.path[0], 2.0 * span * velocity, rtol=1e-9
    )
    np.testing.assert_allclose(
        figure.path[points // 2], figure.at_tca, rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        figure.at_tca, ROTATION @ RELATIVE_POSITION, rtol=1e-9, atol=1e-9
    )
    # Straight: every step is the same.
    steps = np.diff(figure.path, axis=0)
    np.testing.assert_allclose(steps, np.broadcast_to(steps[0], steps.shape))
    assert figure.axes == ("R [m]", "T [m]", "N [m]")


def test_ellipsoid_extent() -> None:
    figure = geometry_figure(CONJUNCTION, 10.0)
    x, y, z = figure.ellipsoid
    centre = figure.at_tca
    assert x.shape == y.shape == z.shape
    assert x.ndim == 2
    # 3 sigma along each RTN axis, centred at the relative position at TCA.
    for surface, mid, sigma in zip((x, y, z), centre, (30.0, 20.0, 10.0), strict=True):
        assert (surface - mid).max() == pytest.approx(3.0 * sigma, rel=1e-3)
        assert (surface - mid).min() == pytest.approx(-3.0 * sigma, rel=1e-3)


def test_ellipsoid_follows_rotated_axes() -> None:
    angle = np.radians(30.0)
    turn = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    covariance = turn @ np.diag([30.0**2, 20.0**2, 10.0**2]) @ turn.T
    figure = geometry_figure(_conjunction(covariance), 10.0, points=73)
    offset = np.stack(figure.ellipsoid, axis=-1) - np.array(figure.at_tca)
    # Every surface point is at Mahalanobis distance 3.
    distance = np.einsum("...i,ij,...j->...", offset, np.linalg.inv(covariance), offset)
    np.testing.assert_allclose(distance, 9.0, rtol=1e-9)
    far = offset.reshape(-1, 3)[np.argmax(np.linalg.norm(offset, axis=-1))]
    assert abs(far[1] / far[0]) == pytest.approx(np.tan(angle), rel=1e-3)


def test_sphere_radius() -> None:
    figure = geometry_figure(CONJUNCTION, 7.5, points=25)
    x, y, z = figure.hbr_sphere
    assert x.shape == (25, 25)
    np.testing.assert_allclose(np.sqrt(x**2 + y**2 + z**2), 7.5, rtol=1e-12)


def test_unknown_covariance_shrinks_the_ellipsoid_to_tca() -> None:
    figure = geometry_figure(_conjunction(np.zeros((3, 3))), 10.0)
    for surface, mid in zip(figure.ellipsoid, figure.at_tca, strict=True):
        np.testing.assert_allclose(surface, mid)


def test_arrays_are_read_only() -> None:
    figure = geometry_figure(TYPICAL_HEAD_ON, 5.0)
    for array in (figure.path, *figure.ellipsoid, *figure.hbr_sphere):
        assert not array.flags.writeable


@pytest.mark.parametrize(
    ("hbr", "span", "points", "message"),
    [
        (0.0, 2.0, 41, "hbr"),
        (float("nan"), 2.0, 41, "hbr"),
        (10.0, 0.0, 41, "span"),
        (10.0, float("inf"), 41, "span"),
        (10.0, 2.0, 1, "points"),
        (10.0, 2.0, 40, "points"),
    ],
)
def test_bad_arguments_are_errors(
    hbr: float, span: float, points: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        geometry_figure(CONJUNCTION, hbr, span=span, points=points)
