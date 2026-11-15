import dataclasses

import numpy as np
import pytest
from builders import head_on

from keepout._types import FloatArray
from keepout.encounter import EncounterPlane
from keepout.figures import encounter_figure

PLANE = head_on(30.0, 100.0, 50.0, 10.0).plane


def test_encounter_ellipse_axes() -> None:
    figure = encounter_figure(PLANE, 10.0)
    assert figure.miss == (30.0, 0.0)
    one_sigma = figure.ellipses[0]
    # Centred at the miss vector: the half-widths are the sigmas.
    offset = one_sigma - np.array(figure.miss)
    assert np.abs(offset[:, 0]).max() == pytest.approx(100.0, rel=1e-3)
    assert np.abs(offset[:, 1]).max() == pytest.approx(50.0, rel=1e-3)
    assert np.linalg.norm(figure.hbr_circle, axis=1) == pytest.approx(10.0, abs=1e-9)


def test_ellipses_scale_with_sigma() -> None:
    figure = encounter_figure(PLANE, 10.0, points=37)
    assert len(figure.ellipses) == 3
    centre = np.array(figure.miss)
    for k, ellipse in enumerate(figure.ellipses, start=1):
        assert ellipse.shape == (37, 2)
        np.testing.assert_allclose(
            ellipse - centre, k * (figure.ellipses[0] - centre), atol=1e-9
        )
    assert figure.hbr_circle.shape == (37, 2)
    # Closed curves.
    np.testing.assert_allclose(figure.hbr_circle[0], figure.hbr_circle[-1], atol=1e-9)


def test_a_rotated_covariance_follows_its_axes() -> None:
    angle = np.radians(30.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    figure = encounter_figure(PLANE, 10.0)
    centre = np.array(figure.miss)
    rotated = encounter_figure(
        _with_covariance(rotation @ PLANE.covariance @ rotation.T), 10.0
    )
    np.testing.assert_allclose(
        np.sort(np.linalg.norm(rotated.ellipses[0] - centre, axis=1)),
        np.sort(np.linalg.norm(figure.ellipses[0] - centre, axis=1)),
        atol=1e-9,
    )
    # The longest axis turns by 30 degrees.
    offset = rotated.ellipses[0] - centre
    far = offset[np.argmax(np.linalg.norm(offset, axis=1))]
    assert abs(far[1] / far[0]) == pytest.approx(np.tan(angle), rel=1e-6)


@pytest.mark.parametrize(
    ("hbr", "points", "message"),
    [(0.0, 181, "hbr"), (float("nan"), 181, "hbr"), (10.0, 2, "points")],
)
def test_bad_arguments_are_errors(hbr: float, points: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        encounter_figure(PLANE, hbr, points=points)


def _with_covariance(covariance: FloatArray) -> EncounterPlane:
    return dataclasses.replace(PLANE, covariance=covariance)
