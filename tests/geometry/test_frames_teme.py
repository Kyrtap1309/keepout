"""TEME (the SGP4 output frame) to GCRF."""

from typing import cast

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import (
    GCRS,
    TEME,
    CartesianDifferential,
    CartesianRepresentation,
)
from astropy.time import Time
from astropy.utils import iers
from builders import EPOCH, LEO_R, LEO_V
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from keepout import frames
from keepout._types import FloatArray
from keepout.frames import teme_to_gcrf, to_gcrf, to_gcrf_all
from keepout.state import StateVector

# Vallado, Crawford, Hujsak and Kelso, "Revisiting Spacetrack Report #3",
# AIAA 2006-6753, Rev. 2, Appendix C: the TEME state at the epoch of
# Vallado's Example 3-15 (Fundamentals of Astrodynamics and Applications,
# 4th ed.), whose GCRF state is given there.
VALLADO_EPOCH = Time("2004-04-06T07:51:28.386009", scale="utc")
VALLADO_TEME_R = np.array([5094.18016210, 6127.64465950, 6380.34453270]) * 1e3
VALLADO_TEME_V = np.array([-4.746131487, 0.785818041, 5.531931288]) * 1e3
VALLADO_GCRF_R = np.array([5102.508958, 6123.011401, 6378.136928]) * 1e3
VALLADO_GCRF_V = np.array([-4.743220157, 0.790536497, 5.533755272]) * 1e3

positions = arrays(np.float64, (3,), elements=st.floats(min_value=-5e7, max_value=5e7))
velocities = arrays(np.float64, (3,), elements=st.floats(min_value=-1e4, max_value=1e4))
mjds = st.floats(min_value=36934.0, max_value=62135.0)  # 1960 .. 2028-12-30


def astropy_teme_to_gcrs(
    epochs: Time, r: FloatArray, v: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """astropy's TEME -> GCRS of states given as rows of ``r`` and ``v``."""
    representation = CartesianRepresentation(
        r.T * u.m, differentials=CartesianDifferential(v.T * u.m / u.s)
    )
    # astropy's inferred types are a union; both are Cartesian here.
    cartesian = cast(
        CartesianRepresentation,
        TEME(representation, obstime=epochs)
        .transform_to(GCRS(obstime=epochs))
        .cartesian,
    )
    velocity = cast(CartesianDifferential, cartesian.differentials["s"])
    return (
        cartesian.xyz.to_value(u.m).T,
        velocity.d_xyz.to_value(u.m / u.s).T,
    )


def test_teme_matches_vallado_2006() -> None:
    # Residuals of 3 mm and 0.5 mm/s: the reference values come with their
    # own Earth-orientation data and corrections; keepout needs none (see
    # test_teme_needs_no_earth_orientation_data).
    r, v = teme_to_gcrf(VALLADO_EPOCH, VALLADO_TEME_R, VALLADO_TEME_V)
    np.testing.assert_allclose(r, VALLADO_GCRF_R, rtol=0, atol=0.01)
    np.testing.assert_allclose(v, VALLADO_GCRF_V, rtol=0, atol=1e-3)
    state = to_gcrf(StateVector(VALLADO_EPOCH, VALLADO_TEME_R, VALLADO_TEME_V, "TEME"))
    assert state.frame == "GCRF"
    assert state.epoch is VALLADO_EPOCH
    np.testing.assert_array_equal(state.r, r)
    np.testing.assert_array_equal(state.v, v)


def test_teme_matches_astropy_over_decades_and_orbits() -> None:
    # astropy goes TEME -> ITRS -> GCRS with IERS Earth-orientation data;
    # keepout rotates directly. The ~1e-10 |r| left is astropy leaving out
    # the TIO locator s' (~12 uas in 2026) on TEME -> ITRS but not back.
    rng = np.random.default_rng(5)
    count = 500
    epochs = Time(rng.uniform(42000.0, 61400.0, count), format="mjd", scale="utc")
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    r = directions * rng.uniform(6.5e6, 4.3e7, size=(count, 1))
    v = rng.normal(size=(count, 3)) * 4e3
    expected_r, expected_v = astropy_teme_to_gcrs(epochs, r, v)
    got_r, got_v = teme_to_gcrf(epochs, r, v)
    error = np.linalg.norm(got_r - expected_r, axis=1)
    assert np.all(error <= 1e-10 * np.linalg.norm(r, axis=1))
    np.testing.assert_allclose(got_v, expected_v, rtol=0, atol=1e-5)


def test_teme_needs_no_earth_orientation_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # TEME -> GCRF is GMST82 - ERA about z and the GCRS -> CIRS rotation:
    # polar motion cancels and UT1 - UTC changes the result by < 0.3 mm.
    # So no IERS table is read, and any epoch converts offline.
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("IERS data must not be needed")

    monkeypatch.setattr(iers.IERS_Auto, "open", refuse)
    monkeypatch.setattr(iers.IERS_B, "open", refuse)
    epochs = Time(["1961-01-01", "2028-12-30"], scale="utc")
    r, v = teme_to_gcrf(epochs, [LEO_R, LEO_R], [LEO_V, LEO_V])
    assert np.all(np.isfinite(r))
    assert np.all(np.isfinite(v))


@given(mjds, positions, velocities)
# Found by Hypothesis: the frame turns at 1.04e-11 rad/s on 1990-12-02.
@example(48227.0, np.full(3, 14.0), np.zeros(3))
def test_teme_to_gcrf_is_a_rotation_plus_its_slow_rate(
    mjd: float, r: FloatArray, v: FloatArray
) -> None:
    epoch = Time(mjd, format="mjd", scale="utc")
    got_r, got_v = teme_to_gcrf(epoch, r, v)
    scale = float(np.linalg.norm(r))
    assert np.linalg.norm(got_r) == pytest.approx(scale, rel=1e-14, abs=1e-8)
    # The frame turns at up to 1.11e-11 rad/s (precession and nutation,
    # measured over 1960-2028): that is all |v| may change by.
    assert abs(np.linalg.norm(got_v) - np.linalg.norm(v)) <= 1.2e-11 * scale + 1e-11


def test_the_velocity_is_the_rate_of_the_gcrf_position() -> None:
    # A point fixed in TEME moves in GCRF as the frame precesses and nutates
    # (~1e-11 rad/s): v_gcrf = M v + dM/dt r, not M v alone. The reference
    # is a five-point derivative of GCRF positions; both carry ~2e-9 m/s of
    # rounding at GEO (GMST and ERA are angles of order 2 pi).
    rng = np.random.default_rng(3)
    epochs = Time(rng.uniform(40000.0, 62000.0, 10), format="mjd", scale="utc")
    step = 1800.0
    offsets = np.array([-2.0, -1.0, 1.0, 2.0]) * step * u.s
    still = np.zeros(3)
    for epoch in epochs:
        direction = rng.normal(size=3)
        r = 4.2e7 * direction / np.linalg.norm(direction)
        moved, _ = teme_to_gcrf(epoch + offsets, r, still)
        derivative = (moved[0] - 8.0 * moved[1] + 8.0 * moved[2] - moved[3]) / (
            12.0 * step
        )
        _, v = teme_to_gcrf(epoch, r, still)
        assert np.linalg.norm(v) > 1e-5
        np.testing.assert_allclose(v, derivative, rtol=0, atol=1e-8)


def test_many_states_at_many_epochs_convert_in_one_call() -> None:
    epochs = EPOCH + np.array([0.0, 1.0, 2.0]) * u.day
    r = np.array([LEO_R, [0.0, 7e6, 1e5], [-4.2e7, 1e3, 0.0]])
    v = np.array([LEO_V, [-7e3, 0.0, 1e3], [0.0, -3075.0, 0.0]])
    got_r, got_v = teme_to_gcrf(epochs, r, v)
    assert got_r.shape == got_v.shape == (3, 3)
    for i in range(3):
        one_r, one_v = teme_to_gcrf(epochs[i], r[i], v[i])
        assert one_r.shape == one_v.shape == (3,)
        np.testing.assert_allclose(got_r[i], one_r, rtol=0, atol=1e-8)
        np.testing.assert_allclose(got_v[i], one_v, rtol=0, atol=1e-11)


def test_epochs_and_states_broadcast() -> None:
    r = np.array([LEO_R, [0.0, 7e6, 1e5]])
    v = np.array([LEO_V, [-7e3, 0.0, 1e3]])
    # One epoch for many states: one rotation for all.
    one_r, one_v = teme_to_gcrf(EPOCH, r, v)
    assert one_r.shape == (2, 3)
    # A grid: epochs on the first axis, objects on the second.
    epochs = EPOCH + np.array([[0.0], [60.0], [120.0]]) * u.s
    grid_r, grid_v = teme_to_gcrf(epochs, r, v)
    assert grid_r.shape == grid_v.shape == (3, 2, 3)
    np.testing.assert_allclose(grid_r[0], one_r, rtol=0, atol=1e-8)
    np.testing.assert_allclose(grid_v[0], one_v, rtol=0, atol=1e-11)
    later_r, later_v = teme_to_gcrf(epochs[2, 0], r[1], v[1])
    np.testing.assert_allclose(grid_r[2, 1], later_r, rtol=0, atol=1e-8)
    np.testing.assert_allclose(grid_v[2, 1], later_v, rtol=0, atol=1e-11)
    # A failed state (SGP4 gives NaN) stays failed, alone.
    r[0, 0] = np.nan
    failed_r, failed_v = teme_to_gcrf(EPOCH, r, v)
    assert np.all(np.isnan(failed_r[0]))
    assert np.all(np.isnan(failed_v[0]))
    np.testing.assert_array_equal(failed_r[1], one_r[1])
    np.testing.assert_array_equal(failed_v[1], one_v[1])


@pytest.mark.parametrize(
    ("epochs", "r", "v", "message"),
    [
        (EPOCH, [1.0, 2.0], [1.0, 2.0, 3.0], r"r must have 3 components"),
        (EPOCH, [1.0, 2.0, 3.0], 1.0, r"v must have 3 components"),
        (
            EPOCH + [0.0, 1.0, 2.0] * u.s,
            np.ones((2, 3)),
            np.ones((2, 3)),
            r"epochs \(3,\), r \(2, 3\) and v \(2, 3\) do not broadcast",
        ),
        (
            EPOCH,
            np.ones((2, 3)),
            np.ones((4, 3)),
            r"epochs \(\), r \(2, 3\) and v \(4, 3\) do not broadcast",
        ),
    ],
)
def test_teme_to_gcrf_rejects_bad_shapes(
    epochs: Time, r: object, v: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        teme_to_gcrf(epochs, r, v)  # type: ignore[arg-type]


def test_teme_states_are_converted_together(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    convert = frames.teme_to_gcrf

    def counting(
        epochs: Time, r: FloatArray, v: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        calls.append(len(epochs))
        return convert(epochs, r, v)

    monkeypatch.setattr(frames, "teme_to_gcrf", counting)
    states = [
        StateVector(EPOCH, LEO_R, LEO_V, "TEME"),
        StateVector(EPOCH, LEO_R, LEO_V, "GCRF"),
        StateVector(EPOCH, [6778137.0, 0.0, 0.0], [0.0, 5900.0, 4400.0], "ITRF"),
        StateVector(EPOCH + 60 * u.s, [0.0, 7e6, 1e5], [-7e3, 0.0, 1e3], "TEME"),
    ]
    converted = to_gcrf_all(states)
    assert calls == [2]
    assert converted[1] is states[1]
    for state, together in zip(states, converted, strict=True):
        alone = to_gcrf(state)
        assert together.frame == "GCRF"
        assert together.epoch is state.epoch
        np.testing.assert_allclose(together.r, alone.r, rtol=0, atol=1e-6)
        np.testing.assert_allclose(together.v, alone.v, rtol=0, atol=1e-9)
    teme = converted[0]
    assert not np.allclose(teme.r, LEO_R, rtol=0, atol=1.0)
