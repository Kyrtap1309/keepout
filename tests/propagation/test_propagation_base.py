"""The propagator interface: ephemerides, error codes and chunked grids."""

import math
from dataclasses import dataclass, field
from typing import Any, Self

import astropy.units as u
import numpy as np
import numpy.typing as npt
import pytest
from astropy.time import Time
from hypothesis import given
from hypothesis import strategies as st
from sgp4.api import SGP4_ERRORS

from keepout.errors import UnsupportedFrame
from keepout.propagation import (
    ERROR_MESSAGES,
    NO_ERROR,
    Ephemeris,
    Propagator,
    describe_error,
    propagate_in_chunks,
    time_offsets,
)
from keepout.state import RefFrame

EPOCH = Time("2026-10-06T00:00:00", scale="utc")


@dataclass(frozen=True)
class LinearPropagator:
    """Objects in straight-line motion; object i fails after ``limits[i]`` s."""

    r0: npt.NDArray[np.float64]
    v0: npt.NDArray[np.float64]
    limits: npt.NDArray[np.float64]
    frame: RefFrame = field(default="GCRF")

    def __len__(self) -> int:
        return len(self.r0)

    def subset(self, indices: npt.ArrayLike) -> Self:
        chosen = np.asarray(indices, dtype=np.intp)
        return type(self)(self.r0[chosen], self.v0[chosen], self.limits[chosen])

    def propagate(self, epoch: Time, offsets: npt.ArrayLike) -> Ephemeris:
        seconds = time_offsets(epoch, offsets)
        r = self.r0[:, None, :] + self.v0[:, None, :] * seconds[None, :, None]
        v = np.broadcast_to(self.v0[:, None, :], r.shape).copy()
        errors = np.where(seconds[None, :] > self.limits[:, None], 6, 0)
        r[errors != 0] = np.nan
        v[errors != 0] = np.nan
        return Ephemeris(epoch, seconds, r, v, errors.astype(np.uint8), self.frame)


def linear(count: int) -> LinearPropagator:
    index = np.arange(count, dtype=np.float64)
    return LinearPropagator(
        r0=np.stack([7e6 + index, index, -index], axis=1),
        v0=np.stack([index, 7.5e3 - index, 0.5 * index], axis=1),
        limits=100.0 * (index + 1.0),
    )


def test_describe_error() -> None:
    assert describe_error(NO_ERROR) == "no error"
    assert describe_error(1) == "error 1: mean eccentricity outside the range 0 to 1"
    assert describe_error(6) == "error 6: decayed: orbit radius below one Earth radius"
    assert describe_error(9) == "error 9: unknown"


def test_error_codes_are_sgp4s() -> None:
    assert set(ERROR_MESSAGES) == set(SGP4_ERRORS)
    with pytest.raises(TypeError):
        ERROR_MESSAGES[7] = "new"  # type: ignore[index]


def test_time_offsets_are_a_read_only_copy() -> None:
    given_offsets = [0.0, 60.0, -30.0]
    seconds = time_offsets(EPOCH, given_offsets)
    assert seconds.dtype == np.float64
    np.testing.assert_array_equal(seconds, given_offsets)
    assert not seconds.flags.writeable
    source = np.array(given_offsets)
    copied = time_offsets(EPOCH, source)
    source[0] = 1.0
    assert copied[0] == 0.0
    np.testing.assert_array_equal(time_offsets(EPOCH, 12.5), [12.5])
    assert time_offsets(EPOCH, []).shape == (0,)


@pytest.mark.parametrize(
    ("epoch", "offsets", "message"),
    [
        (
            EPOCH + [0.0, 1.0] * u.s,
            [0.0],
            r"epoch must be a single time, got shape \(2,\)",
        ),
        (EPOCH, [[0.0, 1.0]], r"offsets must be one-dimensional, got shape \(1, 2\)"),
        (EPOCH, [0.0, math.nan], "offsets must be finite"),
        (EPOCH, [math.inf], "offsets must be finite"),
    ],
)
def test_time_offsets_rejects_bad_grids(
    epoch: Time, offsets: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        time_offsets(epoch, offsets)  # type: ignore[arg-type]


def test_ephemeris_holds_read_only_views() -> None:
    r = np.arange(12.0).reshape(2, 2, 3)
    v = -r
    errors = np.array([[0, 0], [0, 6]], dtype=np.uint8)
    ephemeris = Ephemeris(EPOCH, np.array([0.0, 90.0]), r, v, errors, "TEME")
    for array in (ephemeris.offsets, ephemeris.r, ephemeris.v, ephemeris.errors):
        assert not array.flags.writeable
    # Views, not copies; the caller's arrays stay writeable.
    assert np.shares_memory(ephemeris.r, r)
    assert r.flags.writeable
    np.testing.assert_array_equal(ephemeris.valid, [[True, True], [True, False]])
    assert ephemeris.times.isot.tolist() == [
        "2026-10-06T00:00:00.000",
        "2026-10-06T00:01:30.000",
    ]
    assert ephemeris.frame == "TEME"


@pytest.mark.parametrize(
    ("r_shape", "v_shape", "errors", "message"),
    [
        ((2, 3), (2, 3), np.zeros(2, np.uint8), r"r must have shape \(N, 2, 3\)"),
        ((1, 3, 3), (1, 3, 3), np.zeros((1, 3), np.uint8), r"\(N, 2, 3\) for 2"),
        ((1, 2, 3), (2, 2, 3), np.zeros((1, 2), np.uint8), r"v must have the shape"),
        ((1, 2, 3), (1, 2, 3), np.zeros((2, 2), np.uint8), r"errors must have shape"),
        ((1, 2, 3), (1, 2, 3), np.zeros((1, 2), np.int64), r"type uint8, got int64"),
    ],
)
def test_ephemeris_rejects_mismatched_arrays(
    r_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    errors: npt.NDArray[Any],
    message: str,
) -> None:
    offsets = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match=message):
        Ephemeris(EPOCH, offsets, np.zeros(r_shape), np.zeros(v_shape), errors, "TEME")


def test_ephemeris_rejects_a_bad_grid_or_frame() -> None:
    r = np.zeros((1, 1, 3))
    errors = np.zeros((1, 1), np.uint8)
    with pytest.raises(ValueError, match="offsets must be finite"):
        Ephemeris(EPOCH, np.array([math.nan]), r, r, errors, "TEME")
    with pytest.raises(UnsupportedFrame, match="'TOD'"):
        Ephemeris(EPOCH, np.zeros(1), r, r, errors, "TOD")  # type: ignore[arg-type]


def test_a_propagator_is_anything_with_the_protocol() -> None:
    propagator: Propagator = linear(3)
    assert isinstance(propagator, Propagator)
    assert not isinstance(object(), Propagator)


@given(
    count=st.integers(min_value=0, max_value=5),
    times=st.integers(min_value=0, max_value=30),
    max_states=st.integers(min_value=1, max_value=50),
)
def test_chunks_hold_the_whole_grid(count: int, times: int, max_states: int) -> None:
    propagator = linear(count)
    offsets = np.arange(times) * 37.0 - 100.0
    chunks = list(
        propagate_in_chunks(propagator, EPOCH, offsets, max_states=max_states)
    )
    length = max(1, max_states // max(1, count))
    assert [len(chunk.offsets) for chunk in chunks] == [
        min(length, times - start) for start in range(0, times, length)
    ]
    whole = propagator.propagate(EPOCH, offsets)
    if chunks:
        np.testing.assert_array_equal(
            np.concatenate([chunk.offsets for chunk in chunks]), offsets
        )
        np.testing.assert_array_equal(
            np.concatenate([c.r for c in chunks], axis=1), whole.r
        )
        np.testing.assert_array_equal(
            np.concatenate([c.v for c in chunks], axis=1), whole.v
        )
        np.testing.assert_array_equal(
            np.concatenate([c.errors for c in chunks], axis=1), whole.errors
        )
    else:
        assert times == 0
    assert all(chunk.epoch is EPOCH for chunk in chunks)


def test_the_default_chunk_bounds_memory() -> None:
    # CHUNK_STATES = 1 000 000 states: 2 times of 400 000 objects.
    (chunk,) = propagate_in_chunks(linear(2), EPOCH, np.arange(10.0))
    assert chunk.r.shape == (2, 10, 3)
    first = next(propagate_in_chunks(linear(400_000), EPOCH, np.arange(10.0)))
    assert first.r.shape == (400_000, 2, 3)
    # A chunk has at least one time, even above max_states.
    single = next(propagate_in_chunks(linear(3), EPOCH, np.arange(10.0), max_states=1))
    assert single.r.shape == (3, 1, 3)


def test_chunk_arguments_are_checked_at_the_call() -> None:
    with pytest.raises(ValueError, match="max_states must be at least 1, got 0"):
        propagate_in_chunks(linear(1), EPOCH, [0.0], max_states=0)
    with pytest.raises(ValueError, match="offsets must be finite"):
        propagate_in_chunks(linear(1), EPOCH, [math.nan])
