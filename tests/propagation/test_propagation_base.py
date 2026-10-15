"""The propagator interface: ephemerides, error codes and chunked grids."""

import itertools
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
    RadiusBounds,
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

    def radius_bounds(self, start: Time, duration: float) -> RadiusBounds:
        # The closest and farthest points of each segment from t = 0 to duration.
        speed2 = np.sum(self.v0 * self.v0, axis=1)
        closest = np.clip(-np.sum(self.r0 * self.v0, axis=1) / speed2, 0.0, duration)

        def radius(times: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.linalg.norm(self.r0 + self.v0 * times[:, None], axis=1)

        ends = np.maximum(
            radius(np.zeros(len(self))), radius(np.full(len(self), duration))
        )
        return RadiusBounds(radius(closest), ends)


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
    with pytest.raises(TypeError, match="does not support item assignment"):
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


def test_radius_bounds_hold_read_only_copies() -> None:
    lowest = np.array([7e6, 0.0])
    bounds = RadiusBounds(lowest, np.array([7.1e6, math.inf]))
    lowest[0] = 1.0
    np.testing.assert_array_equal(bounds.lowest, [7e6, 0.0])
    np.testing.assert_array_equal(bounds.highest, [7.1e6, math.inf])
    assert bounds.lowest.dtype == bounds.highest.dtype == np.float64
    assert not bounds.lowest.flags.writeable
    assert not bounds.highest.flags.writeable
    assert len(bounds) == 2
    assert len(RadiusBounds(np.empty(0), np.empty(0))) == 0


@pytest.mark.parametrize(
    ("lowest", "highest", "message"),
    [
        ([1.0, 2.0], [3.0], r"same length, got shapes \(2,\) and \(1,\)"),
        ([[1.0]], [[2.0]], "one-dimensional"),
        ([-1.0], [2.0], "lowest must be finite and >= 0"),
        ([math.nan], [2.0], "lowest must be finite and >= 0"),
        ([math.inf], [math.inf], "lowest must be finite and >= 0"),
        ([2.0], [1.0], "highest must be >= lowest"),
        ([1.0], [math.nan], "highest must be >= lowest"),
    ],
)
def test_radius_bounds_reject_bad_bounds(
    lowest: list[float], highest: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RadiusBounds(lowest, highest)  # type: ignore[arg-type]


@pytest.mark.parametrize("duration", [0.0, 50.0, 1e4])
def test_the_bounds_hold_the_propagated_radius(duration: float) -> None:
    propagator = linear(5)
    bounds = propagator.radius_bounds(EPOCH, duration)
    ephemeris = propagator.propagate(EPOCH, np.linspace(0.0, duration, 101))
    radius = np.linalg.norm(ephemeris.r, axis=-1)
    lowest = np.broadcast_to(bounds.lowest[:, None], radius.shape)
    highest = np.broadcast_to(bounds.highest[:, None], radius.shape)
    valid = ephemeris.valid
    assert np.all(lowest[valid] <= radius[valid] * (1 + 1e-15))
    assert np.all(radius[valid] <= highest[valid] * (1 + 1e-15))


def test_a_propagator_is_anything_with_the_protocol() -> None:
    propagator: Propagator = linear(3)
    assert isinstance(propagator, Propagator)
    assert not isinstance(object(), Propagator)


@given(
    counts=st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=3),
    times=st.integers(min_value=0, max_value=30),
    max_states=st.integers(min_value=1, max_value=50),
    overlap=st.booleans(),
)
def test_chunks_hold_the_whole_grid(
    counts: list[int], times: int, max_states: int, overlap: bool
) -> None:
    propagators = [linear(count) for count in counts]
    offsets = np.arange(times) * 37.0 - 100.0
    chunks = list(
        propagate_in_chunks(
            propagators, EPOCH, offsets, max_states=max_states, overlap=overlap
        )
    )
    assert all(len(chunk) == len(propagators) for chunk in chunks)
    size = max(2 if overlap else 1, max_states // max(1, sum(counts)))
    lengths = [len(chunk[0].offsets) for chunk in chunks]
    assert all(length <= size for length in lengths)
    if not chunks:
        assert times == 0
        return
    # Every chunk after the first starts at the last time of the one before
    # with overlap: dropped, the chunks hold the grid once.
    skip = 1 if overlap else 0
    assert sum(lengths) - skip * (len(chunks) - 1) == times
    for which, propagator in enumerate(propagators):
        parts = [chunk[which] for chunk in chunks]
        if overlap:
            for before, after in itertools.pairwise(parts):
                assert before.offsets[-1] == after.offsets[0]
        whole = propagator.propagate(EPOCH, offsets)
        np.testing.assert_array_equal(
            np.concatenate([parts[0].offsets, *(p.offsets[skip:] for p in parts[1:])]),
            whole.offsets,
        )
        for name in ("r", "v", "errors"):
            pieces = [getattr(part, name) for part in parts]
            joined = np.concatenate(
                [pieces[0], *(piece[:, skip:] for piece in pieces[1:])], axis=1
            )
            np.testing.assert_array_equal(joined, getattr(whole, name))
        assert all(part.epoch is EPOCH for part in parts)


def test_overlapping_chunks_hold_every_interval() -> None:
    (first, second, third) = propagate_in_chunks(
        [linear(1)], EPOCH, np.arange(7.0), max_states=3, overlap=True
    )
    assert [chunk.offsets.tolist() for (chunk,) in (first, second, third)] == [
        [0.0, 1.0, 2.0],
        [2.0, 3.0, 4.0],
        [4.0, 5.0, 6.0],
    ]
    # Two times at least, even above max_states; a grid of one time is one chunk.
    pairs = propagate_in_chunks(
        [linear(3)], EPOCH, np.arange(3.0), max_states=1, overlap=True
    )
    assert [chunk.offsets.tolist() for (chunk,) in pairs] == [[0.0, 1.0], [1.0, 2.0]]
    (alone,) = propagate_in_chunks([linear(1)], EPOCH, [5.0], overlap=True)
    assert alone[0].offsets.tolist() == [5.0]


def test_the_default_chunk_bounds_memory() -> None:
    # CHUNK_STATES = 1 000 000 states: 2 times of 400 000 objects.
    ((chunk,),) = propagate_in_chunks([linear(2)], EPOCH, np.arange(10.0))
    assert chunk.r.shape == (2, 10, 3)
    first = next(propagate_in_chunks([linear(400_000)], EPOCH, np.arange(10.0)))
    assert first[0].r.shape == (400_000, 2, 3)
    # The objects of every propagator count.
    both = next(propagate_in_chunks([linear(200_000)] * 2, EPOCH, np.arange(10.0)))
    assert [part.r.shape for part in both] == [(200_000, 2, 3)] * 2
    # A chunk has at least one time, even above max_states.
    single = next(
        propagate_in_chunks([linear(3)], EPOCH, np.arange(10.0), max_states=1)
    )
    assert single[0].r.shape == (3, 1, 3)


def test_chunk_arguments_are_checked_at_the_call() -> None:
    with pytest.raises(ValueError, match="max_states must be at least 1, got 0"):
        propagate_in_chunks([linear(1)], EPOCH, [0.0], max_states=0)
    with pytest.raises(ValueError, match="offsets must be finite"):
        propagate_in_chunks([linear(1)], EPOCH, [math.nan])
