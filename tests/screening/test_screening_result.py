"""Close approaches and screening results: geometry derived from the two GCRF
states, conjunctions with covariances from elsewhere, and a stable order."""

import math
import pickle

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from screening_data import load_fixture

from keepout._types import FloatArray
from keepout.covariance import make_covariance
from keepout.errors import GeometryError
from keepout.frames import rtn_rotation
from keepout.pc import compute_pc
from keepout.screening import (
    DEFAULT_VOLUME,
    CloseApproach,
    ScreeningResult,
    ScreeningRole,
    ScreeningVolume,
    SkippedObject,
    SkipReason,
)
from keepout.state import StateVector

TCA = Time("2026-10-07T12:00:00", scale="utc")
R_PRIMARY = np.array([7000e3, 0.0, 0.0])
V_PRIMARY = np.array([0.0, 7546.0, 0.0])
"""A circular, prograde, equatorial orbit: R = x, T = y, N = z."""
MISS = np.array([100.0, 0.0, 300.0])
RELATIVE = np.array([3.0, -15092.0, -1.0])
"""Head on, perpendicular to MISS: closest at TCA."""
FIXTURE = load_fixture()


def states(
    epoch: Time = TCA, miss: FloatArray = MISS, relative: FloatArray = RELATIVE
) -> tuple[StateVector, StateVector]:
    primary = StateVector(epoch, R_PRIMARY, V_PRIMARY, "GCRF")
    secondary = StateVector(epoch, R_PRIMARY + miss, V_PRIMARY + relative, "GCRF")
    return primary, secondary


APPROACH = CloseApproach(25544, 48274, *states())


def test_a_close_approach_derives_its_geometry_from_the_states() -> None:
    assert APPROACH.tca == TCA
    assert APPROACH.tca.scale == "utc"
    np.testing.assert_array_equal(APPROACH.relative_position, MISS)
    assert APPROACH.miss_distance == pytest.approx(math.hypot(100.0, 300.0), rel=1e-15)
    np.testing.assert_allclose(APPROACH.miss_rtn, MISS, atol=1e-9)
    assert not APPROACH.miss_rtn.flags.writeable
    np.testing.assert_array_equal(APPROACH.relative_velocity, RELATIVE)
    assert APPROACH.relative_speed == pytest.approx(np.linalg.norm(RELATIVE), rel=1e-15)


def test_the_tca_is_given_in_utc() -> None:
    approach = CloseApproach(1, 2, *states(TCA.tt))
    assert approach.tca.scale == "utc"
    assert abs((approach.tca - TCA).to_value(u.s)) < 1e-9


positions = arrays(
    np.float64, 3, elements=st.floats(min_value=-8e6, max_value=8e6)
).filter(lambda r: np.linalg.norm(r) > 6.4e6)
# Speeds below ~1e-154 m/s make |v|^2 underflow and cost the RTN axes their
# precision (in rtn_rotation too); no orbiting object is that slow.
velocities = arrays(
    np.float64, 3, elements=st.floats(min_value=-8e3, max_value=8e3)
).filter(lambda v: np.linalg.norm(v) >= 1.0)
offsets = arrays(np.float64, 3, elements=st.floats(min_value=-2e5, max_value=2e5))


@given(r=positions, v=velocities, miss=offsets)
def test_the_miss_vector_is_the_relative_position_in_rtn(
    r: FloatArray, v: FloatArray, miss: FloatArray
) -> None:
    angular = np.linalg.norm(np.cross(r, v))
    if angular <= 1e-6 * np.linalg.norm(r) * np.linalg.norm(v):
        return  # no RTN frame
    primary = StateVector(TCA, r, v, "GCRF")
    secondary = StateVector(TCA, r + miss, v, "GCRF")
    approach = CloseApproach(1, 2, primary, secondary)
    scale = 1e-9 * max(np.linalg.norm(r), 1.0)
    np.testing.assert_allclose(approach.miss_rtn, rtn_rotation(r, v) @ miss, atol=scale)
    assert np.linalg.norm(approach.miss_rtn) == pytest.approx(
        approach.miss_distance, abs=scale
    )


@pytest.mark.parametrize(
    ("primary", "secondary", "message"),
    [
        (
            StateVector(TCA, R_PRIMARY, V_PRIMARY, "TEME"),
            states()[1],
            "needs GCRF states, got TEME",
        ),
        (
            states()[0],
            StateVector(TCA, R_PRIMARY, V_PRIMARY, "EME2000"),
            "needs GCRF states, got EME2000",
        ),
        (states()[0], states(TCA + 0.01 * u.s)[1], "epochs differ by 0.01 s"),
    ],
)
def test_a_close_approach_needs_gcrf_states_at_one_epoch(
    primary: StateVector, secondary: StateVector, message: str
) -> None:
    with pytest.raises(GeometryError, match=message):
        CloseApproach(1, 2, primary, secondary)


def test_a_close_approach_becomes_a_conjunction_without_covariances() -> None:
    conjunction = APPROACH.to_conjunction(hard_body_radius=20.0)
    assert conjunction.primary.object_id == "25544"
    assert conjunction.secondary.object_id == "48274"
    assert conjunction.primary.state is APPROACH.primary_state
    assert conjunction.secondary.state is APPROACH.secondary_state
    assert conjunction.unknown_covariance == ("primary", "secondary")
    assert conjunction.hard_body_radius == 20.0
    assert conjunction.tca == TCA
    # Closest at TCA: the miss distance on the encounter plane is the range.
    assert conjunction.miss_distance == pytest.approx(APPROACH.miss_distance, rel=1e-9)


def test_covariances_are_taken_in_gcrf_or_in_each_objects_rtn() -> None:
    rtn = make_covariance(np.diag([1e4, 4e4, 2.5e3, 1e-2, 4e-2, 1e-2]), "RTN", TCA)
    gcrf = make_covariance(np.diag([2e4, 3e4, 1e4, 1e-2, 1e-2, 1e-2]), "GCRF", TCA)
    conjunction = APPROACH.to_conjunction(
        hard_body_radius=20.0, primary_covariance=rtn, secondary_covariance=gcrf
    )
    assert conjunction.primary.covariance is not None
    np.testing.assert_array_equal(
        conjunction.primary.covariance.matrix,
        rtn.in_gcrf(APPROACH.primary_state).matrix,
    )
    assert conjunction.secondary.covariance is gcrf
    assert 0.0 < compute_pc(conjunction).value < 1.0


def test_a_conjunction_needs_covariances_at_tca_and_a_radius() -> None:
    later = make_covariance(np.eye(6), "GCRF", TCA + 1 * u.s)
    with pytest.raises(GeometryError, match="state and covariance epochs differ"):
        APPROACH.to_conjunction(hard_body_radius=20.0, secondary_covariance=later)
    with pytest.raises(ValueError, match="hard_body_radius must be finite and > 0"):
        APPROACH.to_conjunction(hard_body_radius=0.0)


def approach_at(seconds: float, primary: int, secondary: int) -> CloseApproach:
    return CloseApproach(primary, secondary, *states(TCA + seconds * u.s))


def skip(index: int, role: str, reason: SkipReason) -> SkippedObject:
    return SkippedObject(FIXTURE[index], ScreeningRole(role), reason, "why")


def test_a_result_keeps_approaches_in_time_order_and_skipped_by_object() -> None:
    approaches = [
        approach_at(10.0, 2, 1),
        approach_at(0.0, 5, 9),
        approach_at(0.0, 3, 4),
        approach_at(0.0, 3, 2),
        approach_at(-1e-6, 7, 7),
    ]
    skipped = [
        skip(5, "catalog", SkipReason.STALE),
        skip(3, "catalog", SkipReason.PROPAGATION_FAILED),
        skip(3, "catalog", SkipReason.DUPLICATE),
        skip(8, "primary", SkipReason.NO_VOLUME),
    ]
    result = ScreeningResult(
        tuple(approaches),
        tuple(skipped),
        TCA,
        3600.0,
        {25544: DEFAULT_VOLUME, 20580: ScreeningVolume.nasa_gp_data()},
        120,
    )
    assert [(a.primary_number, a.secondary_number) for a in result.approaches] == [
        (7, 7),
        (3, 2),
        (3, 4),
        (5, 9),
        (2, 1),
    ]
    assert [(s.role, s.catalog_number, s.reason) for s in result.skipped] == [
        ("primary", FIXTURE[8].catalog_number, SkipReason.NO_VOLUME),
        ("catalog", FIXTURE[3].catalog_number, SkipReason.DUPLICATE),
        ("catalog", FIXTURE[3].catalog_number, SkipReason.PROPAGATION_FAILED),
        ("catalog", FIXTURE[5].catalog_number, SkipReason.STALE),
    ]
    assert list(result.volumes) == [25544, 20580]
    with pytest.raises(TypeError, match="does not support item assignment"):
        result.volumes[1] = DEFAULT_VOLUME  # type: ignore[index]
    assert result.end == TCA + 1 * u.hour
    assert result.catalog_count == 120


def test_a_result_pickles() -> None:
    result = ScreeningResult(
        (APPROACH,),
        (skip(0, "primary", SkipReason.STALE),),
        TCA,
        60.0,
        {1: DEFAULT_VOLUME},
        5,
    )
    _ = result.approaches[0].miss_rtn  # cached: pickled with the approach
    copy = pickle.loads(pickle.dumps(result))
    assert copy.volumes == {1: DEFAULT_VOLUME}
    assert copy.skipped == result.skipped
    assert (copy.start, copy.duration, copy.catalog_count) == (TCA, 60.0, 5)
    (approach,) = copy.approaches
    np.testing.assert_array_equal(approach.miss_rtn, APPROACH.miss_rtn)
    np.testing.assert_array_equal(
        approach.secondary_state.v, APPROACH.secondary_state.v
    )
    assert approach.tca == TCA


def test_a_skipped_object_names_its_catalog_number() -> None:
    skipped = skip(4, "catalog", SkipReason.UNSUPPORTED)
    assert skipped.catalog_number == FIXTURE[4].catalog_number
    assert SkipReason.FUTURE_EPOCH.value == "future epoch"
