import numpy as np
import pytest
from builders import DIAGONAL_COVARIANCE, EPOCH
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from keepout._types import FloatArray, StateVector
from keepout.covariance import Covariance, make_covariance, problems, repair
from keepout.errors import CovarianceError, GeometryError

VALID = DIAGONAL_COVARIANCE
STATE = StateVector(EPOCH, [7000e3, 100.0, 50.0], [10.0, 7500.0, 1000.0], "GCRF")


def with_entry(row: int, column: int, value: float) -> FloatArray:
    matrix = VALID.copy()
    matrix[row, column] = value
    return matrix


def test_valid_covariance_is_frozen() -> None:
    covariance = Covariance(VALID, "RTN", EPOCH)
    np.testing.assert_array_equal(covariance.matrix, VALID)
    np.testing.assert_array_equal(covariance.position, VALID[:3, :3])
    with pytest.raises(ValueError, match="read-only"):
        covariance.matrix[0, 0] = 1.0


def test_zero_covariance_is_valid() -> None:
    assert problems(np.zeros((6, 6))) == ()


def test_a_small_block_is_checked_at_its_own_scale() -> None:
    # 10 km position sigmas must not hide a negative velocity variance.
    matrix = np.diag([1e8, 1e8, 1e8, 1e-4, 1e-4, -1e-3])
    (problem,) = problems(matrix)
    assert problem.startswith("not positive semi-definite")
    fixed = make_covariance(matrix, "RTN", EPOCH).matrix
    assert np.linalg.eigvalsh(fixed)[0] >= 0.0
    np.testing.assert_allclose(np.diag(fixed)[:5], np.diag(matrix)[:5])


def test_a_velocity_defect_leaves_the_position_block_unchanged() -> None:
    # Position-velocity correlation 1.5 along x: the velocity covariance
    # conditional on position is negative there, the position block is fine.
    matrix = VALID.copy()
    matrix[0, 3] = matrix[3, 0] = 1.5 * np.sqrt(VALID[0, 0] * VALID[3, 3])
    fixed, repairs = repair(matrix)
    (note,) = repairs
    assert note.startswith("clipped 1 negative eigenvalue(s) of the velocity")
    np.testing.assert_array_equal(fixed[:3, :3], matrix[:3, :3])
    np.testing.assert_allclose(fixed[3:, :3], matrix[3:, :3], rtol=1e-12)
    assert problems(fixed) == ()


def test_an_invalid_position_block_is_repaired_first() -> None:
    matrix = VALID.copy()
    matrix[0, 1] = matrix[1, 0] = 1.2 * np.sqrt(VALID[0, 0] * VALID[1, 1])
    fixed, repairs = repair(matrix)
    assert repairs[0].startswith("clipped 1 negative position-block eigenvalue")
    assert problems(fixed) == ()
    np.testing.assert_allclose(fixed[3:, 3:], matrix[3:, 3:], rtol=1e-12)


def test_a_small_block_is_checked_for_symmetry_at_its_own_scale() -> None:
    matrix = np.diag([1e8, 1e8, 1e8, 1e-4, 1e-4, 1e-4])
    matrix[3, 4] = 1e-5  # correlation 0.1 on one side only
    (problem,) = problems(matrix)
    assert problem.startswith("not symmetric")


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.eye(3), "6x6"),
        (with_entry(2, 2, float("nan")), "non-finite"),
        (with_entry(0, 1, 1.0), "not symmetric"),
        (with_entry(5, 5, -1.0), "not positive semi-definite"),
    ],
)
def test_invalid_covariance_is_rejected(matrix: FloatArray, message: str) -> None:
    with pytest.raises(CovarianceError, match=message):
        Covariance(matrix, "RTN", EPOCH)


def test_unknown_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported covariance frame 'ECEF'"):
        # The wrong literal is the point of the test.
        Covariance(VALID, "ECEF", EPOCH)  # type: ignore[arg-type]


def test_raise_policy_rejects_invalid_matrix() -> None:
    with pytest.raises(CovarianceError, match="not positive semi-definite"):
        make_covariance(with_entry(5, 5, -1.0), "RTN", EPOCH, policy="raise")


def test_make_covariance_repairs_with_diagnostic() -> None:
    covariance = make_covariance(with_entry(5, 5, -1.0), "RTN", EPOCH)
    assert covariance.repairs == (
        "clipped 1 negative eigenvalue(s) of the velocity covariance conditional "
        "on position to zero (most negative -1 in correlation units)",
    )
    assert covariance.matrix[5, 5] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(np.diag(covariance.matrix)[:5], np.diag(VALID)[:5])


def test_make_covariance_symmetrises_with_diagnostic() -> None:
    covariance = make_covariance(with_entry(0, 1, 1.0), "RTN", EPOCH)
    # 1 m^2 against sigmas of 10 m and 20 m.
    assert covariance.repairs == (
        "symmetrised covariance (max relative asymmetry 0.005)",
    )
    assert covariance.matrix[0, 1] == pytest.approx(0.5, rel=1e-12)
    assert covariance.matrix[1, 0] == pytest.approx(0.5, rel=1e-12)


def test_valid_matrix_is_not_touched_by_repair_policy() -> None:
    covariance = make_covariance(VALID, "RTN", EPOCH)
    assert covariance.repairs == ()
    np.testing.assert_array_equal(covariance.matrix, VALID)


def test_make_covariance_rejects_an_unknown_frame() -> None:
    with pytest.raises(ValueError, match="unsupported covariance frame 'ECEF'"):
        make_covariance(VALID, "ECEF", EPOCH)  # type: ignore[arg-type]


def test_repairs_survive_frame_changes() -> None:
    covariance = make_covariance(with_entry(0, 1, 1.0), "RTN", EPOCH)
    gcrf = covariance.in_gcrf(STATE)
    assert gcrf.repairs == covariance.repairs
    assert gcrf.in_rtn(STATE).repairs == covariance.repairs


def test_rotated_covariance_is_frozen_and_symmetric() -> None:
    gcrf = Covariance(VALID, "RTN", EPOCH).in_gcrf(STATE)
    np.testing.assert_array_equal(gcrf.matrix, gcrf.matrix.T)
    with pytest.raises(ValueError, match="read-only"):
        gcrf.matrix[0, 0] = 1.0


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown covariance policy 'ignore'"):
        # The wrong literal is the point of the test.
        make_covariance(VALID, "RTN", EPOCH, policy="ignore")  # type: ignore[arg-type]


@given(
    arrays(np.float64, (6, 6), elements=st.floats(min_value=-1e3, max_value=1e3)),
    arrays(np.float64, 6, elements=st.floats(min_value=-6.0, max_value=4.0)),
)
def test_repair_always_yields_a_valid_covariance(
    matrix: FloatArray, log_scales: FloatArray
) -> None:
    # Scales spread as in real CDMs (mm/s velocities next to km positions).
    scales = 10.0**log_scales
    scaled = matrix * np.outer(scales, scales)
    fixed, _ = repair(scaled)
    assert problems(fixed) == ()
    Covariance(fixed, "RTN", EPOCH)


@given(
    arrays(np.float64, (6, 6), elements=st.floats(min_value=-1.0, max_value=1.0)),
    arrays(np.float64, 6, elements=st.floats(min_value=-6.0, max_value=4.0)),
)
def test_repair_leaves_a_valid_covariance_unchanged(
    factor: FloatArray, log_scales: FloatArray
) -> None:
    scales = 10.0**log_scales
    matrix = (factor @ factor.T + np.eye(6)) * np.outer(scales, scales)
    fixed, repairs = repair(matrix)
    assert repairs == ()
    # Compared in correlation units, the scale the check works in.
    outer = np.outer(scales, scales)
    np.testing.assert_allclose(fixed / outer, matrix / outer, rtol=0.0, atol=1e-12)


def test_frame_round_trip() -> None:
    matrix = VALID.copy()
    matrix[0, 1] = matrix[1, 0] = 50.0
    rtn = Covariance(matrix, "RTN", EPOCH)
    gcrf = rtn.in_gcrf(STATE)
    assert gcrf.frame == "GCRF"
    assert gcrf.epoch is EPOCH
    np.testing.assert_allclose(gcrf.in_rtn(STATE).matrix, matrix, atol=1e-9)
    assert gcrf.in_gcrf(STATE) is gcrf
    assert rtn.in_rtn(STATE) is rtn


def test_frame_change_needs_a_gcrf_state() -> None:
    state = StateVector(EPOCH, STATE.r, STATE.v, "EME2000")
    with pytest.raises(GeometryError, match="GCRF state"):
        Covariance(VALID, "RTN", EPOCH).in_gcrf(state)


def test_position_only_covariance_has_a_zero_velocity_block() -> None:
    covariance = Covariance.from_position(4.0 * np.eye(3), "GCRF", EPOCH)
    np.testing.assert_array_equal(covariance.position, 4.0 * np.eye(3))
    np.testing.assert_array_equal(covariance.matrix[3:, :], np.zeros((3, 6)))


def test_position_only_covariance_must_be_3x3() -> None:
    with pytest.raises(CovarianceError, match="must be 3x3"):
        Covariance.from_position(np.eye(6), "GCRF", EPOCH)
