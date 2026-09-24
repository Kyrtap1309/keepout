import numpy as np
import pytest
from astropy.time import Time
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from keepout._types import FloatArray, StateVector
from keepout.covariance import Covariance, make_covariance, problems, repair
from keepout.errors import CovarianceError, GeometryError

EPOCH = Time("2010-03-13T22:37:52.618", scale="utc")
VALID = np.diag([100.0, 400.0, 25.0, 1e-2, 4e-2, 1e-2])
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
    Covariance(np.zeros((6, 6)), "GCRF", EPOCH)


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
    covariance, diagnostics = make_covariance(with_entry(5, 5, -1.0), "RTN", EPOCH)
    assert diagnostics == (
        "clipped 1 negative eigenvalue(s) to zero (most negative -1)",
    )
    assert covariance.matrix[5, 5] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(np.diag(covariance.matrix)[:5], np.diag(VALID)[:5])


def test_make_covariance_symmetrises_with_diagnostic() -> None:
    covariance, diagnostics = make_covariance(with_entry(0, 1, 1.0), "RTN", EPOCH)
    assert diagnostics == ("symmetrised covariance (max asymmetry 1)",)
    assert covariance.matrix[0, 1] == pytest.approx(0.5, rel=1e-12)
    assert covariance.matrix[1, 0] == pytest.approx(0.5, rel=1e-12)


def test_valid_matrix_is_not_touched_by_repair_policy() -> None:
    covariance, diagnostics = make_covariance(VALID, "RTN", EPOCH)
    assert diagnostics == ()
    np.testing.assert_array_equal(covariance.matrix, VALID)


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown covariance policy 'ignore'"):
        # The wrong literal is the point of the test.
        make_covariance(VALID, "RTN", EPOCH, policy="ignore")  # type: ignore[arg-type]


@given(arrays(np.float64, (6, 6), elements=st.floats(min_value=-1e3, max_value=1e3)))
def test_repair_always_yields_a_valid_covariance(matrix: FloatArray) -> None:
    fixed, _ = repair(matrix)
    assert problems(fixed) == ()
    Covariance(fixed, "RTN", EPOCH)


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
