from pathlib import Path

import numpy as np
import pytest

from keepout.errors import CaraFormatError
from keepout.validation.legacy import legacy_objects

COVARIANCE = """\
CR_R = 100.0 [m**2]
CT_R = 0.0 [m**2]
CT_T = 400.0 [m**2]
CN_R = 0.0 [m**2]
CN_T = 0.0 [m**2]
CN_N = 25.0 [m**2]
CRDOT_R = 0.0 [m**2/s]
CRDOT_T = 0.0 [m**2/s]
CRDOT_N = 0.0 [m**2/s]
CRDOT_RDOT = 0.01 [m**2/s**2]
CTDOT_R = 0.0 [m**2/s]
CTDOT_T = 0.0 [m**2/s]
CTDOT_N = 0.0 [m**2/s]
CTDOT_RDOT = 0.0 [m**2/s**2]
CTDOT_TDOT = 0.04 [m**2/s**2]
CNDOT_R = 0.0 [m**2/s]
CNDOT_T = 0.0 [m**2/s]
CNDOT_N = 0.0 [m**2/s]
CNDOT_RDOT = 0.0 [m**2/s**2]
CNDOT_TDOT = 0.0 [m**2/s**2]
CNDOT_NDOT = 0.01 [m**2/s**2]
"""


def legacy_cdm(frame: str = "EME2000") -> str:
    # Non-conforming on purpose: RELATIVE_VELOCITY in [m], NaN, Latin-1 unit.
    return (
        "CCSDS_CDM_VERS = 1.0\n"
        "TCA = 2008-06-27T15:34:55.320\n"
        "RELATIVE_VELOCITY_R = 36.1 [m]\n"
        "RECOMMENDED_OD_SPAN = NaN [d]\n"
        "OBJECT = OBJECT1\n"
        "OBJECT_DESIGNATOR = 28376\n"
        f"REF_FRAME = {frame}\n"
        "RESIDUALS_ACCEPTED = 85.4 [\xb0]\n"
        "X = 7000.0 [km]\nY = 0.0 [km]\nZ = 0.0 [km]\n"
        "X_DOT = 0.0 [km/s]\nY_DOT = 7.5 [km/s]\nZ_DOT = 0.0 [km/s]\n"
        + COVARIANCE
        + "OBJECT = OBJECT2\n"
        "OBJECT_DESIGNATOR = 1399\n"
        f"REF_FRAME = {frame}\n"
        "X = 7000.01 [km]\nY = 0.0 [km]\nZ = 0.0 [km]\n"
        "X_DOT = 0.0 [km/s]\nY_DOT = 0.0 [km/s]\nZ_DOT = 7.5 [km/s]\n" + COVARIANCE
    )


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "legacy.cdm"
    path.write_bytes(text.encode("latin-1"))
    return path


def test_states_and_covariances_are_read_in_si_units(tmp_path: Path) -> None:
    primary, secondary = legacy_objects(write(tmp_path, legacy_cdm("GCRF")))
    assert (primary.object_id, secondary.object_id) == ("28376", "1399")
    np.testing.assert_allclose(primary.state.r, [7000e3, 0.0, 0.0])
    np.testing.assert_allclose(secondary.state.v, [0.0, 0.0, 7500.0])
    assert primary.covariance is not None
    rtn = primary.covariance.in_rtn(primary.state).matrix
    np.testing.assert_allclose(
        np.diag(rtn), [100, 400, 25, 0.01, 0.04, 0.01], atol=1e-9
    )


def test_eme2000_states_are_converted_to_gcrf(tmp_path: Path) -> None:
    primary, _ = legacy_objects(write(tmp_path, legacy_cdm()))
    assert primary.state.frame == "GCRF"
    # EME2000 and GCRF differ by a frame bias of about 23 mas.
    assert np.linalg.norm(primary.state.r - [7000e3, 0.0, 0.0]) < 2.0


def test_an_unused_covariance_is_unknown(tmp_path: Path) -> None:
    _, secondary = legacy_objects(write(tmp_path, legacy_cdm()), known=(True, False))
    assert secondary.covariance is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (legacy_cdm().replace("TCA =", "XCA ="), "no TCA"),
        (legacy_cdm().replace("OBJECT = OBJECT2\n", ""), "expected 2 object sections"),
        (legacy_cdm("TEME"), "unsupported REF_FRAME 'TEME'"),
        (legacy_cdm().replace("CN_N = 25.0", "CN_N = abc", 1), "no numeric CN_N"),
    ],
)
def test_malformed_files_are_format_errors(
    tmp_path: Path, text: str, message: str
) -> None:
    with pytest.raises(CaraFormatError, match=message):
        legacy_objects(write(tmp_path, text))
