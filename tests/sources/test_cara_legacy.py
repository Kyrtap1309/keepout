from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from astropy.time import Time
from builders import known

from keepout import frames
from keepout.errors import CaraFormatError
from keepout.frames import to_gcrf
from keepout.sources.cara_legacy import load_legacy_objects
from keepout.state import StateVector

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
    primary, secondary = load_legacy_objects(write(tmp_path, legacy_cdm("GCRF")))
    assert (primary.object_id, secondary.object_id) == ("28376", "1399")
    np.testing.assert_allclose(primary.state.r, [7000e3, 0.0, 0.0])
    np.testing.assert_allclose(secondary.state.v, [0.0, 0.0, 7500.0])
    rtn = known(primary.covariance).in_rtn(primary.state).matrix
    np.testing.assert_allclose(
        np.diag(rtn), [100, 400, 25, 0.01, 0.04, 0.01], atol=1e-9
    )


def test_eme2000_states_are_converted_to_gcrf(tmp_path: Path) -> None:
    primary, _ = load_legacy_objects(write(tmp_path, legacy_cdm()))
    assert primary.state.frame == "GCRF"
    eme2000 = StateVector(
        Time("2008-06-27T15:34:55.320", scale="utc"),
        [7000e3, 0.0, 0.0],
        [0.0, 7500.0, 0.0],
        "EME2000",
    )
    expected = to_gcrf(eme2000)
    np.testing.assert_allclose(primary.state.r, expected.r, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(primary.state.v, expected.v, rtol=0.0, atol=1e-9)
    # EME2000 and GCRF differ by a frame bias of about 23 mas: ~0.8 m at
    # 7000 km, so an unconverted state is caught.
    shift = float(np.linalg.norm(primary.state.r - eme2000.r))
    assert 0.3 < shift < 2.0


def test_itrf_objects_are_converted_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batches: list[int] = []
    convert = frames._itrf_to_gcrf

    def counting(states: Sequence[StateVector]) -> list[StateVector]:
        batches.append(len(states))
        return convert(states)

    monkeypatch.setattr(frames, "_itrf_to_gcrf", counting)
    primary, secondary = load_legacy_objects(write(tmp_path, legacy_cdm("ITRF")))
    assert batches == [2]
    assert primary.state.frame == secondary.state.frame == "GCRF"


def test_an_unused_covariance_is_unknown(tmp_path: Path) -> None:
    _, secondary = load_legacy_objects(
        write(tmp_path, legacy_cdm()), known=(True, False)
    )
    assert secondary.covariance is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (legacy_cdm().replace("TCA =", "XCA ="), "no TCA"),
        (legacy_cdm().replace("OBJECT = OBJECT2\n", ""), "expected 2 object sections"),
        (legacy_cdm("TOD"), "bad REF_FRAME .unsupported reference frame 'TOD'"),
        (legacy_cdm().replace("REF_FRAME", "XREF_FRAME"), "bad REF_FRAME"),
        (legacy_cdm().replace("CN_N = 25.0", "CN_N = abc", 1), "no numeric CN_N"),
    ],
)
def test_malformed_files_are_format_errors(
    tmp_path: Path, text: str, message: str
) -> None:
    with pytest.raises(CaraFormatError, match=message):
        load_legacy_objects(write(tmp_path, text))
