from pathlib import Path

import numpy as np
import pytest
import scipy.io

from keepout.errors import CaraFormatError
from keepout.sources.cara import load_mat


def test_load_mat_returns_user_variables(tmp_path: Path) -> None:
    path = tmp_path / "vars.mat"
    scipy.io.savemat(path, {"a": np.arange(3.0), "b": 2.5})
    data = load_mat(path)
    assert set(data) == {"a", "b"}
    np.testing.assert_array_equal(data["a"], [0.0, 1.0, 2.0])
    assert data["b"] == 2.5


def test_load_mat_exposes_structs_as_attributes(tmp_path: Path) -> None:
    path = tmp_path / "struct.mat"
    scipy.io.savemat(path, {"s": {"hbr": 20.0}})
    assert load_mat(path)["s"].hbr == 20.0


def test_load_mat_rejects_v73(tmp_path: Path) -> None:
    path = tmp_path / "v73.mat"
    header = b"MATLAB 7.3 MAT-file, Platform: GLNXA64".ljust(116, b" ")
    path.write_bytes(header + b"\x00" * 8 + b"\x00\x02IM")
    with pytest.raises(CaraFormatError, match=r"v73\.mat.*v7\.3"):
        load_mat(path)


def test_load_mat_rejects_non_mat_file(tmp_path: Path) -> None:
    path = tmp_path / "garbage.mat"
    path.write_bytes(b"not a mat file at all" * 10)
    with pytest.raises(CaraFormatError, match="not a readable MAT-file"):
        load_mat(path)


def test_load_mat_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.mat"
    path.write_bytes(b"")
    with pytest.raises(CaraFormatError, match="not a readable MAT-file"):
        load_mat(path)


def test_load_mat_missing_file_is_not_masked(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_mat(tmp_path / "missing.mat")
