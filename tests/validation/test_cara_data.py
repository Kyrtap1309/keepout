import pytest

from keepout.sources import cara

pytestmark = pytest.mark.validation


@pytest.mark.parametrize("file", cara.MANIFEST, ids=lambda f: f.path.rsplit("/", 1)[-1])
def test_manifest_file_downloads_and_verifies(file: cara.CaraFile) -> None:
    assert cara.fetch(file).is_file()


def test_monte_carlo_ocmdb_loads() -> None:
    data = cara.load_mat(cara.fetch(cara.MONTE_CARLO_OCMDB))
    assert data["DB"].shape == (2, 218)
