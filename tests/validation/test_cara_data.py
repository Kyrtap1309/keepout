import pytest

from keepout.sources import cara

pytestmark = pytest.mark.validation


@pytest.mark.parametrize("file", cara.MANIFEST, ids=lambda f: f.path.rsplit("/", 1)[-1])
def test_manifest_file_downloads_and_verifies(file: cara.CaraFile) -> None:
    assert cara.fetch(file).is_file()
