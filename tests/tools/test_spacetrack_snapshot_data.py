"""The committed Space-Track GP catalog: that every file follows
from the catalog and the manifest."""

import pytest
import snapshot_files
import spacetrack_snapshot as snapshot

from keepout.sources.spacetrack import CATALOG

DATA = snapshot.DATA


@pytest.fixture(scope="module")
def manifest() -> snapshot.Manifest:
    return snapshot.read_manifest(DATA)


@pytest.fixture(scope="module")
def inputs() -> snapshot.Inputs:
    return snapshot.inputs_from_data(DATA)


def test_the_manifest_lists_every_file(manifest: snapshot.Manifest) -> None:
    assert set(manifest["files"]) == {snapshot.SNAPSHOT, snapshot.NOTICE}
    assert sorted(path.name for path in DATA.iterdir()) == sorted(
        [snapshot.SNAPSHOT, snapshot.NOTICE, snapshot_files.MANIFEST]
    )


def test_every_file_follows_from_the_data(inputs: snapshot.Inputs) -> None:
    # The manifest too, so its digests are those of the committed files.
    for name, stored in snapshot.stored_files(DATA, inputs).items():
        assert (DATA / name).read_bytes() == stored, name


def test_the_catalog_is_space_tracks_recommended_query(
    manifest: snapshot.Manifest, inputs: snapshot.Inputs
) -> None:
    assert manifest["download"]["url"] == CATALOG.url
    assert manifest["download"]["status"] == 200
    numbers = [omm.catalog_number for omm in inputs.omms]
    assert len(numbers) == manifest["catalog"]["objects"]
    assert numbers == sorted(set(numbers))
