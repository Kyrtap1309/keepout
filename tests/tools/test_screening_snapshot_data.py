"""The committed screening data: that every file follows from
the catalog, the SOCRATES excerpt and the manifest."""

from typing import Any

import pytest
import screening_snapshot as snapshot

from keepout.omm import Omm, load
from keepout.sources.socrates import read_socrates

DATA = snapshot.DATA


@pytest.fixture(scope="module")
def manifest() -> snapshot.Manifest:
    return snapshot.read_manifest(DATA)


@pytest.fixture(scope="module")
def inputs() -> snapshot.Inputs:
    return snapshot.inputs_from_data(DATA)


def test_the_manifest_lists_the_files(manifest: Any) -> None:
    assert set(manifest["files"]) == {
        snapshot.SNAPSHOT,
        snapshot.SOCRATES,
        snapshot.FIXTURE,
        snapshot.NOTICE,
    }


def test_every_file_follows_from_the_data(inputs: snapshot.Inputs) -> None:
    # The manifest too, so its digests are those of the committed files.
    for name, stored in snapshot.stored_files(DATA, inputs).items():
        assert (DATA / name).read_bytes() == stored, name


def test_the_catalog_reads_back(manifest: Any, inputs: snapshot.Inputs) -> None:
    # inputs_from_data reads it with keepout.omm.load.
    numbers = [omm.catalog_number for omm in inputs.omms]
    assert len(numbers) == manifest["catalog"]["objects"]
    assert numbers == sorted(set(numbers))


def test_the_fixture_is_part_of_the_catalog(
    manifest: Any, inputs: snapshot.Inputs
) -> None:
    fixture: tuple[Omm, ...] = load(DATA / snapshot.FIXTURE)
    assert len(fixture) == manifest["fixture"]["objects"]
    by_number = {omm.catalog_number: omm for omm in inputs.omms}
    assert all(by_number[omm.catalog_number] == omm for omm in fixture)


def test_the_excerpt_holds_the_comparable_approaches(
    manifest: Any, inputs: snapshot.Inputs
) -> None:
    approaches = read_socrates(inputs.socrates)
    assert len(approaches) == manifest["socrates"]["comparable"]
    assert snapshot.comparable(approaches, inputs.omms) == list(approaches)
