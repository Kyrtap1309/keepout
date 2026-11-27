"""The committed Iridium-Cosmos excerpt: that every file follows from the
data and the manifest, and that it holds what the case study reads."""

from collections import Counter

import iridium_cosmos as study
import iridium_cosmos_data as data_tool
import pytest


@pytest.fixture(scope="module")
def inputs() -> data_tool.Inputs:
    return data_tool.inputs_from_data(study.DATA)


def test_the_manifest_lists_every_file() -> None:
    manifest = data_tool.read_manifest(study.DATA)
    assert set(manifest["files"]) == {study.CATALOG_FILE, data_tool.NOTICE}
    assert sorted(path.name for path in study.DATA.iterdir()) == sorted(
        [study.CATALOG_FILE, data_tool.NOTICE, data_tool.MANIFEST]
    )


def test_every_file_follows_from_the_data(inputs: data_tool.Inputs) -> None:
    for name, stored in data_tool.stored_files(study.DATA, inputs).items():
        assert (study.DATA / name).read_bytes() == stored, name


def test_the_excerpt_is_what_the_manifest_says(inputs: data_tool.Inputs) -> None:
    selection = inputs.selection
    assert len(inputs.omms) == selection["sets"]
    assert len({omm.catalog_number for omm in inputs.omms}) == selection["objects"]


def test_one_set_per_other_object_known_at_the_decision_point(
    inputs: data_tool.Inputs,
) -> None:
    others = [omm for omm in inputs.omms if omm.catalog_number not in study.PAIR]
    assert max(Counter(omm.catalog_number for omm in others).values()) == 1
    assert all(omm.epoch <= study.IRIDIUM_AS_OF for omm in others)


def test_the_pair_spans_the_month_before() -> None:
    pair = study.read_pair()
    assert {omm.catalog_number for omm in pair} == set(study.PAIR)
    assert {omm.object_name for omm in pair} == set(study.NAMES.values())
    assert all(omm.epoch <= study.TCA_REFERENCE for omm in pair)
    spans = {
        number: [omm.epoch for omm in pair if omm.catalog_number == number]
        for number in study.PAIR
    }
    for epochs in spans.values():
        assert (max(epochs) - min(epochs)).to_value("day") > 25
