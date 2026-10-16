"""The fleet of a synthetic operator: its TOML file, read strictly, and the
example fleet."""

import json
import pickle
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from screening_data import DATA, SNAPSHOT

from keepout.errors import FleetError
from keepout.omm import load
from keepout.omm.model import MAX_CATALOG_NUMBER
from keepout.synthetic import (
    Fleet,
    FleetSatellite,
    OrbitRegime,
    load_fleet,
    orbit_regime,
    read_fleet,
)

EXAMPLE = Path(__file__).parents[2] / "examples" / "synthetic-operator" / "fleet.toml"
"""The example fleet of the documentation and the CLI."""

TEXT = """
# Radius assumed for every catalog object [m].
catalog_hard_body_radius = 1.5

[[satellites]]
catalog_number = 25544
name = "ISS"
hard_body_radius = 50
maneuverable = true

[[satellites]]
catalog_number = 270214
hard_body_radius = 2.5
maneuverable = false
"""


def test_a_fleet_file_is_read() -> None:
    fleet = read_fleet(TEXT)
    assert fleet == Fleet(
        (
            FleetSatellite(25544, 50.0, maneuverable=True, name="ISS"),
            FleetSatellite(270214, 2.5, maneuverable=False),
        ),
        catalog_hard_body_radius=1.5,
    )
    assert isinstance(fleet.satellites[0].hard_body_radius, float)
    assert fleet.catalog_numbers == (25544, 270214)
    assert fleet.find(270214) is fleet.satellites[1]
    assert fleet.find(25545) is None
    assert pickle.loads(pickle.dumps(fleet)) == fleet
    assert hash(fleet) == hash(read_fleet(TEXT))


def satellite(**changes: object) -> str:
    keys: dict[str, object] = {
        "catalog_number": 25544,
        "hard_body_radius": 50.0,
        "maneuverable": True,
        **changes,
    }
    lines = [
        f"{key} = {str(value).lower() if isinstance(value, bool) else value}"
        for key, value in keys.items()
        if not isinstance(value, Omitted)
    ]
    return "[[satellites]]\n" + "\n".join(lines) + "\n"


class Omitted:
    """A key left out of a satellite."""


def fleet_text(*satellites: str, radius: object = "1.0") -> str:
    return f"catalog_hard_body_radius = {radius}\n" + "".join(satellites)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("catalog_hard_body_radius = ", "not TOML: "),
        (fleet_text(satellite()) + "fleet = 1\n", "satellite 1: unknown key"),
        (
            "fleet = 1\n" + fleet_text(satellite()),
            "^unknown key\\(s\\) fleet; expected",
        ),
        ("catalog_hard_body_radius = 1.0\n", "^missing key\\(s\\) satellites$"),
        (fleet_text(radius='"1"') + satellite(), "catalog_hard_body_radius must be a"),
        (fleet_text(radius="-1.0") + satellite(), "catalog_hard_body_radius must be "),
        ("catalog_hard_body_radius = 1.0\nsatellites = 3\n", "an array of tables"),
        ("catalog_hard_body_radius = 1.0\nsatellites = []\n", "at least one satellite"),
        ("catalog_hard_body_radius = 1.0\nsatellites = [1]\n", "1: must be a table"),
        (
            fleet_text(satellite(radius=5.0)),
            "^satellite 1: unknown key\\(s\\) radius; expected catalog_number, "
            "hard_body_radius, maneuverable, name$",
        ),
        (
            fleet_text(
                satellite(), satellite(catalog_number=1, maneuverable=Omitted())
            ),
            "^satellite 2: missing key\\(s\\) maneuverable$",
        ),
        (
            fleet_text(satellite(catalog_number='"25544"')),
            "^satellite 1: catalog_number must be an integer, got '25544'$",
        ),
        (fleet_text(satellite(catalog_number="true")), "must be an integer, got True"),
        (fleet_text(satellite(catalog_number=2.0)), "must be an integer, got 2.0"),
        (
            fleet_text(satellite(catalog_number=0)),
            f"^satellite 1: catalog_number must be in \\[1, {MAX_CATALOG_NUMBER}\\]",
        ),
        (fleet_text(satellite(hard_body_radius='"big"')), "radius must be a number"),
        (fleet_text(satellite(hard_body_radius="true")), "radius must be a number"),
        (
            fleet_text(satellite(hard_body_radius=0)),
            "^satellite 1: hard_body_radius must be finite and > 0, got 0.0$",
        ),
        (fleet_text(satellite(hard_body_radius="nan")), "must be finite and > 0"),
        (fleet_text(satellite(maneuverable='"YES"')), "must be true or false, got"),
        (fleet_text(satellite(name=5)), "^satellite 1: name must be a string, got 5$"),
        (fleet_text(satellite(name='""')), "name must be one non-empty line"),
        (fleet_text(satellite(name='" ISS"')), "name must be one non-empty line"),
        (fleet_text(satellite(name='"ISS\\nZARYA"')), "name must be one non-empty"),
        (
            fleet_text(satellite(), satellite(), satellite(catalog_number=1)),
            "^catalog_number repeated: 25544$",
        ),
    ],
)
def test_a_fleet_file_is_read_strictly(text: str, message: str) -> None:
    with pytest.raises(FleetError, match=message):
        read_fleet(text)


def test_a_fleet_is_checked_without_a_file() -> None:
    with pytest.raises(ValueError, match="at least one satellite"):
        Fleet((), 1.0)
    with pytest.raises(ValueError, match="catalog_number must be in"):
        FleetSatellite(MAX_CATALOG_NUMBER + 1, 1.0, maneuverable=True)
    # A point-sized catalog object leaves the whole radius to the satellite.
    assert Fleet((FleetSatellite(1, 1.0, maneuverable=True),), 0).find(1) is not None


def test_a_fleet_file_is_loaded_with_its_path_in_errors(tmp_path: Path) -> None:
    path = tmp_path / "fleet.toml"
    path.write_text(TEXT, encoding="utf-8")
    assert load_fleet(path) == read_fleet(TEXT)
    assert load_fleet(str(path)) == read_fleet(TEXT)
    path.write_text(fleet_text(satellite(name=5)), encoding="utf-8")
    with pytest.raises(FleetError, match=f"^{path}: satellite 1: name must be"):
        load_fleet(path)
    path.write_bytes(b"\xff" + TEXT.encode())
    with pytest.raises(FleetError, match=f"^{path}: not UTF-8 text"):
        load_fleet(path)
    with pytest.raises(FileNotFoundError):
        load_fleet(tmp_path / "missing.toml")


def toml(fleet: Fleet) -> str:
    """The fleet as a fleet file: Python floats and JSON strings of printable
    characters are TOML."""
    text = f"catalog_hard_body_radius = {fleet.catalog_hard_body_radius!r}\n"
    for item in fleet.satellites:
        text += "[[satellites]]\n"
        text += f"catalog_number = {item.catalog_number}\n"
        if item.name is not None:
            text += f"name = {json.dumps(item.name, ensure_ascii=False)}\n"
        text += f"hard_body_radius = {item.hard_body_radius!r}\n"
        text += f"maneuverable = {str(item.maneuverable).lower()}\n"
    return text


names = st.text(
    st.characters(codec="utf-8", exclude_categories=("Cc", "Cs", "Zl", "Zp")),
    min_size=1,
    max_size=20,
).filter(lambda name: name == name.strip() and len(name.splitlines()) == 1)
satellites = st.builds(
    FleetSatellite,
    catalog_number=st.integers(1, MAX_CATALOG_NUMBER),
    hard_body_radius=st.floats(1e-3, 1e3),
    maneuverable=st.booleans(),
    name=st.none() | names,
)
fleets = st.builds(
    Fleet,
    st.lists(satellites, min_size=1, max_size=4, unique_by=lambda s: s.catalog_number),
    catalog_hard_body_radius=st.floats(0.0, 1e3),
)


@given(fleets)
def test_every_fleet_reads_back_from_its_file(fleet: Fleet) -> None:
    assert read_fleet(toml(fleet)) == fleet


def test_the_example_fleet_is_in_the_frozen_catalog() -> None:
    fleet = load_fleet(EXAMPLE)
    by_number = {omm.catalog_number: omm for omm in load(DATA / SNAPSHOT)}
    elements = [by_number[number] for number in fleet.catalog_numbers]
    assert [(omm.catalog_number, omm.object_name) for omm in elements] == [
        (25544, "ISS (ZARYA)"),
        (38771, "METOP-B"),
        (44476, "INTELSAT 39 (IS-39)"),
    ]
    assert [item.name for item in fleet.satellites] == [
        omm.object_name for omm in elements
    ]
    assert [orbit_regime(omm) for omm in elements] == [
        OrbitRegime.LEO,
        OrbitRegime.LEO,
        OrbitRegime.GEO,
    ]
    assert [item.hard_body_radius for item in fleet.satellites] == [50.0, 10.0, 20.0]
    assert all(item.maneuverable for item in fleet.satellites)
    assert fleet.catalog_hard_body_radius == 1.0
