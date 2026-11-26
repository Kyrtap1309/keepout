"""The timing files import only what old trees of keepout have.

``tools/dashboard_timing.py`` and ``tools/screening_timing.py`` run with the
Python environment of an older tree (the benchmarks' ``compare``), so they
import nothing from ``tools/`` and, of keepout, only the names below: each
there since keepout had what the file times. Adding one to a list means the
benchmark can no longer time trees from before it, so check that first.
"""

import ast
from pathlib import Path
from typing import Final

import pytest
from generated import REPO

TOOLS: Final = REPO / "tools"

MINIMAL_API: Final = {
    # Since keepout.dashboard; of the store, only Store.ingest and
    # Store.cache are called, and of keepout.dashboard.event_data the event
    # page's data path (read_event, trend_of, trend_plot, latest_conjunction,
    # is_max_pc, encounter_plot, geometry_of, geometry_plot).
    "dashboard_timing.py": {
        ("keepout.cdm", "Cdm"),
        ("keepout.cdm", "ObjectMetadata"),
        ("keepout.cdm", "make_cdm"),
        ("keepout.conjunction", "Conjunction"),
        ("keepout.covariance", "Covariance"),
        ("keepout.dashboard", "event_data"),
        ("keepout.dashboard.listing", "read_page"),
        ("keepout.objects", "ObjectState"),
        ("keepout.orbits", "EARTH_MU"),
        ("keepout.propagation.forces", "EARTH_RADIUS"),
        ("keepout.risk", "PcObservation"),
        ("keepout.settings", "Settings"),
        ("keepout.state", "StateVector"),
        ("keepout.store", "CdmInput"),
        ("keepout.store", "Store"),
        ("keepout.store", "open_store"),
    },
    # Since keepout.screening.
    "screening_timing.py": {
        ("keepout.omm", "Omm"),
        ("keepout.omm", "load"),
        ("keepout.screening", "ScreeningResult"),
        ("keepout.screening", "ScreeningVolume"),
        ("keepout.screening", "VolumeRule"),
        ("keepout.screening", "screen"),
        ("keepout.screening.volume", "SOCRATES_RADIUS"),
        ("keepout.units", "DAY"),
    },
}
"""The keepout (module, name) pairs each timing file may import."""


def imports(path: Path) -> set[tuple[str, str]]:
    """The (module, name) pairs ``path`` imports; ``*`` for a whole module."""
    found: set[tuple[str, str]] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {(alias.name, "*") for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            found |= {(node.module or ".", alias.name) for alias in node.names}
    return found


@pytest.mark.parametrize("name", sorted(MINIMAL_API))
def test_a_timing_file_imports_only_the_minimal_api(name: str) -> None:
    keepout = {
        (module, item)
        for module, item in imports(TOOLS / name)
        if module.split(".")[0] == "keepout"
    }
    assert keepout == MINIMAL_API[name]


@pytest.mark.parametrize("name", sorted(MINIMAL_API))
def test_a_timing_file_imports_nothing_from_the_tools(name: str) -> None:
    tools = {path.stem for path in TOOLS.glob("*.py")}
    modules = {module.split(".")[0] for module, _ in imports(TOOLS / name)}
    assert modules & tools == set()
