"""The fleet of a synthetic operator: its satellites, read from a TOML file.

A fleet file names the satellites the operator protects, by catalog number,
with what GP data does not say about them: each satellite's hard-body
radius and whether it can manoeuvre, and the radius assumed for every
other (catalog) object. Their element sets come from the catalog they are
screened against. For example::

    # Radius assumed for every catalog object [m].
    catalog_hard_body_radius = 1.0

    [[satellites]]
    catalog_number = 25544
    name = "ISS"              # optional; else the element set's OBJECT_NAME
    hard_body_radius = 50.0   # radius of a sphere enclosing it [m]
    maneuverable = true

Every key but ``name`` is required and no other key is allowed, so a typo
is an error, not a silent default: a hard-body radius changes every Pc.
:func:`load_fleet` reads a file, :func:`read_fleet` its text; both raise
:class:`~keepout.errors.FleetError` naming the satellite and key at fault.
"""

from __future__ import annotations

import tomllib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from keepout._validation import (
    is_single_line,
    non_negative_float,
    positive_float,
)
from keepout.errors import FleetError
from keepout.omm.model import MAX_CATALOG_NUMBER

__all__ = [
    "FLEET_KEYS",
    "SATELLITE_KEYS",
    "Fleet",
    "FleetSatellite",
    "load_fleet",
    "read_fleet",
]

FLEET_KEYS: Final = frozenset({"catalog_hard_body_radius", "satellites"})
"""The keys of a fleet file; all required."""

SATELLITE_KEYS: Final = frozenset(
    {"catalog_number", "name", "hard_body_radius", "maneuverable"}
)
"""The keys of a satellite (a ``[[satellites]]`` table); all but ``name``
required."""

_OPTIONAL: Final = frozenset({"name"})


@dataclass(frozen=True, slots=True)
class FleetSatellite:
    """One satellite of the fleet.

    Parameters
    ----------
    catalog_number
        Its satellite catalog number (``NORAD_CAT_ID``).
    hard_body_radius
        Radius of a sphere enclosing it [m].
    maneuverable
        Whether it can manoeuvre (a CDM's ``MANEUVERABLE``).
    name
        Its name in the CDMs (``OBJECT_NAME``); if ``None``, the
        ``OBJECT_NAME`` of its element set.

    Raises
    ------
    ValueError
        If ``catalog_number`` is not in ``[1, MAX_CATALOG_NUMBER]``,
        ``hard_body_radius`` is not finite and positive, or ``name`` is
        empty, has several lines or leading or trailing blanks.
    """

    catalog_number: int
    hard_body_radius: float
    maneuverable: bool
    name: str | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.catalog_number <= MAX_CATALOG_NUMBER:
            raise ValueError(
                f"catalog_number must be in [1, {MAX_CATALOG_NUMBER}], "
                f"got {self.catalog_number}"
            )
        radius = positive_float(self.hard_body_radius, "hard_body_radius")
        object.__setattr__(self, "hard_body_radius", radius)
        name = self.name
        if name is not None and (not name or not is_single_line(name)):
            raise ValueError(
                f"name must be one non-empty line without leading or trailing "
                f"blanks, got {name!r}"
            )


@dataclass(frozen=True, slots=True)
class Fleet:
    """The satellites of a synthetic operator.

    Parameters
    ----------
    satellites
        The satellites, in the order given; at least one, each catalog
        number once.
    catalog_hard_body_radius
        Radius [m] assumed for every object that is not a fleet satellite
        (finite, >= 0: 0 puts the whole combined radius on the satellite).

    Raises
    ------
    ValueError
        If there is no satellite, a catalog number repeats, or
        ``catalog_hard_body_radius`` is negative or not finite.
    """

    satellites: tuple[FleetSatellite, ...]
    catalog_hard_body_radius: float

    def __post_init__(self) -> None:
        satellites = tuple(self.satellites)
        if not satellites:
            raise ValueError("a fleet needs at least one satellite")
        counts = Counter(satellite.catalog_number for satellite in satellites)
        repeated = sorted(number for number, count in counts.items() if count > 1)
        if repeated:
            raise ValueError(
                f"catalog_number repeated: {', '.join(map(str, repeated))}"
            )
        radius = non_negative_float(
            self.catalog_hard_body_radius, "catalog_hard_body_radius"
        )
        object.__setattr__(self, "satellites", satellites)
        object.__setattr__(self, "catalog_hard_body_radius", radius)

    @property
    def catalog_numbers(self) -> tuple[int, ...]:
        """The satellites' catalog numbers, in order."""
        return tuple(satellite.catalog_number for satellite in self.satellites)

    def find(self, catalog_number: int) -> FleetSatellite | None:
        """The satellite with this catalog number, or ``None`` if it is not
        in the fleet.

        Parameters
        ----------
        catalog_number
            NORAD catalog number of the satellite.
        """
        for satellite in self.satellites:
            if satellite.catalog_number == catalog_number:
                return satellite
        return None


def _keys(table: Mapping[str, object], allowed: frozenset[str], where: str) -> None:
    """A check that ``table`` has every required key of ``allowed`` and no other.

    Raises
    ------
    FleetError
        If ``table`` has a key outside ``allowed`` or lacks a required one.
    """
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise FleetError(
            f"{where}unknown key(s) {', '.join(unknown)}; expected "
            f"{', '.join(sorted(allowed))}"
        )
    missing = sorted(allowed - _OPTIONAL - set(table))
    if missing:
        raise FleetError(f"{where}missing key(s) {', '.join(missing)}")


def _number(table: Mapping[str, object], key: str, where: str) -> float:
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise FleetError(f"{where}{key} must be a number, got {value!r}")
    return float(value)


def _satellite(item: object, where: str) -> FleetSatellite:
    """The satellite of one ``[[satellites]]`` table.

    Raises
    ------
    FleetError
        If ``item`` is not a table, has unknown or missing keys, a value of
        the wrong type, or values :class:`FleetSatellite` rejects.
    """
    if not isinstance(item, dict):
        raise FleetError(f"{where}must be a table, got {item!r}")
    _keys(item, SATELLITE_KEYS, where)
    number = item["catalog_number"]
    if isinstance(number, bool) or not isinstance(number, int):
        raise FleetError(f"{where}catalog_number must be an integer, got {number!r}")
    maneuverable = item["maneuverable"]
    if not isinstance(maneuverable, bool):
        raise FleetError(
            f"{where}maneuverable must be true or false, got {maneuverable!r}"
        )
    name = item.get("name")
    if name is not None and not isinstance(name, str):
        raise FleetError(f"{where}name must be a string, got {name!r}")
    radius = _number(item, "hard_body_radius", where)
    try:
        return FleetSatellite(number, radius, maneuverable, name)
    except ValueError as exc:
        raise FleetError(f"{where}{exc}") from exc


def read_fleet(text: str) -> Fleet:
    """The fleet of a fleet file's text (TOML; see the module's description).

    Parameters
    ----------
    text
        The fleet file's text.

    Raises
    ------
    FleetError
        If the text is not TOML, a key is unknown or missing, or a value
        has the wrong type or is out of range; the message names the
        satellite (1-based, in file order) and the key.
    """
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise FleetError(f"not TOML: {exc}") from exc
    _keys(data, FLEET_KEYS, "")
    radius = _number(data, "catalog_hard_body_radius", "")
    items = data["satellites"]
    if not isinstance(items, list):
        raise FleetError(
            f"satellites must be an array of tables ([[satellites]]), got {items!r}"
        )
    satellites = tuple(
        _satellite(item, f"satellite {index}: ")
        for index, item in enumerate(items, start=1)
    )
    try:
        return Fleet(satellites, radius)
    except ValueError as exc:
        raise FleetError(str(exc)) from exc


def load_fleet(path: str | Path) -> Fleet:
    """The fleet of a fleet file (UTF-8 TOML; see :func:`read_fleet`).

    Parameters
    ----------
    path
        The fleet file.

    Raises
    ------
    FleetError
        As :func:`read_fleet`, or if the file is not UTF-8 text; the
        message starts with the file's path.
    OSError
        If the file cannot be read.
    """
    data = Path(path).read_bytes()
    try:
        return read_fleet(data.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise FleetError(f"{path}: not UTF-8 text: {exc}") from exc
    except FleetError as exc:
        raise FleetError(f"{path}: {exc}") from exc
