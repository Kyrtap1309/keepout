"""SOCRATES Plus close-approach reports: the rows, in SI units.

CelesTrak's SOCRATES (https://celestrak.org/SOCRATES/) screens the objects
of its GP data with SGP4 and lists, for a week, the approaches within 5 km.
:func:`keepout.sources.celestrak.fetch_socrates` downloads the latest report
as CSV; :func:`read_socrates` (or
:meth:`~keepout.sources.celestrak.SocratesReport.approaches`) reads it into
:class:`SocratesApproach` rows.

SOCRATES ran on the element sets of its own time: TCA minus ``DSE`` gives
each object's epoch, to the 43 s of DSE's three decimals.
:func:`epoch_matches` tells whether two element sets are the ones an
approach was computed from.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Final

from astropy.time import Time

from keepout.ccsds.files import strip_bom
from keepout.errors import SourceFormatError
from keepout.omm import Omm
from keepout.units import DAY, KM, julian_seconds_between

__all__ = [
    "COLUMNS",
    "SocratesApproach",
    "epoch_matches",
    "read_socrates",
]

COLUMNS: Final = (
    "NORAD_CAT_ID_1",
    "OBJECT_NAME_1",
    "DSE_1",
    "NORAD_CAT_ID_2",
    "OBJECT_NAME_2",
    "DSE_2",
    "TCA",
    "TCA_RANGE",
    "TCA_RELATIVE_SPEED",
    "MAX_PROB",
    "DILUTION",
)
"""The columns of a SOCRATES CSV report, in order."""

_TCA_TOLERANCE: Final = 0.5e-3
"""Half the last digit of a TCA in the report [s]: TCA is to the millisecond."""


@dataclass(frozen=True, slots=True)
class SocratesApproach:
    """One row of a SOCRATES report: a close approach of two objects.

    Parameters
    ----------
    primary_number, secondary_number
        Catalog numbers of the two objects (``NORAD_CAT_ID_1``, ``_2``).
    since_epoch_primary, since_epoch_secondary
        Time from each object's element set epoch to TCA [s] (``DSE_1``,
        ``DSE_2``, in days in the report).
    since_epoch_tolerance
        Half the last digit of the DSE values plus half a millisecond of
        TCA [s]: how well the report gives the epochs.
    tca
        Time of closest approach (UTC).
    range_at_tca
        Range at TCA [m] (``TCA_RANGE``, km in the report).
    relative_speed
        Relative speed at TCA [m/s] (``TCA_RELATIVE_SPEED``, km/s).
    line
        The row as the report gives it, line ending included.
    """

    primary_number: int
    secondary_number: int
    since_epoch_primary: float
    since_epoch_secondary: float
    since_epoch_tolerance: float
    tca: Time
    range_at_tca: float
    relative_speed: float
    line: str


def read_socrates(text: str) -> tuple[SocratesApproach, ...]:
    """The approaches of a SOCRATES CSV report, in file order.

    Parameters
    ----------
    text
        The report as CelesTrak sends it: a header of :data:`COLUMNS`, which
        may start with a byte-order mark, and one line per approach.

    Raises
    ------
    SourceFormatError
        If the header is not the report's, a row has another number of
        values, or a value cannot be read.
    """
    header, *lines = text.splitlines(keepends=True) or [""]
    columns = next(csv.reader([strip_bom(header)]), [])
    if tuple(columns) != COLUMNS:
        raise SourceFormatError(f"not a SOCRATES report header: {header.strip()!r}")
    rows = [next(csv.reader([line])) for line in lines]
    for number, row in enumerate(rows, start=2):
        if len(row) != len(COLUMNS):
            raise SourceFormatError(
                f"line {number}: {len(row)} values, not {len(COLUMNS)}"
            )
    records: list[dict[str, str]] = [
        dict(zip(COLUMNS, row, strict=True)) for row in rows
    ]
    try:
        tcas = Time([record["TCA"] for record in records], format="iso", scale="utc")
        return tuple(
            _approach(record, tcas[index], line)
            for index, (record, line) in enumerate(zip(records, lines, strict=True))
        )
    except ValueError as exc:
        raise SourceFormatError(f"cannot read a SOCRATES report: {exc}") from exc


def _approach(record: dict[str, str], tca: Time, line: str) -> SocratesApproach:
    """The approach of one row, by column name, at its parsed ``tca``.

    Raises
    ------
    ValueError
        If a number cannot be read.
    """
    # The last digit of the DSE values [days], as the report rounds them.
    dse_digit = 10.0 ** -max(_decimals(record["DSE_1"]), _decimals(record["DSE_2"]))
    return SocratesApproach(
        primary_number=int(record["NORAD_CAT_ID_1"]),
        secondary_number=int(record["NORAD_CAT_ID_2"]),
        since_epoch_primary=float(record["DSE_1"]) * DAY,
        since_epoch_secondary=float(record["DSE_2"]) * DAY,
        since_epoch_tolerance=0.5 * dse_digit * DAY + _TCA_TOLERANCE,
        tca=tca,
        # The report's km and km/s, in m and m/s.
        range_at_tca=float(record["TCA_RANGE"]) * KM,
        relative_speed=float(record["TCA_RELATIVE_SPEED"]) * KM,
        line=line,
    )


def _decimals(number: str) -> int:
    return len(number.partition(".")[2])


def epoch_matches(approach: SocratesApproach, primary: Omm, secondary: Omm) -> bool:
    """Whether SOCRATES used these element sets: TCA minus DSE is their epoch.

    Computed from UTC Julian days: a leap second between epoch and TCA
    would count one second, far inside the tolerance.

    Parameters
    ----------
    approach
        A row of the report.
    primary, secondary
        Element sets of its two objects.
    """
    return all(
        abs(julian_seconds_between(omm.epoch, approach.tca) - since)
        <= approach.since_epoch_tolerance
        for omm, since in (
            (primary, approach.since_epoch_primary),
            (secondary, approach.since_epoch_secondary),
        )
    )
