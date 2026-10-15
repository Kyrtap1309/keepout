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

from keepout._units import DAY, KM, seconds_between
from keepout.errors import SourceFormatError
from keepout.omm import Omm

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


@dataclass(frozen=True)
class SocratesApproach:
    """One row of a SOCRATES report: a close approach of two objects.

    Parameters
    ----------
    primary, secondary
        Catalog numbers of the two objects (``NORAD_CAT_ID_1``, ``_2``).
    since_epoch_primary, since_epoch_secondary
        Time from each object's element set epoch to TCA [s] (``DSE_1``,
        ``DSE_2``, in days in the report).
    since_epoch_tolerance
        Half the last digit of the DSE values plus half a millisecond of
        TCA [s]: how well the report gives the epochs.
    tca
        Time of closest approach (UTC).
    miss_distance
        Range at TCA [m] (``TCA_RANGE``, km in the report).
    relative_speed
        Relative speed at TCA [m/s] (``TCA_RELATIVE_SPEED``, km/s).
    line
        The row as the report gives it, line ending included.
    """

    primary: int
    secondary: int
    since_epoch_primary: float
    since_epoch_secondary: float
    since_epoch_tolerance: float
    tca: Time
    miss_distance: float
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
    columns = next(csv.reader([header.removeprefix("\ufeff")]), [])
    if tuple(columns) != COLUMNS:
        raise SourceFormatError(f"not a SOCRATES report header: {header.strip()!r}")
    rows = [next(csv.reader([line])) for line in lines]
    for number, row in enumerate(rows, start=2):
        if len(row) != len(COLUMNS):
            raise SourceFormatError(
                f"line {number}: {len(row)} values, not {len(COLUMNS)}"
            )
    try:
        tcas = Time([row[6] for row in rows], format="iso", scale="utc")
        return tuple(
            SocratesApproach(
                primary=int(row[0]),
                secondary=int(row[3]),
                since_epoch_primary=float(row[2]) * DAY,
                since_epoch_secondary=float(row[5]) * DAY,
                since_epoch_tolerance=(
                    0.5 * 10.0 ** -max(_decimals(row[2]), _decimals(row[5])) * DAY
                    + _TCA_TOLERANCE
                ),
                tca=tcas[index],
                miss_distance=float(row[7]) * KM,
                relative_speed=float(row[8]) * KM,
                line=line,
            )
            for index, (row, line) in enumerate(zip(rows, lines, strict=True))
        )
    except ValueError as exc:
        raise SourceFormatError(f"cannot read a SOCRATES report: {exc}") from exc


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
        abs(seconds_between(omm.epoch, approach.tca) - since)
        <= approach.since_epoch_tolerance
        for omm, since in (
            (primary, approach.since_epoch_primary),
            (secondary, approach.since_epoch_secondary),
        )
    )
