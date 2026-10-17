"""CelesTrak test data: a sample of element sets and SOCRATES report lines."""

from pathlib import Path
from typing import Final

from keepout.omm import Omm
from keepout.omm.celestrak import load_csv

SAMPLE: Final = load_csv(
    Path(__file__).parent / "omm" / "data" / "celestrak_sample.csv"
)
"""Eight element sets of CelesTrak CSV: 900 (LEO), 25544 ISS, 51850 (GEO),
30798 (HEO), 38745 (MEO), 100000, 100001 and 100882 (docked to the ISS,
same epoch)."""

SOCRATES_HEADER: Final = (
    "NORAD_CAT_ID_1,OBJECT_NAME_1,DSE_1,NORAD_CAT_ID_2,OBJECT_NAME_2,DSE_2,"
    "TCA,TCA_RANGE,TCA_RELATIVE_SPEED,MAX_PROB,DILUTION\r\n"
)
"""SOCRATES's header; the report has CRLF line endings."""

TCA: Final = "2026-10-07 00:20:47.861"
"""One day (to the millisecond) after the epoch of the ISS and Crew Dragon of
:data:`SAMPLE`."""


def socrates_row(
    first: Omm, second: Omm, dse: tuple[str, str], distance: str, speed: str
) -> str:
    """A SOCRATES line at :data:`TCA`: DSE in days, range in km, speed in km/s."""
    return (
        f"{first.catalog_number},{first.object_name} [+],{dse[0]},"
        f"{second.catalog_number},{second.object_name} [-],{dse[1]},"
        f"{TCA},{distance},{speed},1.000E-05,0.010\r\n"
    )
