"""What ``keepout dashboard`` passes to the dashboard: the environment
variables, the upload limit, and the UTC times the command line and the
dashboard both read.

Below both :mod:`keepout.cli` and :mod:`keepout.dashboard`, which may not
import each other. Astropy and ERFA are imported in the functions:
``keepout --help`` does not load them.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Generator

    from astropy.time import Time

__all__ = [
    "AS_OF_VARIABLE",
    "CACHE_DIR_VARIABLE",
    "MAX_UPLOAD_MB",
    "SETTINGS_VARIABLE",
    "STORE_VARIABLE",
    "dubious_years_ignored",
    "parse_time",
]

STORE_VARIABLE: Final = "KEEPOUT_STORE"
"""The store's file: ``--store`` of the commands that read one, and the
dashboard's store."""

CACHE_DIR_VARIABLE: Final = "KEEPOUT_CACHE_DIR"
"""The cache root of downloaded data: ``--cache-dir``, and the library's
default (:func:`keepout.sources._store.cache_dir`)."""

SETTINGS_VARIABLE: Final = "KEEPOUT_SETTINGS"
"""The dashboard's operator settings file (TOML); unset: the library's
defaults."""

AS_OF_VARIABLE: Final = "KEEPOUT_AS_OF"
"""The time the dashboard sees its data at, ISO 8601 UTC; unset: the
clock."""

MAX_UPLOAD_MB: Final = 1
"""The largest file [MB] the dashboard takes (a CDM is a few kB): Streamlit's
upload limit, and the most bytes (2^20 per MB, as Streamlit counts) the
dashboard reads of a file or of its decompressed data."""


@contextmanager
def dubious_years_ignored() -> Generator[None]:
    """ERFA's "dubious year" warning ignored, and only it.

    ERFA calls a UTC year "dubious" outside its leap-second table, e.g. more
    than about five years past the table's release: the UTC-TAI offset is
    extrapolated there, but a time such as ``--before 2030-01-01`` is
    exact enough for a command line or a dashboard.
    """
    from erfa import ErfaWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r".*dubious year", ErfaWarning)
        yield


def parse_time(text: str) -> Time:
    """A time given on the command line or in the environment: ISO 8601 in
    UTC, e.g. ``2026-10-06``, ``2026-10-06T12:00:00`` or
    ``2026-10-06 12:00``.

    Raises
    ------
    ValueError
        If ``text`` is not such a time.
    """
    from astropy.time import Time

    # The ISO formats only: left to guess, Astropy also reads e.g. "J2026" or
    # "2026:279:12:00" (year, day of year).
    for time_format in ("isot", "iso"):
        try:
            with dubious_years_ignored():
                return Time(text, format=time_format, scale="utc")
        except ValueError:
            continue
    raise ValueError(f"not an ISO 8601 UTC time: {text!r}")
