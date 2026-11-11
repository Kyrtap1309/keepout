"""Options several commands share, as annotated types for their parameters."""

from __future__ import annotations

import warnings
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TypeAlias

import typer

from keepout._validation import require_positive
from keepout.sources.catalog import CATALOG_SOURCES, DEFAULT_SOURCE, CatalogSourceName

# The library is imported in the functions: keepout --help need not load the
# numerical libraries.
if TYPE_CHECKING:
    from astropy.time import Time

    from keepout.pc import PcComputation
    from keepout.settings import Settings

__all__ = [
    "CacheDir",
    "HardBodyRadius",
    "JsonFlag",
    "SettingsFile",
    "SourceOption",
    "Workers",
    "dubious_years_ignored",
    "operator_settings",
    "parse_time",
    "positive",
    "time_option",
]


@contextmanager
def dubious_years_ignored() -> Generator[None]:
    """ERFA's "dubious year" warning ignored, and only it.

    ERFA calls a UTC year "dubious" outside its leap-second table, e.g. more
    than about five years past the table's release: the UTC-TAI offset is
    extrapolated there, but a time such as ``--before 2030-01-01`` is
    exact enough for a command line.
    """
    from erfa import ErfaWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r".*dubious year", ErfaWarning)
        yield


def parse_time(text: str) -> Time:
    """A time given on the command line: ISO 8601 in UTC, e.g.
    ``2026-10-06`` or ``2026-10-06T12:00:00``.

    Raises
    ------
    ValueError
        If ``text`` is not such a time (Typer makes it a usage error).
    """
    from astropy.time import Time

    try:
        with dubious_years_ignored():
            return Time(text, scale="utc")
    except ValueError as exc:
        raise ValueError(f"not an ISO 8601 UTC time: {text!r}") from exc


def time_option(text: str, option: str) -> Time:
    """The time a command-line option gives (:func:`parse_time`).

    Raises
    ------
    typer.BadParameter
        If it is not such a time: a usage error naming ``option``.
    """
    try:
        return parse_time(text)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint=f"'{option}'") from exc


JsonFlag: TypeAlias = Annotated[
    bool,
    typer.Option("--json", help="Print the result as one JSON document."),
]
"""``--json``: the result as JSON instead of tables and text."""

CacheDir: TypeAlias = Annotated[
    Path | None,
    typer.Option(
        "--cache-dir",
        envvar="KEEPOUT_CACHE_DIR",
        file_okay=False,
        help="Cache root for downloaded data (default: the user cache directory).",
    ),
]
"""``--cache-dir``: the cache root, also read from ``$KEEPOUT_CACHE_DIR``,
which ``--help`` shows."""

Workers: TypeAlias = Annotated[
    int | None,
    typer.Option(
        "--workers",
        min=1,
        help="Processes to use (default: one per CPU, at most 4).",
    ),
]
"""``--workers``: processes to compute in (``None``: the library's default,
one per CPU, at most 4)."""

SourceOption: TypeAlias = Annotated[
    CatalogSourceName | None,
    typer.Option(
        "--source",
        help=" ".join(
            [
                *(
                    f"{source.name}: {source.summary}."
                    for source in CATALOG_SOURCES.values()
                ),
                f"Default: {DEFAULT_SOURCE}.",
            ]
        ),
    ),
]
"""``--source``: where the catalog comes from (:data:`CATALOG_SOURCES`);
``None`` is :data:`~keepout.sources.catalog.DEFAULT_SOURCE`, so a command
can tell it apart from an explicit choice."""


def positive(value: float | None) -> float | None:
    """An option's number, checked to be finite and above zero (``None``
    passes).

    Raises
    ------
    typer.BadParameter
        If it is not: a usage error.
    """
    if value is not None:
        try:
            require_positive(value, "value")
        except ValueError as exc:
            raise typer.BadParameter(f"must be a positive number, got {value}") from exc
    return value


HardBodyRadius: TypeAlias = Annotated[
    float | None,
    typer.Option(
        "--hbr",
        callback=positive,
        metavar="METRES",
        help=(
            "Combined hard-body radius [m] (default: the settings' hbr_m, else "
            "the sum of the CDMs' HBR)."
        ),
    ),
]
"""``--hbr``: the combined hard-body radius instead of the CDMs'."""


SettingsFile: TypeAlias = Annotated[
    Path | None,
    typer.Option(
        "--settings",
        exists=True,
        dir_okay=False,
        metavar="FILE",
        help=(
            "Operator settings, TOML: [pc] method, policy, hbr_m; [risk] "
            "green_below, red_from; [maneuver] target_pc (default: the "
            "library's). An option given here wins over the file."
        ),
    ),
]
"""``--settings``: the operator's settings file
(:func:`keepout.settings.load_settings`)."""


def operator_settings(
    path: Path | None,
    *,
    compute: PcComputation | None = None,
    combined_hard_body_radius: float | None = None,
    target_pc: float | None = None,
) -> Settings:
    """The settings of ``--settings``, or the library's defaults, with the
    command line's own options over them (``None``: not given).

    Raises
    ------
    SettingsError
        If the file is not a settings file; the message names the key.
    """
    from dataclasses import replace

    from keepout.settings import Settings, load_settings

    settings = Settings() if path is None else load_settings(path)
    if compute is not None:
        settings = replace(settings, compute=compute)
    if combined_hard_body_radius is not None:
        settings = replace(
            settings, combined_hard_body_radius=combined_hard_body_radius
        )
    if target_pc is not None:
        settings = replace(settings, target_pc=target_pc)
    return settings
