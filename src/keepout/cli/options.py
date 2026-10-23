"""Options several commands share, as annotated types for their parameters."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, TypeAlias

import typer

from keepout._validation import require_positive
from keepout.sources.catalog import CATALOG_SOURCES, DEFAULT_SOURCE, CatalogSourceName

__all__ = [
    "CacheDir",
    "JsonFlag",
    "SourceOption",
    "Workers",
    "positive",
]


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
"""``--cache-dir``: the cache root (:func:`keepout.sources._store.cache_dir`),
also read from ``$KEEPOUT_CACHE_DIR``, which ``--help`` shows."""

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
