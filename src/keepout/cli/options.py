"""Options several commands share, as annotated types for their parameters."""

from __future__ import annotations

import enum
import math
from pathlib import Path
from typing import Annotated, TypeAlias

import typer

__all__ = [
    "CacheDir",
    "CatalogSource",
    "JsonFlag",
    "Workers",
    "positive",
]


class CatalogSource(enum.StrEnum):
    """Where the GP catalog comes from."""

    CELESTRAK = "celestrak"
    """CelesTrak's public groups: no account, about 20 000 objects."""
    SPACETRACK = "spacetrack"
    """Space-Track's full catalog: an account, about 30 000 objects."""


JsonFlag: TypeAlias = Annotated[
    bool,
    typer.Option("--json", help="Print the result as one JSON document."),
]
"""``--json``: the result as JSON instead of tables and text."""

CacheDir: TypeAlias = Annotated[
    Path | None,
    typer.Option(
        "--cache-dir",
        file_okay=False,
        help=(
            "Cache root for downloaded data (default: $KEEPOUT_CACHE_DIR, "
            "else the user cache directory)."
        ),
    ),
]
"""``--cache-dir``: the cache root (:func:`keepout.sources.cache.cache_dir`)."""

Workers: TypeAlias = Annotated[
    int | None,
    typer.Option("--workers", min=1, help="Processes to use (default: one per CPU)."),
]
"""``--workers``: processes to compute in (``None``: one per CPU)."""


def positive(value: float | None) -> float | None:
    """Check an option's number is finite and above zero (``None`` passes).

    Raises
    ------
    typer.BadParameter
        If it is not: a usage error.
    """
    if value is not None and not (math.isfinite(value) and value > 0.0):
        raise typer.BadParameter(f"must be a positive number, got {value}")
    return value
