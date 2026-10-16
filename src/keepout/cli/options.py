"""Options several commands share, as annotated types for their parameters."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, TypeAlias

import typer

__all__ = ["CacheDir", "JsonFlag", "Workers"]

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
