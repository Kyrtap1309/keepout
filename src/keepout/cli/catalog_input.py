"""The catalog a command screens against: a file or a source."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TypeAlias

import typer

from keepout.sources.catalog import CATALOG_SOURCES, DEFAULT_SOURCE, CatalogSourceName

# The library is imported in the function: keepout --help need not load the
# numerical libraries.
if TYPE_CHECKING:
    from keepout.omm import Omm

__all__ = ["CatalogFile", "catalog_input", "require_one_catalog"]

CatalogFile: TypeAlias = Annotated[
    Path | None,
    typer.Option(
        "--catalog",
        exists=True,
        dir_okay=False,
        help=(
            "Screen against the element sets in this file (OMM KVN or XML, "
            "CelesTrak CSV or JSON, TLE; gzipped or not) instead of a source."
        ),
    ),
]
"""``--catalog``: a local catalog file instead of a source."""


def require_one_catalog(catalog: Path | None, source: CatalogSourceName | None) -> None:
    """A check that the command line names a catalog file or a source, not both.

    Raises
    ------
    typer.BadParameter
        If it names both: a usage error.
    """
    if catalog is not None and source is not None:
        raise typer.BadParameter(
            "give either --catalog or --source, not both", param_hint="--source"
        )


def catalog_input(
    catalog: Path | None, source: CatalogSourceName | None, cache_dir: Path | None
) -> tuple[tuple[Omm, ...], str]:
    """The element sets to screen against, from ``--catalog`` or a source,
    and how the summary names them."""
    if catalog is not None:
        from keepout.omm import load

        return load(catalog), str(catalog)
    described = CATALOG_SOURCES[source or DEFAULT_SOURCE]
    return described.fetch(cache_dir).omms, described.title
