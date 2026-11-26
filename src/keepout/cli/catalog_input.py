"""The catalog a command screens against: a file or a source."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NamedTuple, TypeAlias

import typer

from keepout.sources.catalog import (
    CATALOG_SOURCES,
    DEFAULT_SOURCE,
    CatalogReference,
    CatalogSourceName,
)

# The library is imported in the function: keepout --help need not load the
# numerical libraries.
if TYPE_CHECKING:
    from keepout.omm import Omm

__all__ = ["CatalogFile", "CatalogInput", "catalog_input", "require_one_catalog"]

CatalogFile: TypeAlias = Annotated[
    Path | None,
    typer.Option(
        "--catalog",
        exists=True,
        dir_okay=False,
        metavar="FILE",
        help=(
            "Catalog file of element sets to screen against (OMM KVN or XML, "
            "CelesTrak CSV or JSON, TLE; gzipped or not), instead of --source."
        ),
    ),
]
"""``--catalog``: a local catalog file instead of a source."""


class CatalogInput(NamedTuple):
    """The element sets a command screens against."""

    omms: tuple[Omm, ...]
    label: str
    """How the summary names them: the file, or the source's title."""
    reference: CatalogReference
    """The files they were read from, with their digests: what a store
    keeps with the CDMs screened against them."""


def require_one_catalog(
    ctx: typer.Context, catalog: Path | None, source: CatalogSourceName | None
) -> None:
    """A check that the command line names a catalog file or a source, not
    both; a usage error if not."""
    if catalog is not None and source is not None:
        ctx.fail("give either --catalog or --source, not both")


def catalog_input(
    catalog: Path | None, source: CatalogSourceName | None, cache_dir: Path | None
) -> CatalogInput:
    """The element sets to screen against, from ``--catalog`` or a source,
    and the reference of the files they were read from.

    A ``--catalog`` file's digest is taken before it is read, and a
    source's is the one its cache verified on the bytes it read: either
    way, a file changed meanwhile no longer matches its reference.
    """
    if catalog is not None:
        from keepout.omm import load

        reference = CatalogReference.of_file(catalog)
        return CatalogInput(load(catalog), str(catalog), reference)
    described = CATALOG_SOURCES[source or DEFAULT_SOURCE]
    fetched = described.fetch(cache_dir)
    return CatalogInput(
        fetched.omms, described.title, fetched.reference(described.title)
    )
