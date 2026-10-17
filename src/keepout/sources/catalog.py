"""A GP catalog as the sources deliver it: element sets, one per object, and
the downloads they came from.

CelesTrak publishes its catalog in groups (one download each,
:func:`keepout.sources.celestrak.fetch_catalog`); Space-Track's ``gp``
query gives the whole catalog in one
(:func:`keepout.sources.spacetrack.fetch_catalog`). Both give a
:class:`Catalog` of :class:`CatalogPart` downloads. :data:`CATALOG_SOURCES`
names them, what one download of each is and how long a copy is reused,
for the command line.

Importing this module loads no numerical library (``keepout --help``
reads :data:`CATALOG_SOURCES`); astropy loads when a :class:`Retrieval` is
made.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from keepout.units import HOUR

if TYPE_CHECKING:
    from astropy.time import Time

    from keepout.omm.model import Omm
    from keepout.sources.cache import CopyMeta

__all__ = [
    "CATALOG_SOURCES",
    "DEFAULT_SOURCE",
    "Catalog",
    "CatalogPart",
    "CatalogSource",
    "CatalogSourceName",
    "Retrieval",
]

CatalogSourceName: TypeAlias = Literal["celestrak", "spacetrack"]
"""A source of the whole GP catalog, a key of :data:`CATALOG_SOURCES`."""


@dataclass(frozen=True, slots=True)
class Retrieval:
    """Where and when a cached download came from.

    Parameters
    ----------
    url
        The URL it was downloaded from.
    path
        The cached copy.
    retrieved
        When the source sent this copy (UTC).
    checked
        When the source last confirmed it current: the download, or a later
        CelesTrak HTTP 403 for unchanged data (UTC).
    """

    url: str
    path: Path
    retrieved: Time
    checked: Time

    @classmethod
    def of(cls, path: Path, meta: CopyMeta) -> Retrieval:
        """The retrieval of the copy at ``path``, as its metadata records it."""
        from astropy.time import Time

        return cls(
            url=meta.url,
            path=path,
            retrieved=Time(meta.retrieved, format="unix", scale="utc"),
            checked=Time(meta.checked, format="unix", scale="utc"),
        )


@dataclass(frozen=True, slots=True)
class CatalogPart:
    """The element sets of one download of a catalog.

    Parameters
    ----------
    name
        What was downloaded: a CelesTrak GP group (``active``) or the
        Space-Track query class (``gp``).
    omms
        Its element sets, in the source's order.
    retrieval
        Where and when they were downloaded.
    """

    name: str
    omms: tuple[Omm, ...]
    retrieval: Retrieval


@dataclass(frozen=True, slots=True)
class Catalog:
    """The element sets of one or more downloads, one per object.

    Parameters
    ----------
    omms
        The newest element set of each object, by catalog number.
    duplicates
        Element sets left out because another set of the same object is
        newer (see :func:`keepout.omm.deduplicate`); exact copies are not
        listed.
    parts
        The downloads, in the order they were made.
    """

    omms: tuple[Omm, ...]
    duplicates: tuple[Omm, ...]
    parts: tuple[CatalogPart, ...]

    @classmethod
    def of(cls, parts: Iterable[CatalogPart]) -> Catalog:
        """The catalog of ``parts``: each object's newest element set, by
        catalog number."""
        from keepout.omm.model import deduplicate

        downloads = tuple(parts)
        kept, dropped = deduplicate(omm for part in downloads for omm in part.omms)
        ordered = tuple(sorted(kept, key=lambda omm: omm.catalog_number))
        return cls(ordered, dropped, downloads)


@dataclass(frozen=True, slots=True)
class CatalogSource:
    """A source of the whole GP catalog.

    Parameters
    ----------
    name
        How the command line names it (``--source``).
    title
        What it gives, as a table title.
    part_title
        What one download of it is (a :class:`CatalogPart`), as a column
        title: a CelesTrak group, a Space-Track query.
    summary
        What it is and needs, for the ``--source`` help.
    module
        The module whose ``fetch_catalog(*, cache)`` gets its catalog,
        imported on the first :meth:`fetch`.
    reuse
        Seconds a downloaded catalog is reused without asking the source
        again [s]: :data:`keepout.sources.celestrak.TTL`, Space-Track's
        ``gp`` interval (:data:`keepout.sources.spacetrack.CLASS_INTERVALS`).
    """

    name: CatalogSourceName
    title: str
    part_title: str
    summary: str
    module: str
    reuse: float

    def fetch(self, cache: Path | None = None) -> Catalog:
        """The catalog: its cached copy while that is younger than
        :attr:`reuse`, else a download.

        Raises
        ------
        KeepoutError
            As the module's ``fetch_catalog``.
        """
        fetch: Callable[..., Catalog] = importlib.import_module(
            self.module
        ).fetch_catalog
        return fetch(cache=cache)


CATALOG_SOURCES: Final[Mapping[CatalogSourceName, CatalogSource]] = MappingProxyType(
    {
        "celestrak": CatalogSource(
            "celestrak",
            "CelesTrak public GP groups",
            "Group",
            "public groups, no account",
            "keepout.sources.celestrak",
            2 * HOUR,
        ),
        "spacetrack": CatalogSource(
            "spacetrack",
            "Space-Track full GP catalog",
            "Query",
            "the full catalog, needs SPACETRACK_IDENTITY and SPACETRACK_PASSWORD",
            "keepout.sources.spacetrack",
            HOUR,
        ),
    }
)
"""The sources of the whole GP catalog, by name."""

DEFAULT_SOURCE: Final[CatalogSourceName] = "celestrak"
"""The catalog source used unless another is asked for: no account needed."""
