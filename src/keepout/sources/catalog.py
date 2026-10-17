"""A GP catalog as the sources deliver it: element sets, one per object, and
the downloads they came from.

CelesTrak publishes its catalog in groups (one download each,
:func:`keepout.sources.celestrak.fetch_catalog`); Space-Track's ``gp``
query gives the whole catalog in one
(:func:`keepout.sources.spacetrack.fetch_catalog`). Both give a
:class:`Catalog` of :class:`CatalogPart` downloads.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from astropy.time import Time

from keepout.omm.model import Omm, deduplicate
from keepout.sources.cache import CopyMeta

__all__ = ["Catalog", "CatalogPart", "Retrieval"]


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
        downloads = tuple(parts)
        kept, dropped = deduplicate(omm for part in downloads for omm in part.omms)
        ordered = tuple(sorted(kept, key=lambda omm: omm.catalog_number))
        return cls(ordered, dropped, downloads)
