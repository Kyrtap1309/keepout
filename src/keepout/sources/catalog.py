"""A GP catalog as the sources deliver it: element sets, one per object, and
the downloads they came from.

CelesTrak publishes its catalog in groups (one download each,
:func:`keepout.sources.celestrak.fetch_catalog`); Space-Track's ``gp``
query gives the whole catalog in one
(:func:`keepout.sources.spacetrack.fetch_catalog`). Both give a
:class:`Catalog` of :class:`CatalogPart` downloads. :data:`CATALOG_SOURCES`
names them, what one download of each is and how long a copy is reused,
for the command line.

A :class:`CatalogReference` names the files of a catalog with their SHA-256
digests (:meth:`Catalog.reference`, :meth:`CatalogReference.of_file`), so
that the same catalog is read again later (:func:`load_reference`), or not
at all if a file changed.

Importing this module loads no numerical library (``keepout --help``
reads :data:`CATALOG_SOURCES`); astropy loads when a :class:`Retrieval` is
made.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from keepout._defaults import CELESTRAK_TTL, GP_INTERVAL
from keepout.errors import ChecksumMismatch, SourceUnavailable

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping

    from astropy.time import Time

    from keepout.omm.model import Omm
    from keepout.sources.cache import CopyMeta

__all__ = [
    "CATALOG_SOURCES",
    "DEFAULT_SOURCE",
    "Catalog",
    "CatalogFileRef",
    "CatalogPart",
    "CatalogReference",
    "CatalogSource",
    "CatalogSourceName",
    "Retrieval",
    "load_reference",
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
    sha256
        The SHA-256 (hex) of the copy the cache verified when it read it:
        the bytes the element sets were parsed from; ``None`` if unknown.
    """

    url: str
    path: Path
    retrieved: Time
    checked: Time
    sha256: str | None = None

    @classmethod
    def of(cls, path: Path, meta: CopyMeta) -> Retrieval:
        """The retrieval of the copy at ``path``, as its metadata records it.

        Parameters
        ----------
        path
            The cached copy.
        meta
            Its metadata, from :func:`~keepout.sources.cache.read_copy`.

        Returns
        -------
        Retrieval
            The URL, time and digest of the download.
        """
        from astropy.time import Time

        return cls(
            url=meta.url,
            path=path,
            retrieved=Time(meta.retrieved, format="unix", scale="utc"),
            checked=Time(meta.checked, format="unix", scale="utc"),
            sha256=meta.sha256,
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


@contextlib.contextmanager
def _reading(path: Path) -> Generator[None]:
    """A context reading the catalog file ``path``.

    Raises
    ------
    SourceUnavailable
        On an :class:`OSError` in the context.
    """
    try:
        yield
    except OSError as exc:
        raise SourceUnavailable(f"cannot read the catalog file {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    """The SHA-256 (hex) of the file at ``path``.

    Raises
    ------
    SourceUnavailable
        If it cannot be read (e.g. it is gone).
    """
    with _reading(path), path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _verified(file: CatalogFileRef) -> bytes:
    """The bytes of a referenced file, checked against its digest.

    Raises
    ------
    SourceUnavailable
        If the file is gone or cannot be read.
    ChecksumMismatch
        If its digest is not the one referenced.
    """
    with _reading(file.path):
        data = file.path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != file.sha256:
        raise ChecksumMismatch(
            f"the catalog file {file.path} changed since it was referenced: "
            f"SHA-256 {digest}, expected {file.sha256}"
        )
    return data


def _newest(omms: Iterable[Omm]) -> tuple[tuple[Omm, ...], tuple[Omm, ...]]:
    """Each object's newest element set, by catalog number, and the sets
    left out (see :func:`keepout.omm.deduplicate`)."""
    from keepout.omm.model import deduplicate

    kept, dropped = deduplicate(omms)
    return tuple(sorted(kept, key=lambda omm: omm.catalog_number)), dropped


@dataclass(frozen=True, slots=True)
class CatalogFileRef:
    """A file of a catalog, as it was when a screening read it.

    Parameters
    ----------
    name
        What the file is: a download's name (:attr:`CatalogPart.name`) or
        the file's name.
    path
        The file.
    sha256
        Its SHA-256 digest, in hexadecimal.
    url
        Where it was downloaded from; ``None`` for a local file.
    retrieved
        When the source sent it (UTC); ``None`` for a local file.
    """

    name: str
    path: Path
    sha256: str
    url: str | None = None
    retrieved: Time | None = None


@dataclass(frozen=True, slots=True)
class CatalogReference:
    """The files of a catalog, with their digests: a way to read exactly the
    same catalog again (:func:`load_reference`).

    Parameters
    ----------
    label
        What the catalog is, for people: its source, or its file.
    files
        Its files.
    """

    label: str
    files: tuple[CatalogFileRef, ...]

    @classmethod
    def of_file(cls, path: Path) -> CatalogReference:
        """The reference of a catalog read from one local file, labelled and
        named by its absolute path.

        Parameters
        ----------
        path
            The catalog file, as given to :func:`keepout.omm.load`; made absolute.

        Returns
        -------
        CatalogReference
            The reference of the file.

        Raises
        ------
        SourceUnavailable
            If the file cannot be read.
        """
        absolute = Path(path).absolute()
        return cls(
            str(absolute), (CatalogFileRef(absolute.name, absolute, _sha256(absolute)),)
        )

    def snapshot(self, directory: Path) -> CatalogReference:
        """The catalog with its files copied into ``directory``, each named
        by its digest (``<sha256><suffixes>``), so that later downloads
        replacing the originals do not change it.

        A file is written atomically, once its bytes match the digest; one
        already there with that digest is kept as it is.

        Parameters
        ----------
        directory
            Where the copies go; made if missing.

        Returns
        -------
        CatalogReference
            The same reference (label, names, URLs, retrieval times) to the
            copies.

        Raises
        ------
        SourceUnavailable
            If a file is gone or cannot be read.
        ChecksumMismatch
            If a file changed since it was referenced.
        OSError
            If a copy cannot be written.
        """
        from keepout.sources._store import has_digest, store_bytes

        files = []
        for file in self.files:
            target = directory / f"{file.sha256}{''.join(file.path.suffixes)}"
            if not has_digest(target, file.sha256):
                store_bytes(target, _verified(file))
            files.append(dataclasses.replace(file, path=target))
        return CatalogReference(self.label, tuple(files))


def load_reference(reference: CatalogReference) -> tuple[Omm, ...]:
    """The element sets of a referenced catalog, read again from its files:
    each object's newest set, by catalog number, as in :meth:`Catalog.of`.

    Every file is read once, and checked against its digest before any is
    parsed.

    Parameters
    ----------
    reference
        The catalog's files.

    Returns
    -------
    tuple of Omm
        One element set per object, by catalog number.

    Raises
    ------
    SourceUnavailable
        If a file is gone or cannot be read; the message names it.
    ChecksumMismatch
        If a file's digest is not the one referenced (it changed since);
        the message names it.
    OmmParseError
        If a file is not OMM data.
    """
    import keepout.omm

    contents = [(file.path, _verified(file)) for file in reference.files]
    kept, _ = _newest(
        omm for path, data in contents for omm in keepout.omm.load_data(data, path)
    )
    return kept


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
        catalog number.

        Parameters
        ----------
        parts
            The downloads of the catalog, e.g. one per CelesTrak group.

        Returns
        -------
        Catalog
            The catalog, one element set per object.
        """
        downloads = tuple(parts)
        kept, dropped = _newest(omm for part in downloads for omm in part.omms)
        return cls(kept, dropped, downloads)

    def reference(self, label: str) -> CatalogReference:
        """The reference of this catalog: its downloads' files, each with the
        digest its cache verified when the element sets were read from it.

        A file replaced since (e.g. by a cache refresh in another process)
        then no longer matches its reference, and copying or reading it
        again raises :class:`~keepout.errors.ChecksumMismatch`. A download
        whose digest is unknown (:attr:`Retrieval.sha256` is ``None``) is
        hashed as it is now on disk.

        Parameters
        ----------
        label
            What the catalog is, e.g. its source's name.

        Returns
        -------
        CatalogReference
            The files and their digests.

        Raises
        ------
        SourceUnavailable
            If a download's digest is unknown and its file cannot be read.
        """
        return CatalogReference(
            label,
            tuple(
                CatalogFileRef(
                    part.name,
                    part.retrieval.path,
                    part.retrieval.sha256 or _sha256(part.retrieval.path),
                    url=part.retrieval.url,
                    retrieved=part.retrieval.retrieved,
                )
                for part in self.parts
            ),
        )


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

        Parameters
        ----------
        cache
            The cache directory; ``None`` for ``$KEEPOUT_CACHE_DIR``, else the user
            cache directory.

        Returns
        -------
        Catalog
            One element set per object, and the downloads it came from.

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
            CELESTRAK_TTL,
        ),
        "spacetrack": CatalogSource(
            "spacetrack",
            "Space-Track full GP catalog",
            "Query",
            "the full catalog, needs SPACETRACK_IDENTITY and SPACETRACK_PASSWORD",
            "keepout.sources.spacetrack",
            GP_INTERVAL,
        ),
    }
)
"""The sources of the whole GP catalog, by name."""

DEFAULT_SOURCE: Final[CatalogSourceName] = "celestrak"
"""The catalog source used unless another is asked for: no account needed."""
