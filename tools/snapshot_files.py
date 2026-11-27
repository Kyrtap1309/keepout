"""Shared by the snapshot tools: the raw index of saved answers, checked
against their digests; the manifest and the counts it records of a catalog;
the ``build`` and ``verify`` commands; and committed files that stay byte
for byte the same while their data does."""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Final,
    Generic,
    Protocol,
    TypedDict,
    TypeVar,
    get_type_hints,
)

from astropy.time import Time
from generated import read_json, write_or_check

from keepout.ccsds.files import GzipError, gunzip
from keepout.omm.celestrak import dumps_csv, loads_csv
from keepout.sources._store import store_json

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Collection, Mapping, Sequence

    from _typeshed import DataclassInstance

    from keepout._types import JsonValue
    from keepout.omm import Omm

RAW_INDEX: Final = "raw.json"
"""The file in a tool's ``RAW`` directory listing each saved answer: its
URL, file, digest and time."""

RAW_FORMAT: Final = 1
"""Version of the :data:`RAW_INDEX` layout."""

MANIFEST: Final = "manifest.json"
"""The committed file recording the downloads, counts and the SHA-256 digest
of every other committed file."""

MANIFEST_FORMAT: Final = 1
"""Version of the :data:`MANIFEST` layout."""

SIX_DIGITS: Final = 100_000
"""The first six-digit catalog number."""

REDISTRIBUTION: Final = (
    "This is basic space situational awareness data, which the\n"
    "Space-Track.org user agreement allows to redistribute with the\n"
    "citation above. It is not covered by keepout's MIT licence.\n"
)
"""The last paragraph of a snapshot's NOTICE: why the data may be
redistributed, and that keepout's licence does not cover it."""

_Entry = TypeVar("_Entry", bound="DataclassInstance")
_Inputs = TypeVar("_Inputs")
_T = TypeVar("_T")


class Digested(Protocol):
    """A saved file listed in a raw index with its SHA-256 digest."""

    @property
    def file(self) -> str:
        """Its file name in the raw directory."""
        ...

    @property
    def sha256(self) -> str:
        """SHA-256 hex digest of its bytes."""
        ...


class FileDigest(TypedDict):
    """Length and SHA-256 hex digest of a committed file as stored."""

    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class RawIndex(Generic[_Entry]):
    """How a tool lists its saved answers in :data:`RAW_INDEX`.

    The file is a JSON object: ``format`` (:data:`RAW_FORMAT`) and, under
    :attr:`key`, a list of the answers, each an object of the fields of
    :attr:`entry`.

    Parameters
    ----------
    key
        The key of the list.
    entry
        A frozen dataclass of one answer, its fields strings, integers or
        booleans.
    """

    key: str
    entry: type[_Entry]

    def read(self, raw: Path) -> tuple[_Entry, ...]:
        """The answers listed in ``raw`` (none if it has no index).

        Raises
        ------
        ValueError
            If the index is not of :data:`RAW_FORMAT`, or an answer lacks a
            field of :attr:`entry`, has another, or one of another type.
        """
        path = raw / RAW_INDEX
        if not path.exists():
            return ()
        record: JsonValue = json.loads(path.read_text(encoding="utf-8"))
        listed = record.get(self.key) if isinstance(record, dict) else None
        if not isinstance(record, dict) or record.get("format") != RAW_FORMAT:
            raise ValueError(f"{path} is not a raw index of format {RAW_FORMAT}")
        if not isinstance(listed, list):
            raise ValueError(f"{path} lists no {self.key}")
        return tuple(self._entry(item, path) for item in listed)

    def _entry(self, item: JsonValue, path: Path) -> _Entry:
        """One answer of the index at ``path``.

        Raises
        ------
        ValueError
            If ``item`` is not an object of the fields of :attr:`entry`.
        """
        fields = get_type_hints(self.entry)
        if (
            not isinstance(item, dict)
            or set(item) != set(fields)
            or any(type(item[name]) is not kind for name, kind in fields.items())
        ):
            raise ValueError(f"{path} lists {item!r}, not a {self.entry.__name__}")
        return self.entry(**item)

    def write(self, raw: Path, entries: Sequence[_Entry]) -> None:
        """The index of ``entries``, written in ``raw`` atomically."""
        listed = [dataclasses.asdict(entry) for entry in entries]
        store_json(raw / RAW_INDEX, {"format": RAW_FORMAT, self.key: listed})


def read_body(raw: Path, saved: Digested) -> bytes:
    """A saved file of ``raw``, checked against its digest in the index.

    Raises
    ------
    ValueError
        If the file's digest is not the one listed.
    """
    data = (raw / saved.file).read_bytes()
    if hashlib.sha256(data).hexdigest() != saved.sha256:
        raise ValueError(f"{raw / saved.file} does not match its digest in {RAW_INDEX}")
    return data


def read_manifest(data: Path, shape: type[_T]) -> _T:
    """The :data:`MANIFEST` in ``data``, checked to be a ``shape``.

    Raises
    ------
    ValueError
        If it is not (:func:`generated.read_json`).
    """
    return read_json(data / MANIFEST, shape)


def check_round_trip(omms: Sequence[Omm]) -> None:
    """A check that CelesTrak CSV holds the element sets ``omms`` unchanged.

    Raises
    ------
    ValueError
        If the catalog would not read back from CSV as the same element sets.
    """
    if loads_csv(dumps_csv(omms)) != tuple(omms):
        raise ValueError("the catalog does not read back unchanged from CSV")


class CatalogCounts(TypedDict):
    """What a manifest records of every catalog."""

    objects: int
    six_digit: int
    regimes: dict[str, int]
    oldest_epoch: str
    newest_epoch: str


def catalog_counts(
    omms: Sequence[Omm], regime: Callable[[Omm], str], regimes: Sequence[str]
) -> CatalogCounts:
    """The objects of a catalog, how many have six-digit numbers or are of
    each of ``regimes`` (by ``regime``), and its oldest and newest epoch."""
    counts = Counter(regime(omm) for omm in omms)
    epochs = Time([omm.epoch for omm in omms])
    return {
        "objects": len(omms),
        "six_digit": sum(omm.catalog_number >= SIX_DIGITS for omm in omms),
        "regimes": {name: counts[name] for name in regimes},
        "oldest_epoch": omms[int(epochs.argmin())].epoch.isot,
        "newest_epoch": omms[int(epochs.argmax())].epoch.isot,
    }


def file_digests(files: Mapping[str, bytes]) -> dict[str, FileDigest]:
    """Each file's length and digest, by name, as a manifest lists them."""
    return {
        name: {"bytes": len(stored), "sha256": hashlib.sha256(stored).hexdigest()}
        for name, stored in files.items()
    }


def _gunzipped(stored: bytes) -> bytes | None:
    """``stored`` decompressed; None if it is not intact gzip data."""
    try:
        return gunzip(stored)
    except GzipError:
        return None


def stored_files(
    data: Path,
    contents: Mapping[str, bytes],
    gzipped: Collection[str],
    manifest: Callable[[Mapping[str, bytes]], str],
) -> dict[str, bytes]:
    """The bytes each committed file should hold, by name, the manifest last.

    The files named in ``gzipped`` are stored compressed; one already in
    ``data`` with the right contents is kept as it is, so a different zlib
    build does not change a file whose data did not. The :data:`MANIFEST`
    is ``manifest`` of the other files' stored bytes.
    """
    files: dict[str, bytes] = {}
    for name, content in contents.items():
        path = data / name
        stored = path.read_bytes() if name in gzipped and path.exists() else None
        if name not in gzipped:
            files[name] = content
        elif stored is not None and _gunzipped(stored) == content:
            files[name] = stored
        else:
            files[name] = gzip.compress(content, compresslevel=9, mtime=0)
    files[MANIFEST] = manifest(files).encode()
    return files


def add_data_commands(
    add_parser: Callable[..., argparse.ArgumentParser], data: Path, saved: str
) -> argparse.ArgumentParser:
    """A snapshot tool's ``build`` and ``verify`` commands, added to its parser;
    ``build``'s parser, for options of the tool's own.

    Parameters
    ----------
    add_parser
        The ``add_parser`` of the tool's subcommands.
    data
        The default of ``--data``: where the committed data is.
    saved
        What ``RAW`` holds, in ``build``'s help: ``bodies``, ``answers``.

    Returns
    -------
    argparse.ArgumentParser
        The ``build`` command's parser.
    """
    build = add_parser("build", help=f"write the data from the {saved} in RAW")
    build.add_argument("raw", type=Path)
    build.add_argument("--check", action="store_true", help="fail if out of date")
    build.add_argument("--data", type=Path, default=data)
    verify = add_parser(
        "verify", help="check the committed files against each other (offline)"
    )
    verify.add_argument("--data", type=Path, default=data)
    return build


def build_or_verify(
    *,
    raw: Path | None,
    data: Path,
    check: bool,
    from_raw: Callable[[Path], _Inputs],
    from_data: Callable[[Path], _Inputs],
    files: Callable[[Path, _Inputs], Mapping[str, bytes]],
    tool: str,
) -> int:
    """The ``build`` command (or with ``check`` its verification), or
    ``verify`` when there is no ``raw``.

    ``build`` makes the inputs of the files ``from_raw`` (the ``RAW``
    directory), ``verify`` ``from_data`` (the committed data in ``data``),
    and ``files`` gives the bytes each file in ``data`` should hold.

    Returns
    -------
    int
        1 if ``build --check`` or ``verify`` found a file out of date, else 0.
    """
    inputs = from_data(data) if raw is None else from_raw(raw)
    stored = files(data, inputs)
    return write_or_check(
        {data / name: content for name, content in stored.items()},
        check=check or raw is None,
        tool=f"{tool} build",
    )
