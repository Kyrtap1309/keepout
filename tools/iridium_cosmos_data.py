"""The Iridium-Cosmos case study's data: an excerpt of Space-Track's 2009
bulk TLE file.

Usage::

    uv run python tools/iridium_cosmos_data.py build ZIP [--check] [--data DIR]
        [--workers N]
    uv run python tools/iridium_cosmos_data.py verify [--data DIR]

Space-Track.org limits its ``gp_history`` class to one query per object for
an account's lifetime and sends queries "of many objects or large date
ranges" to yearly zip files of TLEs on its cloud storage (API documentation,
"API Use Guidelines"). ``ZIP`` is the file for 2009 (``tle2009.txt.zip``),
downloaded by hand once; it is never committed (about 315 MB).

``build`` streams ``ZIP`` (:func:`read_bulk`), keeping the element sets with
an epoch from :data:`WINDOW_START` (the decision point less
:data:`~keepout.screening.MAX_AGE`) to the collision. Of two sets of one
object and epoch, the one with the higher element set number, issued later,
is kept: the file holds re-issues of a set unchanged and corrections of it
(other elements, same epoch), and the manifest counts both. A set whose
international designator lacks its piece letters (``03037``), which keepout's
TLE reader refuses, gets a blank one (``OBJECT_ID`` ``UNKNOWN``) and its
checksum redone; the manifest lists those objects. Any other set keepout
cannot read is left out and listed in the manifest with the reason (in 2009's
file, four sets of one object with a malformed B*). It then screens
Iridium 33 against every object as known at the decision point, for 7 days, in a
:data:`~iridium_cosmos.EXCERPT_RADIUS` sphere (:func:`excerpt_sets`), and
writes to ``tests/case_study/data/iridium-cosmos/``:

- :data:`~iridium_cosmos.CATALOG_FILE`: every kept set of the pair, and the
  newest set known at the decision point of each object found, by catalog
  number and epoch, as gzipped CelesTrak CSV;
- :data:`NOTICE` and :data:`~snapshot_files.MANIFEST`: source, citation and
  licence; the bulk file's size and SHA-256, its counts, the selection and
  the digests of the files.

Screening is pairwise with the primary, so every approach of Iridium 33 the
full file gives in that window, the excerpt gives too. ``build --check``
fails instead of writing when a file is out of date; ``verify`` checks the
committed files against each other offline. Output is deterministic.
"""

from __future__ import annotations

import argparse
import datetime
import functools
import hashlib
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, TypedDict

import astropy.units as u
import iridium_cosmos as study
import snapshot_files
from generated import json_text, positive
from sgp4.io import compute_checksum
from snapshot_files import MANIFEST, MANIFEST_FORMAT, REDISTRIBUTION, FileDigest

from keepout.errors import OmmParseError
from keepout.omm import tle
from keepout.omm.celestrak import dumps_csv
from keepout.screening import DEFAULT_DURATION, MAX_AGE, ScreeningVolume, screen
from keepout.units import DAY, KM

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from astropy.time import Time

    from keepout.omm import Omm

__all__ = [
    "MANIFEST",
    "NOTICE",
    "TOOL",
    "WINDOW_START",
    "BulkFormatError",
    "BulkRead",
    "excerpt_sets",
    "main",
    "read_bulk",
]

TOOL: Final = "tools/iridium_cosmos_data.py"

NOTICE: Final = "NOTICE"
"""The committed file naming the data's source, citation and licence."""

WINDOW_START: Final = study.IRIDIUM_AS_OF - MAX_AGE * u.s
"""The earliest epoch kept: any older set is too old for screening at the
decision point."""

_DESIGNATOR: Final = slice(9, 17)
"""Line 1's international designator columns."""

_SHORT_DESIGNATOR: Final = re.compile(r"[0-9]{5} *")
"""A designator without its piece letters."""


class BulkFormatError(ValueError):
    """A bulk file that is not a sequence of two-line sets."""


class Unreadable(TypedDict):
    """A set left out because keepout cannot read it."""

    catalog_number: int
    line: str
    reason: str


class BulkCounts(TypedDict):
    """What the manifest records of the bulk file."""

    sets: int
    sets_in_window: int
    objects_in_window: int
    duplicates: int
    corrected: int
    designators_blanked: list[int]
    unreadable: list[Unreadable]


class Bulk(TypedDict):
    """The bulk file, as the manifest records it."""

    file: str
    bytes: int
    sha256: str
    member: str
    member_bytes: int
    counts: BulkCounts


class Selection(TypedDict):
    """How the excerpt was chosen, as the manifest records it."""

    window_start: str
    window_end: str
    screening_start: str
    screening_days: float
    radius_km: float
    screened_objects: int
    approaches: int
    objects: int
    sets: int


class Manifest(TypedDict):
    """The :data:`~snapshot_files.MANIFEST` file."""

    format: Literal[1]
    bulk: Bulk
    selection: Selection
    files: dict[str, FileDigest]


@dataclass(frozen=True, slots=True)
class BulkRead:
    """The element sets of a bulk file in a window.

    Parameters
    ----------
    sets
        One set per object and epoch, by catalog number and epoch.
    counts
        Sets in the file, in the window, objects, sets left out for a
        later one of the same object and epoch and how many of those had
        other elements, objects whose designator was blanked, sets left out
        as unreadable.
    """

    sets: tuple[Omm, ...]
    counts: BulkCounts


def _day_key(time: Time) -> tuple[int, float]:
    """``time`` as a TLE epoch sorts: year and day of year with fraction."""
    moment = time.utc.datetime
    start = datetime.datetime(moment.year, 1, 1)  # noqa: DTZ001 (naive UTC, as Time gives)
    return moment.year, 1 + (moment - start) / datetime.timedelta(days=1)


def _line_key(line: str) -> tuple[int, float]:
    """The epoch of a TLE's line 1, as :func:`_day_key` gives it."""
    year = int(line[18:20])
    return year + (1900 if year >= 57 else 2000), float(line[20:32])


def _pairs(lines: Iterable[bytes], member: str) -> Iterator[tuple[int, str, str]]:
    """The two lines of each set, with line 1's number in the member.

    Raises
    ------
    BulkFormatError
        If a line is not line 1 followed by line 2.
    """
    first: tuple[int, str] | None = None
    for number, raw in enumerate(lines, start=1):
        line = raw.decode("ascii").rstrip("\r\n").removesuffix("\\")
        if first is None:
            if not line.startswith("1 "):
                raise BulkFormatError(f"{member}:{number}: expected line 1 of a TLE")
            first = (number, line)
            continue
        if not line.startswith("2 "):
            raise BulkFormatError(f"{member}:{number}: expected line 2 of a TLE")
        yield first[0], first[1], line
        first = None
    if first is not None:
        raise BulkFormatError(f"{member}:{first[0]}: line 1 without line 2")


def _blank_short_designator(line: str) -> str | None:
    """Line 1 with a blank designator if its own lacks the piece letters;
    None if it does not."""
    if not _SHORT_DESIGNATOR.fullmatch(line[_DESIGNATOR]):
        return None
    blanked = line[: _DESIGNATOR.start] + " " * 8 + line[_DESIGNATOR.stop :]
    return blanked[:68] + str(compute_checksum(blanked))


def _parsed(one: str, two: str) -> Omm | str:
    """The set's OMM, or why keepout cannot read it."""
    try:
        (omm,) = tle.loads(f"{one}\n{two}\n")
    except OmmParseError as error:
        return str(error)
    return omm


def read_bulk(path: Path, start: Time, end: Time) -> BulkRead:
    """The sets of a bulk TLE zip with an epoch from ``start`` to ``end``,
    one per object and epoch (the module explains which).

    The zip's one member is streamed; only the window's sets are parsed.

    Raises
    ------
    BulkFormatError
        If the lines are not line 1 and line 2 in turn (the member and line
        number named).
    ValueError
        If the zip does not hold exactly one member.
    """
    low, high = _day_key(start), _day_key(end)
    kept: dict[tuple[int, str], tuple[str, str, Omm]] = {}
    total = in_window = duplicates = corrected = 0
    short: set[int] = set()
    unreadable: list[Unreadable] = []
    with zipfile.ZipFile(path) as archive:
        (member,) = archive.namelist()
        with archive.open(member) as raw:
            for number, line_one, two in _pairs(raw, member):
                total += 1
                one = _blank_short_designator(line_one) or line_one
                if not low <= _line_key(one) <= high:
                    continue
                in_window += 1
                if one is not line_one:
                    short.add(int(one[2:7]))
                omm = _parsed(one, two)
                if isinstance(omm, str):
                    unreadable.append(
                        {
                            "catalog_number": int(one[2:7]),
                            "line": f"{member}:{number}",
                            "reason": omm,
                        }
                    )
                    continue
                key = (omm.catalog_number, one[18:32])
                if key not in kept:
                    kept[key] = (one, two, omm)
                    continue
                duplicates += 1
                old_one, old_two, old = kept[key]
                corrected += (old_one[:64], old_two[:68]) != (one[:64], two[:68])
                if omm.element_set_no > old.element_set_no:
                    kept[key] = (one, two, omm)
    sets = tuple(
        omm
        for _, _, omm in sorted(
            kept.values(),
            key=lambda kept_set: (kept_set[2].catalog_number, kept_set[2].epoch.jd),
        )
    )
    return BulkRead(
        sets=sets,
        counts={
            "sets": total,
            "sets_in_window": in_window,
            "objects_in_window": len({omm.catalog_number for omm in sets}),
            "duplicates": duplicates,
            "corrected": corrected,
            "designators_blanked": sorted(short),
            "unreadable": unreadable,
        },
    )


def excerpt_sets(
    sets: Sequence[Omm], *, workers: int | None = None
) -> tuple[tuple[Omm, ...], Selection]:
    """The excerpt and how it was chosen: every set of the pair, and of each
    object passing within :data:`~iridium_cosmos.EXCERPT_RADIUS` of Iridium
    33 in the 7 days from the decision point, as known then, its newest set
    known then: the one screening at the decision point takes.
    """
    known = study.known_at(sets, study.IRIDIUM_AS_OF)
    primary = [omm for omm in known if omm.catalog_number == study.IRIDIUM_33]
    volume = ScreeningVolume.sphere(
        study.EXCERPT_RADIUS, label=f"{study.EXCERPT_RADIUS / KM:g} km sphere"
    )
    result = screen(
        primary, known, study.IRIDIUM_AS_OF, DEFAULT_DURATION, volume, workers=workers
    )
    found = {approach.secondary_number for approach in result.approaches}
    newest: dict[int, Omm] = {}
    for omm in known:
        if omm.catalog_number in found - set(study.PAIR):
            best = newest.get(omm.catalog_number)
            if best is None or omm.epoch > best.epoch:
                newest[omm.catalog_number] = omm
    pair = (omm for omm in sets if omm.catalog_number in study.PAIR)
    excerpt = tuple(
        sorted(
            (*pair, *newest.values()),
            key=lambda omm: (omm.catalog_number, omm.epoch.jd),
        )
    )
    return excerpt, {
        "window_start": WINDOW_START.isot,
        "window_end": study.TCA_REFERENCE.isot,
        "screening_start": study.IRIDIUM_AS_OF.isot,
        "screening_days": DEFAULT_DURATION / DAY,
        "radius_km": study.EXCERPT_RADIUS / KM,
        "screened_objects": len({omm.catalog_number for omm in known}),
        "approaches": len(result.approaches),
        "objects": len({omm.catalog_number for omm in excerpt}),
        "sets": len(excerpt),
    }


@dataclass(frozen=True, slots=True)
class Inputs:
    """What the committed files are made from: the bulk file's record, the
    selection and the excerpt's sets."""

    bulk: Bulk
    selection: Selection
    omms: tuple[Omm, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def inputs_from_bulk(path: Path, *, workers: int | None = None) -> Inputs:
    """The inputs, from the bulk zip at ``path``.

    Raises
    ------
    BulkFormatError, ValueError
        As :func:`read_bulk`; ValueError also if the excerpt would not read
        back from CSV unchanged.
    """
    read = read_bulk(path, WINDOW_START, study.TCA_REFERENCE)
    omms, selection = excerpt_sets(read.sets, workers=workers)
    snapshot_files.check_round_trip(omms)
    with zipfile.ZipFile(path) as archive:
        (info,) = archive.infolist()
    bulk: Bulk = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "member": info.filename,
        "member_bytes": info.file_size,
        "counts": read.counts,
    }
    return Inputs(bulk=bulk, selection=selection, omms=omms)


def read_manifest(data: Path) -> Manifest:
    """The :data:`~snapshot_files.MANIFEST` in ``data``.

    Raises
    ------
    ValueError
        If it is not one this tool writes.
    """
    return snapshot_files.read_manifest(data, Manifest)


def inputs_from_data(data: Path) -> Inputs:
    """The inputs, from the committed excerpt and its manifest."""
    manifest = read_manifest(data)
    omms = study.read_catalog(data, named=False)
    return Inputs(bulk=manifest["bulk"], selection=manifest["selection"], omms=omms)


def render_notice(inputs: Inputs) -> str:
    """The NOTICE: source, citation, what the excerpt holds, licence."""
    selection = inputs.selection
    return (
        "Data: USSPACECOM/18 SDS via Space-Track.org (www.space-track.org),\n"
        f"from its yearly bulk TLE file {inputs.bulk['file']} (2009).\n"
        "\n"
        f"Excerpt made by {TOOL} (see {MANIFEST}):\n"
        "\n"
        f"- {study.CATALOG_FILE}: the element sets of Iridium 33 and\n"
        f"  Cosmos 2251 with an epoch from {selection['window_start']} to\n"
        f"  {selection['window_end']} UTC, and the newest set known at\n"
        f"  {selection['screening_start']} UTC of every object passing within\n"
        f"  {selection['radius_km']:g} km of Iridium 33 in the "
        f"{selection['screening_days']:g} days from then,\n"
        "  as gzipped CelesTrak CSV (the OMM keywords; the TLEs carry no names).\n"
        "\n"
        f"{MANIFEST} lists the bulk file's size and SHA-256, the counts and\n"
        "the SHA-256 digests of these files.\n"
        "\n"
        f"{REDISTRIBUTION}"
    )


def contents(inputs: Inputs) -> dict[str, bytes]:
    """Every committed file but the manifest, uncompressed, by name."""
    return {
        study.CATALOG_FILE: dumps_csv(inputs.omms).encode(),
        NOTICE: render_notice(inputs).encode(),
    }


def render_manifest(inputs: Inputs, files: Mapping[str, bytes]) -> str:
    """The manifest of the stored bytes ``files`` of the other files."""
    manifest: Manifest = {
        "format": MANIFEST_FORMAT,
        "bulk": inputs.bulk,
        "selection": inputs.selection,
        "files": snapshot_files.file_digests(files),
    }
    return json_text(manifest)


def stored_files(data: Path, inputs: Inputs) -> dict[str, bytes]:
    """The bytes every committed file should hold, the manifest last."""
    manifest = functools.partial(render_manifest, inputs)
    return snapshot_files.stored_files(
        data, contents(inputs), {study.CATALOG_FILE}, manifest
    )


def main(argv: Sequence[str] | None = None) -> int:
    """The entry point: ``build`` or ``verify``.

    Returns
    -------
    int
        1 if ``build --check`` or ``verify`` found a file out of date, else 0.
    """
    parser = argparse.ArgumentParser(prog=TOOL)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="write the excerpt from the bulk ZIP")
    build.add_argument("raw", type=Path, metavar="ZIP")
    build.add_argument("--check", action="store_true", help="fail if out of date")
    build.add_argument("--data", type=Path, default=study.DATA)
    build.add_argument("--workers", type=positive, default=None)
    verify = commands.add_parser("verify", help="check the committed files (offline)")
    verify.add_argument("--data", type=Path, default=study.DATA)
    args = parser.parse_args(argv)
    building = args.command == "build"
    return snapshot_files.build_or_verify(
        raw=args.raw if building else None,
        data=args.data,
        check=building and args.check,
        from_raw=functools.partial(inputs_from_bulk, workers=args.workers)
        if building
        else inputs_from_data,
        from_data=inputs_from_data,
        files=stored_files,
        tool=TOOL,
    )


if __name__ == "__main__":
    sys.exit(main())
