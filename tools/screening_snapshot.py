"""The frozen screening data: the public catalog and SOCRATES, downloaded.

Usage::

    uv run python tools/screening_snapshot.py download RAW
    uv run python tools/screening_snapshot.py build RAW [--check] [--data DIR]
    uv run python tools/screening_snapshot.py verify [--data DIR]

``download`` fetches CelesTrak's public GP groups
(:data:`~keepout.sources.celestrak.PUBLIC_GROUPS`) and then the latest
SOCRATES Plus report, one request per URL, through keepout's CelesTrak
client with a new, empty cache (the user's cache is not touched). Each
response body is saved unchanged in the directory ``RAW`` before the client
reads it. ``build`` writes, from those bodies alone (no network), the
committed data in ``tests/screening/data/``:

- :data:`SNAPSHOT`: the public catalog, one element set per object, as
  gzipped CelesTrak CSV;
- :data:`SOCRATES`: the SOCRATES approaches keepout can compare exactly, as
  gzipped CSV lines unchanged from the report (see below);
- :data:`FIXTURE`: about 200 objects of the catalog for tests
  (:func:`select_fixture`);
- :data:`NOTICE` and :data:`~snapshot_files.MANIFEST`: source and licence;
  downloads, counts and SHA-256 digests.

``build --check`` fails instead of writing when a file is out of date.
``verify`` checks the committed files against each other offline: it
rebuilds every file it can from the catalog, the excerpt and the manifest.
Output is deterministic (gzip without a time stamp; a gzipped file whose
contents are unchanged is kept as it is, whatever zlib would write now).

The SOCRATES report is about 20 MB (some 190 000 approaches), too large to
commit. SOCRATES ran on the element sets of its own time (TCA minus ``DSE``
gives each object's epoch, to the 43 s of DSE's three decimals), and
CelesTrak serves only the latest element sets, so an approach can be
compared exactly only if the snapshot holds the same element sets of both
objects (:func:`~keepout.sources.socrates.epoch_matches`). Those approaches
are the excerpt; the manifest counts the others. The report window starts at
0 h UTC and uses element sets of the evening before, while most objects get
new element sets several times a day: download soon after SOCRATES's data
cut-off to keep more approaches comparable.

CelesTrak's usage policy (https://celestrak.org/usage-policy.php) applies to
``download``: GP data changes about every two hours and each URL may be
downloaded once per change; a repeat download of unchanged data is answered
with HTTP 403, and a client that keeps asking after any error is blocked by
CelesTrak's firewall. The tool asks for each URL once and stops at the first
answer other than 200 OK (nothing is retried). ``download`` resumes into an
existing ``RAW``: URLs already saved there are not asked for again, so
after an error wait for CelesTrak's next update (two hours or more) and run
it again. A maintainer refreshing the data runs ``download`` into a new
directory, then ``build``, and commits the result.

The data is USSPACECOM (18th Space Defense Squadron) data from
Space-Track.org, republished by CelesTrak; Space-Track's user agreement
allows redistribution of such basic SSA data with citation, which the NOTICE
gives. It is not covered by keepout's MIT licence.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import gzip
import math
import sys
import tempfile
import textwrap
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias, TypedDict, TypeVar, cast, get_args
from urllib.parse import parse_qs, urlsplit

import astropy.units as u
import snapshot_files
from astropy.time import Time
from generated import REPO, day_text, json_text, second_text
from snapshot_files import (
    MANIFEST,
    MANIFEST_FORMAT,
    REDISTRIBUTION,
    SIX_DIGITS,
    CatalogCounts,
    FileDigest,
    RawIndex,
    catalog_counts,
    file_digests,
    read_body,
)

from keepout.equinoctial import semi_major_axis
from keepout.errors import SourceUnavailable
from keepout.omm import Omm, load
from keepout.omm.celestrak import dumps_csv, loads_csv
from keepout.screening import CloseApproach, ScreeningVolume, screen
from keepout.screening.volume import SFS_DEEP_SPACE_PERIOD, SFS_ECCENTRICITY
from keepout.sources.cache import http_download, store_bytes
from keepout.sources.catalog import Catalog
from keepout.sources.celestrak import (
    GP_URL,
    PUBLIC_GROUPS,
    SOCRATES_URL,
    Downloader,
    SocratesReport,
    fetch_catalog,
    fetch_socrates,
)
from keepout.sources.socrates import SocratesApproach, epoch_matches, read_socrates
from keepout.synthetic.covariance_model import LEO_HIGHEST_APOGEE
from keepout.units import DAY, seconds_between

TOOL: Final = "tools/screening_snapshot.py"
DATA: Final = REPO / "tests/screening/data"
"""Where ``build`` writes the committed data."""

SNAPSHOT: Final = "celestrak-public.csv.gz"
"""The catalog: CelesTrak CSV of every object of the public groups, one
element set per object (the newest), by catalog number, gzipped."""

SOCRATES: Final = "socrates-excerpt.csv.gz"
"""The header and the approaches of the SOCRATES report whose element sets
are in the catalog (:func:`~keepout.sources.socrates.epoch_matches`), lines
unchanged, gzipped."""

FIXTURE: Final = "celestrak-fixture.csv"
"""About 200 objects of the catalog (:func:`select_fixture`), CelesTrak CSV."""

NOTICE: Final = "NOTICE"
"""Source, retrieval time and licence of the data."""

GZIPPED: Final = frozenset({SNAPSHOT, SOCRATES})

EARTH_RADIUS: Final = 6_378_137.0
"""Equatorial radius of the Earth [m] (WGS 84), for altitudes."""

CLOSEST: Final = 10
"""Closest pairs of comparable SOCRATES approaches taken into the fixture."""

CLOSEST_FAST: Final = 20
"""Closest pairs taken of the approaches at :data:`FAST` or more."""

FAST: Final = 1_000.0
"""Least relative speed of a fast approach [m/s]: not formation flying."""

CLOSEST_PER_REGIME: Final = 5
"""Closest pairs taken per regime of the first object."""

PER_REGIME: Final[Mapping[str, int]] = {"LEO": 60, "MEO": 30, "GEO": 40, "HEO": 30}
"""Objects of each regime taken into the fixture, evenly spaced."""

EXTREMES: Final = 5
"""Six-digit numbers, highest eccentricities, lowest perigees and oldest
epochs taken into the fixture (each)."""

_T = TypeVar("_T")


class BodyRecord(TypedDict):
    """A :class:`Body` as JSON, in the raw index and the manifest."""

    url: str
    file: str
    sha256: str
    size: int
    retrieved: str


@dataclass(frozen=True, slots=True)
class Body:
    """One response body saved in the ``RAW`` directory.

    Parameters
    ----------
    url
        The URL it answered.
    file
        Its file name in ``RAW``.
    sha256
        SHA-256 hex digest of the body.
    size
        Its length in bytes.
    retrieved
        When it was received, UTC, ISO 8601 with milliseconds and a ``Z``.
    """

    url: str
    file: str
    sha256: str
    size: int
    retrieved: str

    def record(self) -> BodyRecord:
        """The body as JSON."""
        return cast("BodyRecord", dataclasses.asdict(self))


INDEX: Final = RawIndex("bodies", Body)
"""The bodies saved in ``RAW``, in :data:`~snapshot_files.RAW_INDEX`."""


def raw_name(url: str) -> str:
    """The file name of the body of ``url`` in ``RAW``.

    Raises
    ------
    ValueError
        If ``url`` is neither a GP group nor the SOCRATES report.
    """
    if url == SOCRATES_URL:
        return "socrates-sort-minRange.csv"
    groups = parse_qs(urlsplit(url).query).get("GROUP", [])
    if url.startswith(f"{GP_URL}?") and len(groups) == 1:
        return f"gp-{groups[0]}.csv"
    raise ValueError(f"not a CelesTrak URL this tool downloads: {url}")


def utc_text(unix: float) -> str:
    """A Unix time as UTC ISO 8601 text with milliseconds and a ``Z``."""
    return f"{Time(unix, format='unix', scale='utc').isot}Z"


class RawBodies:
    """A downloader that keeps every body in a ``RAW`` directory.

    A URL already saved there is answered from the saved body, never asked
    for again. Otherwise, if ``download`` is given, the URL is downloaded
    once and the body is saved (and listed in :data:`INDEX`) before it
    is returned; with no ``download`` it is an error.

    Parameters
    ----------
    raw
        The directory; created when the first body is saved.
    download
        Downloads a URL, or ``None`` to work offline.
    clock
        The current Unix time.
    """

    def __init__(
        self,
        raw: Path,
        download: Downloader | None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.raw = raw
        self.download = download
        self.clock = clock
        self.bodies: dict[str, Body] = {body.url: body for body in INDEX.read(raw)}

    def __call__(self, url: str) -> bytes:
        """The body of ``url``: the saved one, or downloaded and saved.

        Raises
        ------
        SourceUnavailable
            If ``url`` is not saved and there is no downloader.
        """
        saved = self.bodies.get(url)
        if saved is not None:
            return read_body(self.raw, saved)
        if self.download is None:
            raise SourceUnavailable(f"{url} is not saved in {self.raw}")
        name = raw_name(url)
        data = self.download(url)
        retrieved = utc_text(self.clock())
        digest = store_bytes(self.raw / name, data)
        self.bodies[url] = Body(url, name, digest, len(data), retrieved)
        INDEX.write(self.raw, tuple(self.bodies.values()))
        return data


def fetch(raw: Path, download: Downloader | None) -> tuple[Catalog, SocratesReport]:
    """The public catalog and the SOCRATES report, read through ``RAW``.

    Read with keepout's CelesTrak client into a new, empty cache, so every
    body is checked as the client checks a download. The groups come first,
    then SOCRATES.

    Parameters
    ----------
    raw
        The ``RAW`` directory.
    download
        Downloads a URL not yet saved in ``raw``; ``None`` to only read
        saved bodies.
    """
    bodies = RawBodies(raw, download)
    with tempfile.TemporaryDirectory() as cache:
        catalog = fetch_catalog(PUBLIC_GROUPS, cache=Path(cache), download=bodies)
        report = fetch_socrates(cache=Path(cache), download=bodies)
    return catalog, report


def socrates_header(text: str) -> str:
    """The header line of a SOCRATES report, line ending and any byte-order
    mark included."""
    return text.splitlines(keepends=True)[0] if text else ""


def comparable(
    approaches: Sequence[SocratesApproach], omms: Sequence[Omm]
) -> list[SocratesApproach]:
    """The approaches whose two objects are in ``omms`` with the element sets
    SOCRATES used (:func:`~keepout.sources.socrates.epoch_matches`), in
    report order."""
    by_number = {omm.catalog_number: omm for omm in omms}
    return [
        approach
        for approach in approaches
        if approach.primary in by_number
        and approach.secondary in by_number
        and epoch_matches(
            approach, by_number[approach.primary], by_number[approach.secondary]
        )
    ]


def nearest_approach(
    first: Omm, second: Omm, tca: Time, *, window: float, radius: float
) -> CloseApproach | None:
    """The approach of two objects keepout finds nearest ``tca``, if any.

    The two objects are screened with each other from ``window`` [s] before
    to ``window`` after ``tca``, in a sphere of ``radius`` [m] around the
    first.
    """
    result = screen(
        [first],
        [second],
        tca - window * u.s,
        2.0 * window,
        ScreeningVolume.sphere(radius),
    )
    return min(
        result.approaches,
        key=lambda approach: abs(seconds_between(tca, approach.tca)),
        default=None,
    )


Regime: TypeAlias = Literal["LEO", "MEO", "GEO", "HEO"]
"""Orbit regimes the fixture spreads over (:func:`regime`)."""

REGIMES: Final[tuple[Regime, ...]] = get_args(Regime)


def perigee_altitude(omm: Omm) -> float:
    """Perigee altitude above the equatorial radius [m], from mean elements."""
    return semi_major_axis(omm.mean_motion) * (1.0 - omm.eccentricity) - EARTH_RADIUS


def regime(omm: Omm) -> Regime:
    """The orbit regime of an element set, from its mean elements.

    HEO: eccentricity 0.25 or more; LEO: apogee altitude up to 2000 km;
    GEO: period 1300 to 1800 min; MEO: the rest.
    """
    if omm.eccentricity >= SFS_ECCENTRICITY:
        return "HEO"
    axis = semi_major_axis(omm.mean_motion)
    if axis * (1.0 + omm.eccentricity) - EARTH_RADIUS <= LEO_HIGHEST_APOGEE:
        return "LEO"
    lowest, highest = SFS_DEEP_SPACE_PERIOD
    if lowest <= 2.0 * math.pi / omm.mean_motion <= highest:
        return "GEO"
    return "MEO"


def spread(items: Sequence[_T], count: int) -> list[_T]:
    """``count`` items evenly spaced through ``items`` (all if fewer)."""
    if len(items) <= count:
        return list(items)
    return [items[(index * len(items)) // count] for index in range(count)]


def select_fixture(
    omms: Sequence[Omm], approaches: Sequence[SocratesApproach]
) -> list[Omm]:
    """The test fixture: about 200 objects of the catalog, by a fixed rule.

    - The objects of the :data:`CLOSEST` closest pairs of comparable SOCRATES
      approaches (:func:`comparable`; a pair counts once, however many
      approaches it has), the :data:`CLOSEST_FAST` closest pairs at
      :data:`FAST` or more, and the :data:`CLOSEST_PER_REGIME` closest of
      each regime of the first object.
    - :data:`PER_REGIME` objects of each regime, evenly spaced in catalog
      order.
    - :data:`EXTREMES` objects with six-digit catalog numbers (evenly spaced),
      and the :data:`EXTREMES` with the highest eccentricity, the lowest
      perigee and the oldest epoch.

    Ties break by catalog number (approaches: by TCA, then numbers), so the
    rule gives the same objects for the same data in any order.

    Parameters
    ----------
    omms
        The catalog, one element set per object.
    approaches
        The SOCRATES report, or any part of it holding its comparable
        approaches.

    Returns
    -------
    list[Omm]
        The chosen element sets, by catalog number.
    """
    ordered = sorted(omms, key=lambda omm: omm.catalog_number)
    by_number = {omm.catalog_number: omm for omm in ordered}
    regimes = {omm.catalog_number: regime(omm) for omm in ordered}
    closest = sorted(
        comparable(approaches, ordered),
        key=lambda a: (a.miss_distance, a.tca.jd1, a.tca.jd2, a.primary, a.secondary),
    )
    pairs = _pairs(closest, CLOSEST)
    pairs += _pairs([a for a in closest if a.relative_speed >= FAST], CLOSEST_FAST)
    for name in REGIMES:
        of_regime = [a for a in closest if regimes[a.primary] == name]
        pairs += _pairs(of_regime, CLOSEST_PER_REGIME)
    chosen = {number for pair in pairs for number in pair}
    for name in REGIMES:
        members = [omm for omm in ordered if regimes[omm.catalog_number] == name]
        chosen.update(omm.catalog_number for omm in spread(members, PER_REGIME[name]))
    extremes = (
        spread([omm for omm in ordered if omm.catalog_number >= SIX_DIGITS], EXTREMES),
        sorted(ordered, key=lambda omm: -omm.eccentricity)[:EXTREMES],
        sorted(ordered, key=perigee_altitude)[:EXTREMES],
        sorted(ordered, key=lambda omm: (omm.epoch.jd1, omm.epoch.jd2))[:EXTREMES],
    )
    chosen.update(omm.catalog_number for group in extremes for omm in group)
    return [by_number[number] for number in sorted(chosen)]


def _pairs(approaches: Sequence[SocratesApproach], count: int) -> list[tuple[int, int]]:
    """The first ``count`` distinct pairs of objects of ``approaches``."""
    pairs = dict.fromkeys((a.primary, a.secondary) for a in approaches)
    return list(pairs)[:count]


class SocratesSummary(TypedDict):
    """What the manifest records of the whole SOCRATES report."""

    approaches: int
    """Rows of the report."""
    both_in_catalog: int
    """Rows whose two objects are in the catalog."""
    comparable: int
    """Rows whose two objects are in the catalog with the element sets
    SOCRATES used: the rows of the excerpt."""
    first_tca: str
    last_tca: str
    newest_epoch: str
    """The newest element set epoch SOCRATES used (TCA minus DSE, to the
    minute): the data cut-off of its run."""


class Retrieved(TypedDict):
    """When the bodies were received: the first and the last, UTC ISO 8601."""

    first: str
    last: str


class CatalogSummary(CatalogCounts):
    """What the manifest records of the catalog."""

    group_rows: dict[str, int]
    duplicates: int


class FixtureSummary(TypedDict):
    """What the manifest records of the fixture."""

    objects: int
    regimes: dict[Regime, int]


class Manifest(TypedDict):
    """The :data:`~snapshot_files.MANIFEST` file."""

    format: Literal[1]
    retrieved: Retrieved
    downloads: list[BodyRecord]
    catalog: CatalogSummary
    socrates: SocratesSummary
    fixture: FixtureSummary
    files: dict[str, FileDigest]


def read_manifest(data: Path) -> Manifest:
    """The :data:`~snapshot_files.MANIFEST` in ``data``.

    Raises
    ------
    ValueError
        If it is not one this tool writes.
    """
    return snapshot_files.read_manifest(data, Manifest)


def read_catalog(data: Path) -> tuple[Omm, ...]:
    """The catalog in ``data`` (:data:`SNAPSHOT`)."""
    return load(data / SNAPSHOT)


def read_excerpt(data: Path) -> str:
    """The SOCRATES excerpt in ``data`` (:data:`SOCRATES`), as text."""
    return contents_of(SOCRATES, (data / SOCRATES).read_bytes()).decode("utf-8")


def retrieval_day(manifest: Manifest) -> str:
    """The day the data was retrieved (UTC, ISO 8601): of the first body."""
    return day_text(manifest["retrieved"]["first"])


def summarise_socrates(
    approaches: Sequence[SocratesApproach], omms: Sequence[Omm]
) -> SocratesSummary:
    """Counts and times of a whole SOCRATES report against the catalog."""
    numbers = {omm.catalog_number for omm in omms}
    by_tca = sorted(approaches, key=lambda a: (a.tca.jd1, a.tca.jd2))
    newest = max(
        a.tca.jd1 + (a.tca.jd2 - since / DAY)
        for a in approaches
        for since in (a.since_epoch_primary, a.since_epoch_secondary)
    )
    return SocratesSummary(
        approaches=len(approaches),
        both_in_catalog=sum(
            a.primary in numbers and a.secondary in numbers for a in approaches
        ),
        comparable=len(comparable(approaches, omms)),
        first_tca=by_tca[0].tca.isot,
        last_tca=by_tca[-1].tca.isot,
        newest_epoch=Time(newest, format="jd", scale="utc").isot[:16],
    )


@dataclass(frozen=True, slots=True)
class Inputs:
    """What the committed files are made from.

    Parameters
    ----------
    downloads
        The saved bodies, in download order (groups, then SOCRATES).
    group_rows
        Element sets in each group's body, by group name.
    duplicates
        Element sets left out because the object has a newer one.
    omms
        The catalog, one element set per object, by catalog number.
    socrates
        The SOCRATES report, whole or the excerpt.
    summary
        Counts and times of the whole report.
    """

    downloads: tuple[Body, ...]
    group_rows: dict[str, int]
    duplicates: int
    omms: tuple[Omm, ...]
    socrates: str
    summary: SocratesSummary


def inputs_from_raw(raw: Path) -> Inputs:
    """The inputs, from the bodies saved in ``raw`` (no network).

    Raises
    ------
    ValueError
        If the catalog would not read back from CSV as the same element sets.
    """
    catalog, report = fetch(raw, None)
    snapshot_files.check_round_trip(catalog.omms)
    return Inputs(
        downloads=INDEX.read(raw),
        group_rows={part.name: len(part.omms) for part in catalog.parts},
        duplicates=len(catalog.duplicates),
        omms=catalog.omms,
        socrates=report.text,
        summary=summarise_socrates(report.approaches(), catalog.omms),
    )


def inputs_from_data(data: Path) -> Inputs:
    """The inputs, from the committed catalog and excerpt, and what only the
    manifest records (downloads, group sizes, duplicates, the whole
    report's summary).

    Raises
    ------
    ValueError
        If the manifest is not one this tool writes.
    """
    manifest = read_manifest(data)
    return Inputs(
        downloads=tuple(Body(**body) for body in manifest["downloads"]),
        group_rows=manifest["catalog"]["group_rows"],
        duplicates=manifest["catalog"]["duplicates"],
        omms=read_catalog(data),
        socrates=read_excerpt(data),
        summary=manifest["socrates"],
    )


def contents(inputs: Inputs) -> dict[str, bytes]:
    """Every committed file but the manifest, uncompressed, by name."""
    kept = comparable(read_socrates(inputs.socrates), inputs.omms)
    excerpt = socrates_header(inputs.socrates) + "".join(a.line for a in kept)
    fixture = select_fixture(inputs.omms, kept)
    return {
        SNAPSHOT: dumps_csv(inputs.omms).encode(),
        SOCRATES: excerpt.encode(),
        FIXTURE: dumps_csv(fixture).encode(),
        NOTICE: render_notice(inputs).encode(),
    }


def retrieved_span(inputs: Inputs) -> tuple[str, str]:
    """The first and last retrieval times, UTC ISO 8601."""
    times = sorted(body.retrieved for body in inputs.downloads)
    return times[0], times[-1]


def render_notice(inputs: Inputs) -> str:
    """The NOTICE: the data's source, retrieval time, files and licence."""
    start, end = retrieved_span(inputs)
    first, last = second_text(start), second_text(end)
    until = last[11:] if day_text(end) == day_text(start) else last
    groups = textwrap.fill(
        f"- {SNAPSHOT}: the GP element sets of CelesTrak's groups "
        f"{', '.join(inputs.group_rows)}, one (the newest) per object;",
        width=76,
        subsequent_indent="  ",
    )
    return (
        "Data: USSPACECOM/18 SDS via Space-Track.org, retrieved from CelesTrak\n"
        f"(celestrak.org) on {day_text(start)}.\n"
        "\n"
        f"Retrieved {first} to {until} UTC by tools/screening_snapshot.py:\n"
        "\n"
        f"{groups}\n"
        f"- {FIXTURE}: about 200 of those objects (see the tool);\n"
        f"- {SOCRATES}: the approaches of the SOCRATES Plus report\n"
        "  (sorted by minimum range) whose two objects are in the catalog with\n"
        "  the element sets SOCRATES used, lines as downloaded.\n"
        "\n"
        f"{MANIFEST} lists the URLs, retrieval times, counts and SHA-256\n"
        "digests.\n"
        "\n"
        f"{REDISTRIBUTION}"
    )


def render_manifest(inputs: Inputs, files: Mapping[str, bytes]) -> str:
    """The manifest: downloads, counts and file digests.

    Parameters
    ----------
    inputs
        What the files are made from.
    files
        The stored bytes of every other committed file, by name.
    """
    fixture = loads_csv(contents_of(FIXTURE, files[FIXTURE]).decode())
    first, last = retrieved_span(inputs)
    manifest: Manifest = {
        "format": MANIFEST_FORMAT,
        "retrieved": {"first": first, "last": last},
        "downloads": [body.record() for body in inputs.downloads],
        "catalog": {
            "group_rows": inputs.group_rows,
            "duplicates": inputs.duplicates,
            **catalog_counts(inputs.omms, regime, REGIMES),
        },
        "socrates": inputs.summary,
        "fixture": {
            "objects": len(fixture),
            "regimes": {
                name: sum(regime(omm) == name for omm in fixture) for name in REGIMES
            },
        },
        "files": file_digests(files),
    }
    return json_text(manifest)


def contents_of(name: str, stored: bytes) -> bytes:
    """A committed file's contents: decompressed if it is gzipped."""
    return gzip.decompress(stored) if name in GZIPPED else stored


def stored_files(data: Path, inputs: Inputs) -> dict[str, bytes]:
    """The bytes every committed file should hold, the manifest last
    (:func:`snapshot_files.stored_files`)."""
    manifest = functools.partial(render_manifest, inputs)
    return snapshot_files.stored_files(data, contents(inputs), GZIPPED, manifest)


def main(argv: Sequence[str] | None = None) -> int:
    """The entry point: downloading the bodies, building the data, or verifying it.

    Returns
    -------
    int
        0 on success; 1 if ``build --check`` or ``verify`` found a file out
        of date.
    """
    parser = argparse.ArgumentParser(prog=TOOL)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser(
        "download", help="download each URL once, saving the bodies in RAW"
    )
    download.add_argument("raw", type=Path)
    snapshot_files.add_data_commands(commands.add_parser, DATA, "bodies")
    args = parser.parse_args(argv)
    if args.command == "download":
        catalog, report = fetch(args.raw, http_download)
        print(
            f"saved {len(catalog.parts) + 1} bodies in {args.raw}: "
            f"{len(catalog.omms)} objects, "
            f"{len(report.approaches())} SOCRATES approaches"
        )
        return 0
    return snapshot_files.build_or_verify(
        args, inputs_from_raw, inputs_from_data, stored_files, TOOL
    )


if __name__ == "__main__":
    sys.exit(main())
