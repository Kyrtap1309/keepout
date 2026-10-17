"""Download Space-Track's GP catalog and public CDMs once; freeze the catalog.

Usage::

    uv run python tools/spacetrack_snapshot.py download RAW [--cache DIR]
    uv run python tools/spacetrack_snapshot.py build RAW [--check] [--data DIR]
    uv run python tools/spacetrack_snapshot.py verify [--data DIR]
    uv run python tools/spacetrack_snapshot.py compare RAW [--workers N]

``download`` needs a Space-Track account (``SPACETRACK_IDENTITY`` and
``SPACETRACK_PASSWORD``, see :mod:`keepout.sources.spacetrack`). With one
:class:`~keepout.sources.spacetrack.SpaceTrackClient` it sends four requests,
each once: the login, the full GP catalog
(:data:`~keepout.sources.spacetrack.CATALOG`), the public CDM summaries of
the approaches to come (:data:`~keepout.sources.spacetrack.PUBLIC_CDMS`) and
the logout. Each answer is saved unchanged in the new directory ``RAW``
(listed with its status, digest and time in :data:`INDEX`) before the
client reads it; an answer holding the account's identity or password is
saved with them replaced by ``<hidden>``. Nothing is retried: a second
request to a URL, or a ``RAW`` that already holds a download, is refused.
The client's cache (the user's, or ``--cache``) records the queries, so the
client's limits (``gp`` once an hour, ``cdm_public`` once every 8 hours)
also hold for the next ``download`` or ``keepout fetch``.

``build`` works offline from ``RAW`` alone: it replays the saved answers to
keepout's client (:func:`~keepout.sources.spacetrack.fetch_catalog` with a
new, empty cache), so the catalog is read exactly as a download would be,
and writes the committed data in ``tests/screening/spacetrack-data/``:

- :data:`SNAPSHOT`: the catalog, one element set per object (the newest),
  by catalog number, as gzipped CelesTrak CSV (the OMM keywords;
  Space-Track's further columns are left out);
- :data:`NOTICE` and :data:`~snapshot_files.MANIFEST`: source, citation and
  licence; the download, counts and SHA-256 digests.

``build --check`` fails instead of writing when a file is out of date;
``verify`` checks the committed files against each other offline. Output is
deterministic (gzip without a time stamp; a gzipped file whose contents are
unchanged is kept as it is).

``compare`` reads the public CDMs in ``RAW`` and screens, for each, its two
objects with the catalog's element sets around Space-Track's TCA
(:func:`compare`), printing a summary. Public CDMs, and anything derived
from them, may not be redistributed without Space-Track's approval
(user agreement): ``RAW`` stays on the maintainer's disk, nothing of it is
committed, and the summary is printed, not written. Only the GP data, basic
space situational awareness data that Space-Track allows to redistribute
with citation, is committed.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypedDict, cast

import snapshot_files
from generated import REPO, json_text, positive
from screening_snapshot import REGIMES, nearest_approach, regime, utc_text
from snapshot_files import (
    MANIFEST,
    MANIFEST_FORMAT,
    RAW_INDEX,
    CatalogCounts,
    FileDigest,
    RawIndex,
    catalog_counts,
    file_digests,
    read_body,
)

from keepout._workers import process_pool, resolve_workers
from keepout.errors import SourceFormatError, SourceUnavailable
from keepout.omm import Omm, load
from keepout.omm.celestrak import dumps_csv
from keepout.screening.result import CloseApproach
from keepout.sources.cache import store_bytes
from keepout.sources.catalog import Catalog
from keepout.sources.spacetrack import (
    CATALOG,
    HIDDEN,
    LOGIN_URL,
    LOGOUT_URL,
    PUBLIC_CDMS,
    Credentials,
    PublicCdm,
    Request,
    Response,
    SpaceTrackClient,
    Transport,
    UrllibTransport,
    fetch_catalog,
    fetch_public_cdms,
    read_public_cdms,
)
from keepout.sources.spacetrack import read_catalog as read_gp
from keepout.units import seconds_between

TOOL: Final = "tools/spacetrack_snapshot.py"
DATA: Final = REPO / "tests/screening/spacetrack-data"
"""Where ``build`` writes the committed data."""

RAW_VARIABLE: Final = "KEEPOUT_SPACETRACK_RAW"
"""The environment variable naming a ``RAW`` directory for the local
validation tests (they are skipped without it)."""

RAW_FILES: Final[Mapping[str, str]] = {
    LOGIN_URL: "login.json",
    CATALOG.url: "gp.json",
    PUBLIC_CDMS.url: "cdm_public.json",
    LOGOUT_URL: "logout.json",
}
"""The only URLs ``download`` sends to, in order, and the file of each
answer in ``RAW``."""

SNAPSHOT: Final = "spacetrack-gp.csv.gz"
"""The catalog: CelesTrak CSV of every object, one element set per object,
by catalog number, gzipped."""

NOTICE: Final = "NOTICE"
"""Source, citation, retrieval time and licence of the data."""

REPLAY: Final = Credentials("replay", "replay")
"""Credentials for replaying saved answers (no server sees them)."""

WINDOW: Final = 600.0
"""Time searched on each side of a public CDM's TCA [s] (:func:`compare`)."""

SEARCH_RADIUS: Final = 100e3
"""Radius of the sphere searched around the first object [m]: far beyond
the errors of GP data, so the range minimum is found wherever it is."""


class SavedRecord(TypedDict):
    """A :class:`Saved` answer as JSON, in the raw index and the manifest."""

    method: str
    url: str
    status: int
    file: str
    sha256: str
    size: int
    retrieved: str
    redacted: bool


@dataclass(frozen=True, slots=True)
class Saved:
    """One answer saved in ``RAW``.

    Parameters
    ----------
    method
        The request's method, ``GET`` or ``POST``.
    url
        The URL it answered.
    status
        Its HTTP status.
    file
        Its file name in ``RAW``.
    sha256
        SHA-256 hex digest of the saved body.
    size
        Length of the saved body in bytes.
    retrieved
        When it was received, UTC, ISO 8601 with milliseconds and a ``Z``.
    redacted
        Whether the identity or password were replaced by ``<hidden>``.
    """

    method: str
    url: str
    status: int
    file: str
    sha256: str
    size: int
    retrieved: str
    redacted: bool

    def record(self) -> SavedRecord:
        """The answer as JSON."""
        return cast("SavedRecord", dataclasses.asdict(self))


INDEX: Final = RawIndex("answers", Saved)
"""The answers saved in ``RAW``, in :data:`~snapshot_files.RAW_INDEX`."""


def scrub(body: bytes, credentials: Credentials) -> bytes:
    """``body`` with the identity and password hidden
    (:meth:`~keepout.sources.spacetrack.Credentials.redact`); the same bytes
    if it holds neither."""
    text = body.decode("utf-8", "surrogateescape")
    clean = credentials.redact(text)
    return body if clean == text else clean.encode("utf-8", "surrogateescape")


class Recorder:
    """A transport that sends each of :data:`RAW_FILES`'s URLs at most once
    and saves every answer in ``raw`` before returning it.

    Parameters
    ----------
    transport
        Sends the requests.
    raw
        The directory; created when the first answer is saved.
    credentials
        The account, hidden in the saved answers.
    clock
        The current Unix time.
    """

    def __init__(
        self,
        transport: Transport,
        raw: Path,
        credentials: Credentials,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.transport = transport
        self.raw = raw
        self.credentials = credentials
        self.clock = clock
        self.saved: dict[str, Saved] = {}
        self.sent: set[str] = set()

    def __call__(self, request: Request, /) -> Response:
        """Send ``request`` once and save its answer.

        Raises
        ------
        SourceUnavailable
            If the URL is not one of :data:`RAW_FILES` or was sent before
            (nothing is sent).
        """
        name = RAW_FILES.get(request.url)
        if name is None:
            raise SourceUnavailable(f"{TOOL} does not send to {request.url}")
        if request.url in self.sent:
            raise SourceUnavailable(
                f"{TOOL} sends each request once; {request.url} was sent already"
            )
        self.sent.add(request.url)
        response = self.transport(request)
        body = scrub(response.body, self.credentials)
        digest = store_bytes(self.raw / name, body)
        self.saved[request.url] = Saved(
            method=request.method,
            url=request.url,
            status=response.status,
            file=name,
            sha256=digest,
            size=len(body),
            retrieved=utc_text(self.clock()),
            redacted=body != response.body,
        )
        INDEX.write(self.raw, tuple(self.saved.values()))
        return response


class Replay:
    """A transport answering from the answers saved in ``raw`` (offline).

    Raises
    ------
    ValueError
        If ``raw`` has no index, or a saved body does not match its digest.
    """

    def __init__(self, raw: Path) -> None:
        self.raw = raw
        self.saved = {saved.url: saved for saved in INDEX.read(raw)}
        if not self.saved:
            raise ValueError(f"no answers saved in {raw}")

    def __call__(self, request: Request, /) -> Response:
        """The saved answer to ``request``.

        Raises
        ------
        SourceUnavailable
            If no answer to its URL is saved.
        """
        saved = self.saved.get(request.url)
        if saved is None:
            raise SourceUnavailable(f"{request.url} is not saved in {self.raw}")
        return Response(saved.status, read_body(self.raw, saved))


def download(
    raw: Path,
    *,
    cache: Path | None = None,
    credentials: Credentials | None = None,
    transport: Transport | None = None,
    clock: Callable[[], float] = time.time,
) -> tuple[Saved, ...]:
    """Send the four requests of :data:`RAW_FILES` once, saving the answers.

    Parameters
    ----------
    raw
        A directory without a download; created.
    cache
        The client's cache root (its request record keeps the limits);
        the user's if ``None``.
    credentials
        The account; from the environment if ``None``.
    transport
        Sends the requests; a new
        :class:`~keepout.sources.spacetrack.UrllibTransport` if ``None``.
    clock
        The current Unix time.

    Returns
    -------
    tuple[Saved, ...]
        The saved answers, in the order received.

    Raises
    ------
    ValueError
        If ``raw`` already holds a download, or a query was answered from the
        client's cache instead of Space-Track (nothing more is sent).
    RateLimited, AuthError, SourceUnavailable, SourceFormatError
        As :meth:`~keepout.sources.spacetrack.SpaceTrackClient.fetch`; the
        answers received so far are saved.
    """
    if (raw / RAW_INDEX).exists():
        raise ValueError(
            f"{raw} already holds a download; Space-Track data is downloaded "
            "once, into a new directory"
        )
    if credentials is None:
        credentials = Credentials.from_env()
    recorder = Recorder(
        UrllibTransport() if transport is None else transport,
        raw,
        credentials,
        clock,
    )
    with SpaceTrackClient(
        credentials, cache=cache, transport=recorder, now=clock
    ) as client:
        for query in (CATALOG, PUBLIC_CDMS):
            # The answer is read after the download, from RAW.
            _, retrieval = client.fetch(query, bytes)
            if query.url not in recorder.saved:
                raise ValueError(
                    f"{query.url} was answered from the cache "
                    f"({retrieval.path}); download into a new cache"
                )
    return tuple(recorder.saved.values())


def saved_answer(raw: Path, url: str) -> bytes:
    """The body of the answer to ``url`` saved in ``raw``.

    Raises
    ------
    ValueError
        If no answer to ``url`` is saved, or the body does not match its
        digest.
    """
    for saved in INDEX.read(raw):
        if saved.url == url:
            return read_body(raw, saved)
    raise ValueError(f"no answer to {url} in {raw}")


def replay_catalog(raw: Path) -> Catalog:
    """The catalog, read by keepout's client from the answers in ``raw``."""
    with tempfile.TemporaryDirectory() as cache:
        return fetch_catalog(
            cache=Path(cache), credentials=REPLAY, transport=Replay(raw)
        )


def replay_public_cdms(raw: Path) -> tuple[PublicCdm, ...]:
    """The public CDMs, read by keepout's client from the answers in ``raw``."""
    with tempfile.TemporaryDirectory() as cache:
        return fetch_public_cdms(
            cache=Path(cache), credentials=REPLAY, transport=Replay(raw)
        ).cdms


class CatalogSummary(CatalogCounts):
    """What the manifest records of the catalog."""

    rows: int
    duplicates: int


class Manifest(TypedDict):
    """The :data:`~snapshot_files.MANIFEST` file."""

    format: Literal[1]
    download: SavedRecord
    catalog: CatalogSummary
    files: dict[str, FileDigest]


@dataclass(frozen=True, slots=True)
class Inputs:
    """What the committed files are made from.

    Parameters
    ----------
    download
        The saved ``gp`` answer.
    rows
        Element sets in it.
    duplicates
        Element sets left out because the object has a newer one.
    omms
        The catalog, one element set per object, by catalog number.
    """

    download: Saved
    rows: int
    duplicates: int
    omms: tuple[Omm, ...]


def inputs_from_raw(raw: Path) -> Inputs:
    """The inputs, from the answers saved in ``raw`` (no network).

    Raises
    ------
    ValueError
        If ``raw`` has no ``gp`` answer, or the catalog would not read back
        from CSV as the same element sets.
    """
    catalog = replay_catalog(raw)
    snapshot_files.check_round_trip(catalog.omms)
    (gp,) = (saved for saved in INDEX.read(raw) if saved.url == CATALOG.url)
    return Inputs(
        download=gp,
        rows=len(catalog.parts[0].omms),
        duplicates=len(catalog.duplicates),
        omms=catalog.omms,
    )


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


def retrieval_day(manifest: Manifest) -> str:
    """The day the catalog was retrieved, UTC, ISO 8601."""
    return manifest["download"]["retrieved"][:10]


def inputs_from_data(data: Path) -> Inputs:
    """The inputs, from the committed catalog and what only the manifest
    records (the download, the rows and duplicates).

    Raises
    ------
    ValueError
        If the manifest is not one this tool writes.
    """
    manifest = read_manifest(data)
    return Inputs(
        download=Saved(**manifest["download"]),
        rows=manifest["catalog"]["rows"],
        duplicates=manifest["catalog"]["duplicates"],
        omms=read_catalog(data),
    )


def render_notice(inputs: Inputs) -> str:
    """The NOTICE: the data's source, citation, retrieval time and licence."""
    retrieved = inputs.download.retrieved[:19].replace("T", " ")
    return (
        "Data: USSPACECOM/18 SDS via Space-Track.org (www.space-track.org),\n"
        f"retrieved on {retrieved[:10]}.\n"
        "\n"
        f"Retrieved {retrieved} UTC by tools/spacetrack_snapshot.py with\n"
        "Space-Track's recommended query for the GP catalog (see manifest.json):\n"
        "\n"
        f"- {SNAPSHOT}: the GP element sets of every object not decayed\n"
        "  with an epoch in the 10 days before, one (the newest) per object, by\n"
        "  catalog number, as gzipped CelesTrak CSV (the OMM keywords).\n"
        "\n"
        f"{MANIFEST} lists the URL, retrieval time, counts and SHA-256\n"
        "digests.\n"
        "\n"
        "This is basic space situational awareness data, which the\n"
        "Space-Track.org user agreement allows to redistribute with the\n"
        "citation above. It is not covered by keepout's MIT licence.\n"
    )


def contents(inputs: Inputs) -> dict[str, bytes]:
    """Every committed file but the manifest, uncompressed, by name."""
    return {
        SNAPSHOT: dumps_csv(inputs.omms).encode(),
        NOTICE: render_notice(inputs).encode(),
    }


def render_manifest(inputs: Inputs, files: Mapping[str, bytes]) -> str:
    """The manifest: the download, counts and file digests.

    Parameters
    ----------
    inputs
        What the files are made from.
    files
        The stored bytes of every other committed file, by name.
    """
    manifest: Manifest = {
        "format": MANIFEST_FORMAT,
        "download": inputs.download.record(),
        "catalog": {
            "rows": inputs.rows,
            "duplicates": inputs.duplicates,
            **catalog_counts(inputs.omms, regime, REGIMES),
        },
        "files": file_digests(files),
    }
    return json_text(manifest)


def stored_files(data: Path, inputs: Inputs) -> dict[str, bytes]:
    """The bytes every committed file should hold, the manifest last
    (:func:`snapshot_files.stored_files`)."""
    manifest = functools.partial(render_manifest, inputs)
    return snapshot_files.stored_files(data, contents(inputs), {SNAPSHOT}, manifest)


@dataclass(frozen=True, slots=True)
class Match:
    """A public CDM and keepout's nearest approach of the same two objects.

    Parameters
    ----------
    cdm
        The public CDM.
    approach
        keepout's approach of the two objects nearest Space-Track's TCA;
        ``None`` if keepout found no range minimum within :data:`WINDOW` of
        it.
    """

    cdm: PublicCdm
    approach: CloseApproach | None

    @property
    def tca_difference(self) -> float | None:
        """TCA by keepout minus TCA by Space-Track [s]; ``None`` without an
        approach."""
        if self.approach is None:
            return None
        return seconds_between(self.cdm.tca, self.approach.tca)

    @property
    def miss_distance(self) -> float | None:
        """Range by keepout at its TCA [m]; ``None`` without an approach."""
        return None if self.approach is None else self.approach.miss_distance


def _match(job: tuple[PublicCdm, Omm, Omm]) -> Match:
    """The CDM's match: its two objects screened with each other from
    :data:`WINDOW` before to :data:`WINDOW` after its TCA, in a
    :data:`SEARCH_RADIUS` sphere (:func:`screening_snapshot.nearest_approach`)."""
    cdm, first, second = job
    approach = nearest_approach(
        first, second, cdm.tca, window=WINDOW, radius=SEARCH_RADIUS
    )
    return Match(cdm, approach)


@dataclass(frozen=True, slots=True)
class Comparison:
    """Public CDMs compared with keepout's screening of the same pairs.

    Parameters
    ----------
    matches
        The CDMs whose two objects are in the catalog, in the CDMs' order.
    outside_catalog
        The CDMs with an object not in the catalog.
    """

    matches: tuple[Match, ...]
    outside_catalog: int


def compare(
    cdms: Sequence[PublicCdm], omms: Sequence[Omm], *, workers: int = 1
) -> Comparison:
    """Screen the two objects of every public CDM with the catalog's
    element sets around its TCA (:class:`Match`).

    Parameters
    ----------
    cdms
        The public CDMs.
    omms
        The catalog, one element set per object.
    workers
        Processes to screen in; 1 screens in this one.
    """
    by_number = {omm.catalog_number: omm for omm in omms}
    jobs = [
        (
            cdm,
            by_number[cdm.object1.catalog_number],
            by_number[cdm.object2.catalog_number],
        )
        for cdm in cdms
        if cdm.object1.catalog_number in by_number
        and cdm.object2.catalog_number in by_number
    ]
    if workers == 1:
        matches = [_match(job) for job in jobs]
    else:
        with process_pool(resolve_workers(workers)) as pool:
            matches = list(pool.map(_match, jobs, chunksize=16))
    return Comparison(tuple(matches), len(cdms) - len(jobs))


class Spread(TypedDict):
    """Median and 90th percentile of a quantity; ``None`` without values."""

    median: float | None
    p90: float | None


class ComparisonSummary(TypedDict):
    """What :func:`summarise` reports of a comparison."""

    cdms: int
    outside_catalog: int
    compared: int
    found: int
    tca_difference: Spread
    """Of |keepout's TCA - Space-Track's| [s]."""
    range_difference: Spread
    """Of |keepout's range - MIN_RNG| [m]."""
    range_ratio: Spread
    """Of MIN_RNG [m] / keepout's range: of order 1 (below 1, GP data's
    errors lengthening keepout's ranges), not 1000 as if MIN_RNG were in
    kilometres."""


def spread(values: Sequence[float]) -> Spread:
    """The median and 90th percentile of ``values``."""
    if not values:
        return {"median": None, "p90": None}
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "p90": ordered[min(len(ordered) - 1, (9 * len(ordered)) // 10)],
    }


def summarise(comparison: Comparison) -> ComparisonSummary:
    """Counts, and how keepout's approaches differ from the public CDMs."""
    found = [
        (match.cdm, match.approach)
        for match in comparison.matches
        if match.approach is not None
    ]
    misses = [(cdm.min_range, approach.miss_distance) for cdm, approach in found]
    return {
        "cdms": len(comparison.matches) + comparison.outside_catalog,
        "outside_catalog": comparison.outside_catalog,
        "compared": len(comparison.matches),
        "found": len(found),
        "tca_difference": spread(
            [abs(seconds_between(cdm.tca, approach.tca)) for cdm, approach in found]
        ),
        "range_difference": spread([abs(miss - rng) for rng, miss in misses]),
        "range_ratio": spread([rng / miss for rng, miss in misses if miss > 0.0]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Download the answers, build the data from them, verify it, or compare.

    Returns
    -------
    int
        0 on success; 1 if ``build --check`` or ``verify`` found a file out
        of date, or ``download`` saved an answer keepout cannot read.
    """
    parser = argparse.ArgumentParser(prog=TOOL)
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser(
        "download", help="send the four requests once, saving the answers in RAW"
    )
    get.add_argument("raw", type=Path)
    get.add_argument("--cache", type=Path, default=None)
    snapshot_files.add_data_commands(commands.add_parser, DATA, "answers")
    check = commands.add_parser(
        "compare", help="compare the public CDMs in RAW with keepout (offline)"
    )
    check.add_argument("raw", type=Path)
    check.add_argument("--workers", type=positive, default=1)
    args = parser.parse_args(argv)
    if args.command == "download":
        return _download(args.raw, args.cache)
    if args.command == "compare":
        comparison = compare(
            replay_public_cdms(args.raw),
            replay_catalog(args.raw).omms,
            workers=args.workers,
        )
        print(json.dumps(summarise(comparison), indent=2))
        return 0
    return snapshot_files.build_or_verify(
        args, inputs_from_raw, inputs_from_data, stored_files, TOOL
    )


def _download(raw: Path, cache: Path | None) -> int:
    """``download``: save the answers, then read them from ``raw``."""
    for saved in download(raw, cache=cache):
        hidden = f" ({HIDDEN} credentials)" if saved.redacted else ""
        print(
            f"{saved.method} {saved.url}: HTTP {saved.status}, "
            f"{saved.size} bytes in {saved.file}{hidden}"
        )
    try:
        omms = read_gp(saved_answer(raw, CATALOG.url))
        cdms = read_public_cdms(saved_answer(raw, PUBLIC_CDMS.url))
    except SourceFormatError as exc:
        print(f"saved, but keepout cannot read it: {exc}")
        return 1
    print(f"{len(omms)} element sets, {len(cdms)} public CDMs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
