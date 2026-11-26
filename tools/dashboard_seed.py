"""The dashboard's seed: CDM files of keepout's example synthetic operator on
the frozen catalog, the operator settings to assess them with, and a store
made of them.

Usage: ``uv run python tools/dashboard_seed.py [--check] [--cara] [--dir DIR]``.

Runs the example fleet (``examples/synthetic-operator/fleet.toml``) against
the committed public catalog (``tests/screening/data/``) as
``tools/synthetic_operator_report.py`` does, but as an operator receives
updates: every :data:`CADENCE` (12 h) over the first two days of the
snapshot's week (:data:`ISSUES`, 4 runs), each run screening the rest of the
week and issuing its CDMs at its own start (``CREATION_DATE``). An approach
screened by several runs thus gets several CDMs, which a store groups into
one event. Writes each CDM to ``examples/dashboard/cdms/<MESSAGE_ID>.kvn``
(:data:`SEED_DIR`), removing any other ``.kvn`` file there. The creation
dates follow from the snapshot, not the clock, so a run gives the same files
again; ``TCA`` is rounded to the millisecond and the real values to
:data:`SIGNIFICANT_DIGITS` (:func:`seed_text`) so that no digit depends on
the machine. Needs no network. ``--check`` fails instead of writing, naming
each file that differs or should not be there.

Every run screens the same element sets, and the model's covariance depends
on the time from an element set's epoch to TCA, not on the creation date:
an event's CDMs give the same Pc (a flat trend), and only the time to TCA,
and so the decision, changes between them.

The demo: :data:`DEMO_DIR` holds one more CDM, :func:`demo_update`, which
the demo uploads to show a rising trend; writing and ``--check`` cover it.
``--store PATH`` builds the demo's store instead (:func:`demo_store`): the
seed with its Pc and the cached manoeuvre plan of :data:`DEMO_EVENT`, as
the Docker image packages it.

``--cara`` adds CARA's Metop-B test CDM (:data:`CARA_CDM_ID`) as
``<id>.cdm``. CARA's files are not redistributed: it is read from the local
cache only (``keepout fetch cara`` puts it there), never downloaded, and
``examples/dashboard/.gitignore`` keeps it out of git. Without ``--cara``,
neither writing nor ``--check`` touches a ``.cdm`` file.

:func:`seed_store` makes a store of the files in :data:`SEED_DIR`, assessed with
``examples/dashboard/settings.toml`` (:data:`SETTINGS`), for the dashboard's
tests and demo; :func:`seed_as_of` is the time the seed is seen at.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

import screening_snapshot as snapshot
import synthetic_operator_report as report
from astropy import units as u
from astropy.time import Time
from generated import REPO, write_or_check

from keepout._workers import DEFAULT_WORKERS
from keepout.cdm import Cdm, kvn, load, make_message_id, read_text
from keepout.cdm.keywords import covariance_names
from keepout.cdm.model import Field, Section
from keepout.errors import SourceUnavailable
from keepout.screening import DEFAULT_DURATION, DEFAULT_VOLUME
from keepout.settings import load_settings
from keepout.sources import cara
from keepout.sources.catalog import CatalogReference
from keepout.synthetic import load_fleet
from keepout.synthetic.operator import run
from keepout.units import DAY, elapsed_seconds

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from keepout.ccsds.keywords import FieldValue
    from keepout.store import CdmInput


TOOL: Final = "tools/dashboard_seed.py"
SEED_DIR: Final = REPO / "examples/dashboard/cdms"
"""The seed's CDM files."""

SETTINGS: Final = REPO / "examples/dashboard/settings.toml"
"""The operator settings the seed is assessed with."""

SYNTHETIC: Final = ".kvn"
"""The suffix of the synthetic operator's files: the committed seed."""

CARA_SUFFIX: Final = ".cdm"
"""The suffix of the CARA file, which is not committed."""

DEMO_DIR: Final = REPO / "examples/dashboard/demo"
"""The CDM the demo uploads (:func:`demo_update`)."""

DEMO_EVENT: Final = "3befe6f1305d"
"""The seed's event the demo plans and updates: INTELSAT 39 (44476) against
55841, yellow, its TCA (2026-10-08 12:03 UTC) a day after the seed's last
issue, and quick to plan (about 10 s; the other upcoming events take 18 to
55 s, and the first cannot be planned: its burn window has closed)."""

DEMO_OBJECTS: Final = ("44476", "55841")
"""The designators of :data:`DEMO_EVENT`'s objects."""

DEMO_AS_OF: Final = Time("2026-10-07T18:00:00", scale="utc")
"""The creation of the demo's update, and the time the Docker demo is seen
at (``--as-of`` in ``compose.yaml``): 6 hours after the seed's last issue,
so that nothing shown is newer than the time it is seen at."""

DEMO_COMMENT: Final = (
    "keepout demo: the seed's latest CDM of this event, re-issued with a "
    "smaller miss and covariance to show a rising trend"
)
"""The first header comment of the demo's update."""

MISS_SCALE: Final = 0.1
"""How much :func:`demo_update` shrinks the miss (3647 m to 365 m)."""

COVARIANCE_SCALE: Final = 0.04
"""How much :func:`demo_update` scales both covariances (each sigma by 1/5,
as fresher element sets would give). The miss alone cannot make the event
red: with the seed's covariances (sigma T about 11 km at GEO) its Pc stops
near 1.5e-5 as the miss goes to 0. With both, the update's Pc is 3.2e-4,
red under :data:`SETTINGS` (red from 1e-4); a miss of 0.25 gives 1.2e-4,
too near the threshold."""

CADENCE: Final = DAY / 2
"""How often [s] the seed's operator screens and issues CDMs."""

ISSUES: Final = 4
"""How many times it does: over the first two days of the week, so that at
the last issue, the time the seed is seen at (:func:`seed_as_of`), most of
the week's approaches are still ahead."""

SIGNIFICANT_DIGITS: Final = 7
"""Significant digits of the seed's real values: fewer than the report's
10, as the TCA a screening finds can move by tens of microseconds between
runs and machines (the runs of one approach give TCAs up to 17 us apart,
moving the states at their 8th digit); 7 digits keep the metre."""

_TCA: Final = re.compile(r"^(?P<key>TCA\s*=\s*)(?P<value>\S+)$")

CARA_CDM_ID: Final = "000038771_conj_000030802_20201216_182131_20201215_171306"
"""CARA's Metop-B test CDM, the one ``tools/maneuver_report.py`` plans
(its ``CDM_ID``): Metop-B (38771) against 30802."""

CARA_FILE: Final = cara.PC_METHOD_CDMS[CARA_CDM_ID]
"""Where that CDM is in CARA's repository, and its digest."""


def seed_text(text: str) -> str:
    """A synthetic CDM's KVN ``text`` as the seed writes it: ``TCA`` to the
    millisecond (as the ``MESSAGE_ID`` gives it) and every real value to
    :data:`SIGNIFICANT_DIGITS` significant digits, those below
    :data:`synthetic_operator_report.NOISE_FLOOR` as 0.0
    (:func:`synthetic_operator_report.rounded`)."""

    def tca(line: str) -> str:
        match = _TCA.match(line)
        if match is None:
            return line
        value = Time(match["value"], scale="utc", precision=3).isot
        return f"{match['key']}{value}"

    lines = report.rounded(text, SIGNIFICANT_DIGITS).split("\n")
    return "\n".join(tca(line) for line in lines)


def issue_times(start: Time) -> list[Time]:
    """When the seed's operator issues CDMs: :data:`ISSUES` times, every
    :data:`CADENCE` from ``start`` (the start of the snapshot's week)."""
    return [start + index * CADENCE * u.s for index in range(ISSUES)]


def synthetic_cdms(*, workers: int = DEFAULT_WORKERS) -> list[tuple[str, str]]:
    """The synthetic operator's CDMs, issued at each of :func:`issue_times`
    for the approaches after it in the snapshot's week: each file name
    (``<MESSAGE_ID>.kvn``) and rounded KVN text, by file name.

    Parameters
    ----------
    workers
        Processes to screen with; the CDMs do not depend on it.
    """
    start = report.window_start(snapshot.read_manifest(snapshot.DATA))
    end = start + DEFAULT_DURATION * u.s
    fleet = load_fleet(report.FLEET)
    catalog = snapshot.read_catalog(snapshot.DATA)
    cdms = [
        cdm
        for issued in issue_times(start)
        for cdm in run(
            fleet,
            catalog,
            issued,
            elapsed_seconds(issued, end),
            DEFAULT_VOLUME,
            workers=workers,
            thresholds=report.THRESHOLDS,
            decision_config=report.DECISION,
        ).cdms
    ]
    return sorted(
        (f"{cdm.message_id}{SYNTHETIC}", seed_text(kvn.dumps(cdm))) for cdm in cdms
    )


def _not_cached(url: str) -> bytes:
    """What a download of the CARA CDM does here: refuse.

    Raises
    ------
    SourceUnavailable
        Always.
    """
    raise SourceUnavailable(
        f"the CARA CDM {CARA_CDM_ID} is not in the cache ({url} is not "
        "downloaded here); run keepout fetch cara"
    )


def cara_cdm(*, cache: Path | None = None) -> tuple[str, str]:
    """CARA's Metop-B CDM from the local cache, its digest verified: its file
    name (``<id>.cdm``) and text.

    Parameters
    ----------
    cache
        The cache directory, as for :func:`keepout.sources.cara.commit_dir`.

    Raises
    ------
    SourceUnavailable
        If the cache has no copy of it with CARA's digest; nothing is
        downloaded.
    """
    path = cara.fetch(CARA_FILE, cache=cache, download=_not_cached)
    return f"{CARA_CDM_ID}{CARA_SUFFIX}", path.read_text(encoding="utf-8")


def seed_cdms(*, cara: bool) -> list[tuple[str, str]]:
    """The seed's CDMs: each file name and text, the same on every run.

    Parameters
    ----------
    cara
        Add CARA's Metop-B CDM from the local cache (:func:`cara_cdm`).

    Raises
    ------
    SourceUnavailable
        If ``cara`` and the cache lacks the CDM.
    """
    extra = [cara_cdm()] if cara else []
    return [*synthetic_cdms(), *extra]


def _owned(directory: Path, *, cara: bool) -> list[Path]:
    """The files in ``directory`` that a seed of ``cara`` writes or removes."""
    suffixes = {SYNTHETIC, CARA_SUFFIX} if cara else {SYNTHETIC}
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix in suffixes)


def write_seed(cdms: Sequence[tuple[str, str]], directory: Path, *, check: bool) -> int:
    """Write ``cdms`` into ``directory`` and remove the seed files it holds
    besides, or with ``check`` report each file that differs or should not
    be there.

    A ``.kvn`` file is the seed's; a ``.cdm`` file (CARA's) is only when
    ``cdms`` has one: without ``--cara``, a CARA file added before is
    neither removed nor checked.

    Returns
    -------
    int
        1 if ``check`` found a file out of date or not generated, else 0.

    Raises
    ------
    OSError
        If a file cannot be read, written or removed.
    """
    names = {name for name, _ in cdms}
    cara = any(Path(name).suffix == CARA_SUFFIX for name in names)
    stale = [p for p in _owned(directory, cara=cara) if p.name not in names]
    status = write_or_check(
        {directory / name: text for name, text in cdms}, check=check, tool=TOOL
    )
    for path in stale:
        if check:
            print(f"{path} is not generated; run {TOOL}")
            status = 1
        else:
            path.unlink()
            print(f"removed {path}")
    return status


def _replaced(
    section: Section,
    values: Mapping[str, FieldValue],
    *,
    first_comment: str | None = None,
) -> Section:
    """``section`` with the fields of ``values`` given those values, and
    ``first_comment`` (if given) as its first comment, after the version
    line of a header."""
    fields = [
        dataclasses.replace(field, value=values[field.keyword])
        if field.keyword in values
        else field
        for field in section.fields
    ]
    if first_comment is not None:
        fields.insert(1, Field("COMMENT", first_comment))
    return dataclasses.replace(section, fields=tuple(fields))


def _number(section: Section, keyword: str) -> float:
    """The real value of ``keyword`` in ``section``.

    Raises
    ------
    ValueError
        If the section has no real value of ``keyword``.
    """
    value = section.get(keyword)
    if not isinstance(value, float):
        raise ValueError(f"no real value of {keyword} in the {section.kind} section")
    return value


def demo_update() -> tuple[str, str]:
    """The CDM the demo uploads: :data:`DEMO_EVENT`'s latest seed CDM,
    created at :data:`DEMO_AS_OF` with the miss scaled by
    :data:`MISS_SCALE` and the covariances by :data:`COVARIANCE_SCALE`; its
    file name (``<MESSAGE_ID>.kvn``) and text.

    Object 2 is moved towards object 1 along their separation (its
    velocity, and so the relative velocity, kept), the relative position
    and miss distance scaled with it, and :data:`DEMO_COMMENT` says so.

    Raises
    ------
    ValueError
        If the seed has no CDM of :data:`DEMO_OBJECTS`.
    """
    prefix = "_".join(DEMO_OBJECTS) + "_"
    files = [path for path in seed_files() if path.name.startswith(prefix)]
    if not files:
        raise ValueError(f"no seed CDM of {DEMO_OBJECTS} in {SEED_DIR}; run {TOOL}")
    cdm = max((load(path) for path in files), key=lambda c: c.creation_date.tai.mjd)
    message_id = make_message_id(*DEMO_OBJECTS, cdm.tca, DEMO_AS_OF)
    one, two = cdm.object_sections
    moved: dict[str, FieldValue] = {
        axis: _number(one, axis)
        + MISS_SCALE * (_number(two, axis) - _number(one, axis))
        for axis in ("X", "Y", "Z")
    }
    scaled: dict[str, FieldValue] = {
        keyword: MISS_SCALE * _number(cdm.relative, keyword)
        for keyword in (
            "MISS_DISTANCE",
            "RELATIVE_POSITION_R",
            "RELATIVE_POSITION_T",
            "RELATIVE_POSITION_N",
        )
    }
    covariance = set(covariance_names(9))

    def narrowed(section: Section, values: Mapping[str, FieldValue]) -> Section:
        scaled_covariance: dict[str, FieldValue] = {
            field.keyword: COVARIANCE_SCALE * _number(section, field.keyword)
            for field in section.fields
            if field.keyword in covariance
        }
        return _replaced(section, {**scaled_covariance, **values})

    update = Cdm(
        header=_replaced(
            cdm.header,
            {"CREATION_DATE": DEMO_AS_OF, "MESSAGE_ID": message_id},
            first_comment=DEMO_COMMENT,
        ),
        relative=_replaced(cdm.relative, scaled),
        object_sections=(narrowed(one, {}), narrowed(two, moved)),
    )
    return f"{message_id}{SYNTHETIC}", seed_text(kvn.dumps(update))


def write_demo(*, check: bool) -> int:
    """Write :func:`demo_update` into :data:`DEMO_DIR`, removing any other
    ``.kvn`` file there, or with ``check`` report what differs.

    Returns
    -------
    int
        1 if ``check`` found a file out of date or not generated, else 0.

    Raises
    ------
    OSError
        If a file cannot be read, written or removed.
    """
    return write_seed([demo_update()], DEMO_DIR, check=check)


def demo_file() -> Path:
    """The demo's update CDM in :data:`DEMO_DIR` (:func:`write_demo`).

    Raises
    ------
    ValueError
        If :data:`DEMO_DIR` does not hold exactly one ``.kvn`` file.
    """
    files = sorted(DEMO_DIR.glob("*.kvn"))
    if len(files) != 1:
        raise ValueError(
            f"expected one .kvn file in {DEMO_DIR}, found {len(files)}; run {TOOL}"
        )
    return files[0]


def demo_store(path: Path) -> Path:
    """The demo's store: :func:`seed_store` and the manoeuvre plan of
    :data:`DEMO_EVENT`, computed by a worker and cached, with
    :data:`SETTINGS`.

    The store is closed when this returns, its write-ahead log merged into
    the file: the file and its catalog copies (``<name>.catalogs``) are the
    whole store, to be copied together.

    Parameters
    ----------
    path
        The store's file: made, or added to if it is a store.

    Returns
    -------
    pathlib.Path
        ``path``.

    Raises
    ------
    RuntimeError
        If the plan's job does not succeed.
    StoreError
        As :func:`seed_store`.
    """
    from keepout.store import JobQueue, JobStatus, open_store
    from keepout.store.worker import run_worker

    seed_store(path)
    with open_store(path) as store:
        job = JobQueue(store).submit(DEMO_EVENT, load_settings(SETTINGS)).job
    if job is None:  # Planned before: the store was the demo's already.
        return path
    run_worker(path, until_idle=True)
    with open_store(path) as store:
        (done,) = JobQueue(store).jobs_by_id([job.id])
    if done.status is not JobStatus.SUCCEEDED:
        raise RuntimeError(
            f"the plan of {DEMO_EVENT} did not succeed: {done.status}, {done.error}"
        )
    return path


def seed_files(directory: Path | None = None) -> list[Path]:
    """The seed's CDM files in ``directory`` (:data:`SEED_DIR` if
    ``None``), CARA's if there, by name."""
    return _owned(SEED_DIR if directory is None else directory, cara=True)


def seed_as_of(directory: Path | None = None) -> Time:
    """The time the seed is seen at: the latest ``CREATION_DATE`` of its
    files in ``directory`` (:data:`SEED_DIR` if ``None``).

    Raises
    ------
    ValueError
        If there is no seed file.
    CdmParseError
        If a file is not a CDM.
    """
    directory = SEED_DIR if directory is None else directory
    creations = [load(path).creation_date for path in seed_files(directory)]
    if not creations:
        raise ValueError(f"no seed CDM in {directory}; run {TOOL}")
    return max(creations, key=lambda time: time.tai.mjd)


def _seed_input(file: Path) -> CdmInput:
    """The CDM of a seed file, its origin the file's path from the
    repository's root (the same on every machine).

    Raises
    ------
    CdmParseError
        If the file is not a CDM.
    OSError
        If it cannot be read.
    """
    from keepout.store import CdmInput

    text = read_text(file)
    origin = (
        file.relative_to(REPO).as_posix() if file.is_relative_to(REPO) else str(file)
    )
    return CdmInput.parse(text, origin)


def seed_store(path: Path, *, cara: bool = False) -> Path:
    """A store of the seed in :data:`SEED_DIR` (:func:`seed_files`): its
    CDMs ingested and their Pc computed with :data:`SETTINGS`.

    The synthetic CDMs are stored with the snapshot's catalog, which they
    were screened from; CARA's, with ``cara``, without a catalog. Its CDM
    gives no ``HBR`` (only a comment does), so with the seed's settings it
    has no Pc. Each CDM's origin is its file's path from the repository's
    root.

    Parameters
    ----------
    path
        The store's file: made, or added to if it is a store.
    cara
        Add CARA's Metop-B CDM too, the ``.cdm`` file that ``--cara`` writes
        into :data:`SEED_DIR`.

    Returns
    -------
    pathlib.Path
        ``path``.

    Raises
    ------
    FileNotFoundError
        If ``cara`` and :data:`SEED_DIR` has no CARA file.
    StoreError
        If ``path`` cannot be made a store or holds other CDMs of the same
        ``MESSAGE_ID``.
    """
    from keepout.store import open_store

    files = seed_files(SEED_DIR)
    synthetic = [_seed_input(file) for file in files if file.suffix == SYNTHETIC]
    extra = (
        [_seed_input(file) for file in files if file.suffix == CARA_SUFFIX]
        if cara
        else []
    )
    if cara and not extra:
        raise FileNotFoundError(
            f"no CARA CDM in {SEED_DIR}; run keepout fetch cara, then {TOOL} --cara"
        )
    catalog = CatalogReference.of_file(snapshot.DATA / snapshot.SNAPSHOT)
    with open_store(path) as store:
        store.ingest(synthetic, catalog=catalog)
        if extra:
            store.ingest(extra)
        store.refresh(load_settings(SETTINGS))
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """The entry point writing (or with ``--check`` verifying) the seed.

    Returns
    -------
    int
        0 on success; 1 if, with ``--check``, a file is out of date or
        should not be there, or if ``--cara`` finds no cached CDM.
    """
    parser = argparse.ArgumentParser(prog=TOOL)
    parser.add_argument("--check", action="store_true", help="fail if out of date")
    parser.add_argument(
        "--cara",
        action="store_true",
        help="add CARA's Metop-B CDM from the cache (keepout fetch cara)",
    )
    parser.add_argument("--dir", type=Path, help="where to write the seed's CDMs")
    parser.add_argument(
        "--store",
        type=Path,
        metavar="PATH",
        help="build the demo's store at PATH instead of writing CDMs",
    )
    args = parser.parse_args(argv)
    if args.store is not None:
        given = [
            flag
            for flag, value in (
                ("--check", args.check),
                ("--cara", args.cara),
                ("--dir", args.dir is not None),
            )
            if value
        ]
        if given:
            parser.error(f"argument --store: not allowed with argument {given[0]}")
        demo_store(args.store)
        print(f"wrote the demo's store {args.store}")
        return 0
    try:
        cdms = seed_cdms(cara=args.cara)
    except SourceUnavailable as exc:
        print(f"{TOOL}: {exc}", file=sys.stderr)
        return 1
    if args.dir is not None:  # Elsewhere: the seed only, not the demo's update.
        return write_seed(cdms, args.dir, check=args.check)
    status = write_seed(cdms, SEED_DIR, check=args.check)
    return max(status, write_demo(check=args.check))


if __name__ == "__main__":
    sys.exit(main())
