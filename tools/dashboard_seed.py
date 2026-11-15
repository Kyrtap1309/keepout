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
from keepout.cdm import kvn, load, read_text
from keepout.errors import SourceUnavailable
from keepout.screening import DEFAULT_DURATION, DEFAULT_VOLUME
from keepout.settings import load_settings
from keepout.sources import cara
from keepout.sources.catalog import CatalogReference
from keepout.synthetic import load_fleet
from keepout.synthetic.operator import run
from keepout.units import DAY, elapsed_seconds

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    parser.add_argument("--dir", type=Path, default=SEED_DIR, help="where to write")
    args = parser.parse_args(argv)
    try:
        cdms = seed_cdms(cara=args.cara)
    except SourceUnavailable as exc:
        print(f"{TOOL}: {exc}", file=sys.stderr)
        return 1
    return write_seed(cdms, args.dir, check=args.check)


if __name__ == "__main__":
    sys.exit(main())
