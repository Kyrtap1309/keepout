"""NASA CARA Analysis Tools reference data: pinned download, cache and loading.

keepout never vendors CARA files (they are distributed under the NASA Open
Source Agreement). Files are downloaded on demand from a pinned commit,
verified by SHA-256 and cached under
:func:`~keepout.sources.cache.cache_dir`.
"""

from __future__ import annotations

import csv
import re
import urllib.parse
import zipfile
import zlib
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Final, TypeAlias

from keepout.errors import CaraFormatError
from keepout.sources.cache import (
    Downloader,
    cache_dir,
    has_digest,
    http_download,
    store,
)

__all__ = [
    "ALFANO_2009_RESULTS",
    "ALFANO_2009_SUPPLEMENT",
    "ALFANO_2009_TCA",
    "CARA_COMMIT",
    "CARA_REPO",
    "DILUTION_CDMS",
    "FRISBEE_CDM",
    "HALL_3D_CDM",
    "MANIFEST",
    "MANIFEST_FILE",
    "NON_PD_CDM",
    "PC_METHOD_CDMS",
    "PC_METHOD_TABLE",
    "PREFETCH_WORKERS",
    "CaraFile",
    "Cell",
    "Downloader",
    "commit_dir",
    "fetch",
    "load_csv",
    "load_xls_rows",
    "load_xlsx",
    "prefetch",
]

CARA_REPO: Final = "nasa/CARA_Analysis_Tools"
CARA_COMMIT: Final = "1c78beb1f7aefc384886c3be1d09e4681acb5d37"


@dataclass(frozen=True, slots=True)
class CaraFile:
    """A file in the CARA repository at :data:`CARA_COMMIT`.

    Parameters
    ----------
    path
        Path relative to the repository root.
    sha256
        Expected SHA-256 hex digest of the file contents.
    """

    path: str
    sha256: str

    @property
    def url(self) -> str:
        """Raw download URL pinned to :data:`CARA_COMMIT`."""
        return (
            f"https://raw.githubusercontent.com/{CARA_REPO}/{CARA_COMMIT}/"
            f"{urllib.parse.quote(self.path)}"
        )


MANIFEST_FILE: Final = "cara_manifest.sha256"
"""Every CARA file keepout uses, one ``<sha256>  <path>`` per line."""


_MANIFEST_LINE: Final = re.compile(r"(?P<digest>[0-9a-f]{64}) [ *](?P<path>\S.*)")
"""``sha256sum`` output: digest, space, text (space) or binary (``*``) mode, path."""


def _parse_manifest(text: str) -> dict[str, str]:
    """Path -> SHA-256 from ``sha256sum`` output; blank lines are skipped.

    Raises
    ------
    ValueError
        If a line is not ``sha256sum`` output, or a path is listed twice.
    """
    digests: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        match = _MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise ValueError(
                f"manifest line {number} is not sha256sum output: {line!r}"
            )
        path = match.group("path")
        if path in digests:
            raise ValueError(f"manifest line {number} repeats {path!r}")
        digests[path] = match.group("digest")
    return digests


_SHA256: Final[Mapping[str, str]] = _parse_manifest(
    resources.files("keepout.sources").joinpath(MANIFEST_FILE).read_text("utf-8")
)


def _file(path: str) -> CaraFile:
    return CaraFile(path, _SHA256[path])


MANIFEST: Final[tuple[CaraFile, ...]] = tuple(_file(path) for path in _SHA256)

ALFANO_2009_RESULTS: Final = _file(
    "DistributedMatlab/ProbabilityOfCollision/Pc3D_Hall_Utils/"
    "Output_AlfanoTestCases_Pc3D/A09TestCases_01_12_Distributed.csv"
)
ALFANO_2009_TCA: Final[tuple[CaraFile, ...]] = tuple(
    _file(f"DataFiles/AlfanoInputData/Alfano_2009_Test_Cases/Case{n} data at TCA.xls")
    for n in range(1, 13)
)
"""Alfano (2009) test cases 1-12: ECI states and covariances at TCA (.xls)."""

PC_METHOD_TABLE: Final = _file(
    "DataFiles/PcTestCaseCDMs/CARA_PcMethod_Test_Conjunctions.xlsx"
)
"""CARA Pc results for 53 real conjunctions (one row per CDM)."""

PC_METHOD_CDMS: Final[Mapping[str, CaraFile]] = {
    PurePosixPath(file.path).stem: file
    for file in MANIFEST
    if file.path.startswith("DataFiles/PcTestCaseCDMs/") and file.path.endswith(".cdm")
}
"""The 53 CDMs of :data:`PC_METHOD_TABLE`, keyed by conjunction ID."""

NON_PD_CDM: Final = _file(
    "DataFiles/SampleCDMs/OmitronTestCase_Test07_NonPDCovariance.cdm"
)
"""A CDM whose secondary covariance is not positive semi-definite."""

ALFANO_2009_SUPPLEMENT: Final = _file(
    "DistributedMatlab/ProbabilityOfCollision/Pc3D_Hall_Utils/"
    "Input_AlfanoTestCases_Pc3D/Alfano09_CARA_Supplementary_Data.xlsx"
)
"""Alfano (2009) time bounds ``A09_FinalTime_s`` (the 3-D and Monte Carlo
window is +- that, as in CARA's ``AlfanoTestCases_Pc3D.m``)."""

HALL_3D_CDM: Final = _file("DataFiles/SampleCDMs/OmitronTestCase_Test08_3DNc.cdm")
"""A CDM whose 3-D Pc exceeds its 2-D Pc by 15 orders of magnitude."""

FRISBEE_CDM: Final = _file("DataFiles/SampleCDMs/FrisbeeMaxPcTestCase_Test01.cdm")
"""The manufactured example of Frisbee (2015), used with one covariance."""

DILUTION_CDMS: Final[Mapping[str, CaraFile]] = {
    name: _file(f"DataFiles/SampleCDMs/OmitronTestCase_{name}.cdm")
    for name in (
        "Test01_HighPc",
        "Test02_MaxRadialSigma",
        "Test03_MaxIntrackSigma",
        "Test05_MinMiss",
    )
}
"""Omitron test CDMs of CARA's ``DilutionMaxPc_UnitTest.m``."""


def commit_dir(cache: Path | None = None) -> Path:
    """The folder of the cached files of :data:`CARA_COMMIT`.

    ``<cache>/cara/<CARA_COMMIT>``, ``<cache>`` being
    :func:`~keepout.sources.cache.cache_dir` of ``cache``; a file's copy is
    at its :attr:`CaraFile.path` below it.
    """
    return cache_dir(cache) / "cara" / CARA_COMMIT


def fetch(
    file: CaraFile,
    *,
    cache: Path | None = None,
    download: Downloader = http_download,
) -> Path:
    """A local, verified copy of ``file``, downloaded if needed.

    A cached copy is reused only if its SHA-256 matches; otherwise it is
    re-downloaded. The download is written atomically. A copy this process
    has already verified (or written) is not hashed again unless its inode,
    size or modification time changed.

    Raises
    ------
    SourceUnavailable
        If the download fails.
    ChecksumMismatch
        If the downloaded bytes do not match ``file.sha256``. Nothing is
        written to the cache in this case.
    """
    target = commit_dir(cache) / file.path
    if has_digest(target, file.sha256):
        return target

    data = download(file.url)
    store(target, lambda sink: sink.write(data), file.sha256, file.url)
    return target


PREFETCH_WORKERS: Final = 8
"""Concurrent downloads in :func:`prefetch`."""


def prefetch(
    files: Iterable[CaraFile],
    *,
    cache: Path | None = None,
    download: Downloader = http_download,
) -> None:
    """The download of every file not cached yet, concurrently.

    Raises
    ------
    SourceUnavailable, ChecksumMismatch
        As :func:`fetch`, for the first file that fails.
    """
    with ThreadPoolExecutor(max_workers=PREFETCH_WORKERS) as pool:
        # Consuming the results re-raises the first failure.
        list(pool.map(lambda file: fetch(file, cache=cache, download=download), files))


Cell: TypeAlias = object
"""A worksheet or table cell as the reader gives it (text, a number, a date,
``None`` or ``""`` when empty, ...): narrowed where it is used."""


def load_csv(path: Path) -> list[dict[str, str]]:
    """The rows of a CSV file with a header row, one dict per row.

    Raises
    ------
    CaraFormatError
        If the file has no header row.
    """
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise CaraFormatError(f"{path}: no header row")
        return list(reader)


def _blank(value: object) -> bool:
    """Whether a worksheet cell is empty."""
    return value in (None, "")


def load_xlsx(path: Path) -> list[dict[str, Cell]]:
    """The rows of the first sheet of an Excel 2007+ workbook, one dict per row,
    header in row 1.

    Empty cells after the last header name, and rows with no value, are
    ignored. Needs the ``validation`` extra (``openpyxl``).

    Raises
    ------
    CaraFormatError
        If the file is not a readable workbook, its header row is missing or
        has an empty or repeated name, or a row has values past the header.
    """
    import openpyxl  # optional dependency (validation extra)

    try:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile, zlib.error) as exc:
        raise CaraFormatError(f"{path}: not a readable .xlsx workbook ({exc})") from exc
    try:
        rows = list(workbook.worksheets[0].iter_rows(values_only=True))
    finally:
        workbook.close()
    if not rows or not any(rows[0]):
        raise CaraFormatError(f"{path}: no header row")
    cells = list(rows[0])
    while cells and _blank(cells[-1]):
        cells.pop()
    header = [str(name) for name in cells if not _blank(name)]
    if len(header) != len(cells) or len(set(header)) != len(header):
        raise CaraFormatError(f"{path}: header has empty or repeated names")
    records: list[dict[str, Cell]] = []
    for number, row in enumerate(rows[1:], start=2):
        if all(map(_blank, row)):
            continue
        values = list(row) + [None] * (len(header) - len(row))
        if not all(map(_blank, values[len(header) :])):
            raise CaraFormatError(f"{path}: row {number} has values past the header")
        records.append(dict(zip(header, values[: len(header)], strict=True)))
    return records


def load_xls_rows(path: Path) -> list[list[Cell]]:
    """The rows of the first sheet of a legacy Excel (.xls) workbook.

    Needs the ``validation`` extra (``xlrd``). Empty cells are ``""``.

    Raises
    ------
    CaraFormatError
        If the file is not a readable .xls workbook.
    """
    import xlrd  # optional dependency (validation extra)
    from xlrd.biffh import XLRDError

    try:
        sheet = xlrd.open_workbook(str(path)).sheet_by_index(0)
    except (OSError, XLRDError, ValueError) as exc:
        raise CaraFormatError(f"{path}: not a readable .xls workbook ({exc})") from exc
    return [list(sheet.row_values(index)) for index in range(sheet.nrows)]
