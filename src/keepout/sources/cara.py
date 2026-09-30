"""NASA CARA Analysis Tools reference data: pinned download, cache and loading.

keepout never vendors CARA files (they are distributed under the NASA Open
Source Agreement). Files are downloaded on demand from a pinned commit,
verified by SHA-256 and cached under :func:`cache_dir`.
"""

from __future__ import annotations

import csv
import hashlib
import http.client
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any, TypeAlias

from keepout.errors import CaraFormatError, ChecksumMismatch, SourceUnavailable

CARA_REPO = "nasa/CARA_Analysis_Tools"
CARA_COMMIT = "1c78beb1f7aefc384886c3be1d09e4681acb5d37"

DOWNLOAD_TIMEOUT = 60.0
"""Seconds :func:`http_download` waits for the server."""

Downloader: TypeAlias = Callable[[str], bytes]


@dataclass(frozen=True)
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


MANIFEST_FILE = "cara_manifest.sha256"
"""Every CARA file keepout uses, one ``<sha256>  <path>`` per line."""


_MANIFEST_LINE = re.compile(r"(?P<digest>[0-9a-f]{64}) [ *](?P<path>\S.*)")
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


_SHA256 = _parse_manifest(
    resources.files(__package__).joinpath(MANIFEST_FILE).read_text("utf-8")
)


def _file(path: str) -> CaraFile:
    return CaraFile(path, _SHA256[path])


MANIFEST: tuple[CaraFile, ...] = tuple(_file(path) for path in _SHA256)

ALFANO_2009_RESULTS = _file(
    "DistributedMatlab/ProbabilityOfCollision/Pc3D_Hall_Utils/"
    "Output_AlfanoTestCases_Pc3D/A09TestCases_01_12_Distributed.csv"
)
ALFANO_2009_TCA: tuple[CaraFile, ...] = tuple(
    _file(f"DataFiles/AlfanoInputData/Alfano_2009_Test_Cases/Case{n} data at TCA.xls")
    for n in range(1, 13)
)
"""Alfano (2009) test cases 1-12: ECI states and covariances at TCA (.xls)."""

PC_METHOD_TABLE = _file("DataFiles/PcTestCaseCDMs/CARA_PcMethod_Test_Conjunctions.xlsx")
"""CARA Pc results for 53 real conjunctions (one row per CDM)."""

PC_METHOD_CDMS: Mapping[str, CaraFile] = {
    PurePosixPath(file.path).stem: file
    for file in MANIFEST
    if file.path.startswith("DataFiles/PcTestCaseCDMs/") and file.path.endswith(".cdm")
}
"""The 53 CDMs of :data:`PC_METHOD_TABLE`, keyed by conjunction ID."""

NON_PD_CDM = _file("DataFiles/SampleCDMs/OmitronTestCase_Test07_NonPDCovariance.cdm")
"""A CDM whose secondary covariance is not positive semi-definite."""

ALFANO_2009_SUPPLEMENT = _file(
    "DistributedMatlab/ProbabilityOfCollision/Pc3D_Hall_Utils/"
    "Input_AlfanoTestCases_Pc3D/Alfano09_CARA_Supplementary_Data.xlsx"
)
"""Alfano (2009) time bounds ``A09_FinalTime_s`` (the 3-D and Monte Carlo
window is +- that, as in CARA's ``AlfanoTestCases_Pc3D.m``)."""

HALL_3D_CDM = _file("DataFiles/SampleCDMs/OmitronTestCase_Test08_3DNc.cdm")
"""A CDM whose 3-D Pc exceeds its 2-D Pc by 15 orders of magnitude."""

FRISBEE_CDM = _file("DataFiles/SampleCDMs/FrisbeeMaxPcTestCase_Test01.cdm")
"""The manufactured example of Frisbee (2015), used with one covariance."""

DILUTION_CDMS: Mapping[str, CaraFile] = {
    name: _file(f"DataFiles/SampleCDMs/OmitronTestCase_{name}.cdm")
    for name in (
        "Test01_HighPc",
        "Test02_MaxRadialSigma",
        "Test03_MaxIntrackSigma",
        "Test05_MinMiss",
    )
}
"""Omitron test CDMs of CARA's ``DilutionMaxPc_UnitTest.m``."""


def cache_dir() -> Path:
    """Root cache directory for downloaded reference data.

    ``$KEEPOUT_CACHE_DIR`` if set, else ``$XDG_CACHE_HOME/keepout``, else
    ``~/.cache/keepout``. The directory is not created here.
    """
    override = os.environ.get("KEEPOUT_CACHE_DIR")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "keepout"


def http_download(url: str) -> bytes:
    """Download ``url`` and return the body.

    Raises
    ------
    SourceUnavailable
        On any network or HTTP error.
    """
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response:
            body: bytes = response.read()
            return body
    except (OSError, http.client.HTTPException) as exc:
        raise SourceUnavailable(f"cannot download {url}: {exc}") from exc


def fetch(
    file: CaraFile,
    *,
    cache: Path | None = None,
    download: Downloader = http_download,
) -> Path:
    """Return a local, verified copy of ``file``, downloading it if needed.

    A cached copy is reused only if its SHA-256 matches; otherwise it is
    re-downloaded. The download is written atomically.

    Raises
    ------
    SourceUnavailable
        If the download fails.
    ChecksumMismatch
        If the downloaded bytes do not match ``file.sha256``. Nothing is
        written to the cache in this case.
    """
    root = cache if cache is not None else cache_dir()
    target = root / "cara" / CARA_COMMIT / file.path
    if target.is_file():
        with target.open("rb") as cached:
            if hashlib.file_digest(cached, "sha256").hexdigest() == file.sha256:
                return target

    data = download(file.url)
    actual = hashlib.sha256(data).hexdigest()
    if actual != file.sha256:
        raise ChecksumMismatch(
            f"{file.url}: expected sha256 {file.sha256}, got {actual}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    # A unique temporary name, so concurrent fetches never share a partial file.
    descriptor, name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".part"
    )
    partial = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return target


PREFETCH_WORKERS = 8
"""Concurrent downloads in :func:`prefetch`."""


def prefetch(
    files: Iterable[CaraFile],
    *,
    cache: Path | None = None,
    download: Downloader = http_download,
) -> None:
    """Make sure every file is cached, downloading missing ones concurrently.

    Raises
    ------
    SourceUnavailable, ChecksumMismatch
        As :func:`fetch`, for the first file that fails.
    """
    with ThreadPoolExecutor(max_workers=PREFETCH_WORKERS) as pool:
        # Consuming the results re-raises the first failure.
        list(pool.map(lambda file: fetch(file, cache=cache, download=download), files))


def load_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file with a header row into one dict per row.

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


def load_xlsx(path: Path) -> list[dict[str, Any]]:
    """Read the first sheet of an Excel 2007+ workbook, header in row 1.

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
    while cells and cells[-1] in (None, ""):
        cells.pop()
    header = [str(name) for name in cells if name not in (None, "")]
    if len(header) != len(cells) or len(set(header)) != len(header):
        raise CaraFormatError(f"{path}: header has empty or repeated names")
    records: list[dict[str, Any]] = []
    for number, row in enumerate(rows[1:], start=2):
        if not any(value not in (None, "") for value in row):
            continue
        values = list(row) + [None] * (len(header) - len(row))
        if any(value not in (None, "") for value in values[len(header) :]):
            raise CaraFormatError(f"{path}: row {number} has values past the header")
        records.append(dict(zip(header, values[: len(header)], strict=True)))
    return records


def load_xls_rows(path: Path) -> list[list[Any]]:
    """Read the first sheet of a legacy Excel (.xls) workbook as rows.

    Needs the ``validation`` extra (``xlrd``). Empty cells are ``""``.

    Raises
    ------
    CaraFormatError
        If the file is not a readable .xls workbook.
    """
    import xlrd  # optional dependency (validation extra)

    try:
        sheet = xlrd.open_workbook(str(path)).sheet_by_index(0)
    except (OSError, xlrd.XLRDError, ValueError) as exc:
        raise CaraFormatError(f"{path}: not a readable .xls workbook ({exc})") from exc
    return [list(sheet.row_values(index)) for index in range(sheet.nrows)]
