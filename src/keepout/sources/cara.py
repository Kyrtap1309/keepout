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
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import scipy.io
from scipy.io.matlab import MatReadError

from keepout.errors import CaraFormatError, ChecksumMismatch, SourceUnavailable
from keepout.sources._cara_pc_cdms import PC_METHOD_CDM_SHA256

CARA_REPO = "nasa/CARA_Analysis_Tools"
CARA_COMMIT = "1c78beb1f7aefc384886c3be1d09e4681acb5d37"

Downloader = Callable[[str], bytes]


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


MONTE_CARLO_OCMDB = CaraFile(
    "DistributedMatlab/MonteCarloPc/UnitTests/InputFiles/OCMDB_TestData.mat",
    "d737ca5b060402b6d6dee184ca020cffc064440fba17e9d5c47ca6ddeda556b4",
)
FIND_NEARBY_CA = CaraFile(
    "DistributedMatlab/ProbabilityOfCollision/Utils/UnitTests/InputFiles/"
    "FindNearbyCA_TestCases.mat",
    "f8bd97924efff942abf4b10552739f685224da25db83f359f4b40d7a134e1b79",
)
ALFANO_2009_RESULTS = CaraFile(
    "DistributedMatlab/ProbabilityOfCollision/Pc3D_Hall_Utils/"
    "Output_AlfanoTestCases_Pc3D/A09TestCases_01_12_Distributed.csv",
    "b7680d2f8463ec267b3577cb80895ee2d935e016a1b5def652bb3807f19185a9",
)
ALFANO_2009_TCA: tuple[CaraFile, ...] = tuple(
    CaraFile(
        "DataFiles/AlfanoInputData/Alfano_2009_Test_Cases/"
        f"Case{number} data at TCA.xls",
        sha256,
    )
    for number, sha256 in enumerate(
        (
            "ada05407fe729cc5766840f0565cdec7972c9c6e21bc09cebc379c498d1d5f85",
            "37c315bf90fa999c90e54f19df93c37eeac0996331e0335b43daee9b2ec0f024",
            "655c0182eb945142fd99d729044f7fca8d14ad5d49fb229317020496bb57264a",
            "323c48cbd9b3034a1f328b8de0106924c0cdf9948504124498fd81c7126e2f91",
            "876ff1396108b660ba1c769db4390e01d0f1e84058a8ae8743ec9ab1be52b0a8",
            "d4a037ed819ba19ca38b00b4a7d82a92a243c1315549fe9a00b4f6263c78ae3c",
            "817de09c9bf04e70eacc9069832ebd0d532595aa5dee0b377b2b65fc45ee2c2c",
            "695e3b5a1c44b5885efaa90be46f8d851596c0646dd95623ba65de9c092616e3",
            "f7a8f6429b1c1f24df1a7abd6d8841551a8b2f7b1407fdc860bd0edae0202633",
            "47b75745f9a679acf120ba13c408e9c5e0d85e26a802762c7f0321380de117ef",
            "2c8c4aec2b091ca22bea81bb221abdf86bed652f7910644fcd8139f488ce191e",
            "c85af92e1dbad8830b059089a9d29bb0d902b1f44c2de4647993e239034ffdce",
        ),
        start=1,
    )
)
"""Alfano (2009) test cases 1-12: ECI states and covariances at TCA (.xls)."""

PC_METHOD_TABLE = CaraFile(
    "DataFiles/PcTestCaseCDMs/CARA_PcMethod_Test_Conjunctions.xlsx",
    "dcb1f5bf51703ee793582d21b1f669dc388d9eb0b9f4b3871f13055cc885eb85",
)
"""CARA Pc results for 53 real conjunctions (one row per CDM)."""

PC_METHOD_CDMS: Mapping[str, CaraFile] = {
    conjunction_id: CaraFile(f"DataFiles/PcTestCaseCDMs/{conjunction_id}.cdm", sha256)
    for conjunction_id, sha256 in PC_METHOD_CDM_SHA256.items()
}
"""The 53 CDMs of :data:`PC_METHOD_TABLE`, keyed by conjunction ID."""

NON_PD_CDM = CaraFile(
    "DataFiles/SampleCDMs/OmitronTestCase_Test07_NonPDCovariance.cdm",
    "ddb62be74ceaa06f385f1e36b0704ac9b6d3c69ddf7b2acbaf489e2d9b77a4bf",
)
"""A CDM whose secondary covariance is not positive semi-definite."""

MANIFEST: tuple[CaraFile, ...] = (
    MONTE_CARLO_OCMDB,
    FIND_NEARBY_CA,
    ALFANO_2009_RESULTS,
    *ALFANO_2009_TCA,
    PC_METHOD_TABLE,
    *PC_METHOD_CDMS.values(),
    NON_PD_CDM,
)


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
        with urllib.request.urlopen(url, timeout=60) as response:
            body: bytes = response.read()
            return body
    except (OSError, http.client.HTTPException) as exc:
        raise SourceUnavailable(f"cannot download {url}: {exc}") from exc


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    actual = _sha256(data)
    if actual != file.sha256:
        raise ChecksumMismatch(
            f"{file.url}: expected sha256 {file.sha256}, got {actual}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(data)
    partial.replace(target)
    return target


def load_mat(path: Path) -> dict[str, Any]:
    """Load a MATLAB v5 MAT-file into a dict of variables.

    Metadata keys (``__header__``, ``__version__``, ``__globals__``) are
    removed. MATLAB structs are returned as objects with attribute access.

    Raises
    ------
    CaraFormatError
        If the file is not a readable MATLAB v5 file (including v7.3/HDF5).
    OSError
        If ``path`` cannot be opened (e.g. ``FileNotFoundError``,
        ``PermissionError``).
    """
    with path.open("rb") as stream:
        try:
            raw = scipy.io.loadmat(stream, squeeze_me=True, struct_as_record=False)
        except NotImplementedError as exc:
            raise CaraFormatError(
                f"{path}: MATLAB v7.3 (HDF5) files are not supported"
            ) from exc
        except (ValueError, MatReadError, OSError, zlib.error) as exc:
            raise CaraFormatError(f"{path}: not a readable MAT-file ({exc})") from exc
    return {key: value for key, value in raw.items() if not key.startswith("__")}


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

    Needs the ``validation`` extra (``openpyxl``).

    Raises
    ------
    CaraFormatError
        If the file is not a readable workbook or has no header row.
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
    header = [str(name) for name in rows[0]]
    return [dict(zip(header, row, strict=True)) for row in rows[1:] if any(row)]


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
