"""NASA CARA Analysis Tools reference data: pinned download, cache and loading.

keepout never vendors CARA files (they are distributed under the NASA Open
Source Agreement). Files are downloaded on demand from a pinned commit,
verified by SHA-256 and cached under :func:`cache_dir`.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import scipy.io
from scipy.io.matlab import MatReadError

from keepout.errors import CaraFormatError, ChecksumMismatch, SourceUnavailable

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
            f"https://raw.githubusercontent.com/{CARA_REPO}/{CARA_COMMIT}/{self.path}"
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
MANIFEST: tuple[CaraFile, ...] = (
    MONTE_CARLO_OCMDB,
    FIND_NEARBY_CA,
    ALFANO_2009_RESULTS,
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
    except (OSError, http.client.HTTPException) as exc:
        raise SourceUnavailable(f"cannot download {url}: {exc}") from exc
    return body


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
    if target.is_file() and _sha256(target.read_bytes()) == file.sha256:
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
