"""ESA's Collision Avoidance Challenge data set (Kelvins): download and loading.

The data set holds 199 082 anonymised CDMs that ESA's Space Debris Office
received from 2015 to 2019, in 15 321 conjunction events (Uriot et al.,
"Spacecraft collision avoidance challenge: design and results of a machine
learning competition", Astrodynamics 6, 2022). Each row is one CDM reduced to
103 numbers: the time to TCA, ESA's own log10 Pc (``risk``), the relative
state in RTN, both objects' RTN covariances as sigmas and correlations, orbit
and tracking quality. There are no epochs, object identities or absolute
states.

The data is © ESA, licensed under Creative Commons Attribution 4.0
(https://creativecommons.org/licenses/by/4.0/), published on Zenodo,
doi:10.5281/zenodo.4463683. keepout downloads it on demand from a pinned
record, verified by SHA-256, and caches only the raw table.
"""

from __future__ import annotations

import csv
import gzip
import itertools
import math
import shutil
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NamedTuple, TextIO, TypeAlias

import numpy as np

from keepout._types import FloatArray, read_only
from keepout._units import DAY
from keepout.errors import KelvinsFormatError
from keepout.risk.observation import PcObservation
from keepout.sources.cache import Sink, cache_dir, has_digest, http_stream, store

__all__ = [
    "ARCHIVE_SHA256",
    "ARCHIVE_URL",
    "COLUMNS",
    "DOI",
    "FEATURES",
    "RAW_MEMBER",
    "RAW_SHA256",
    "RISK_FLOOR",
    "SOURCE",
    "ZENODO_RECORD",
    "KelvinsEvent",
    "StreamDownloader",
    "fetch_raw",
    "load",
]

ZENODO_RECORD: Final = "4463683"
"""The Zenodo record of the data set (version of 2021-01-25)."""

DOI: Final = "10.5281/zenodo.4463683"
"""The data set's DOI, to cite it."""

ARCHIVE_URL: Final = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/"
    "Collision%20Avoidance%20Challenge%20-%20Dataset.zip/content"
)
"""The record's archive (221 MB; no login needed)."""

ARCHIVE_SHA256: Final = (
    "df1500146705305006ea506eeccc97f5c2e9593928d6650c503c4f5b5529d0bb"
)

RAW_MEMBER: Final = (
    "Collision Avoidance Challenge - Dataset/raw_data/raw_data_2015-2019.gz"
)
"""The raw table in the archive: every CDM of the competition's training and
test sets, with ``event_id`` unique across both (gzipped CSV, 118 MB)."""

RAW_SHA256: Final = "9c30d0bcc42a66818ee0958482694ca7fc1168682b02d3f1e26220198e89c06b"

RISK_FLOOR: Final = -30.0
"""ESA's floor of ``risk`` (log10 Pc): a CDM at the floor says only that the
Pc is at most 1e-30."""

SOURCE: Final = "ESA (Kelvins)"
"""The ``source`` of the observations: ESA's own Pc, method unpublished."""

_COLUMN_NAMES: Final = """
    event_id time_to_tca mission_id risk max_risk_estimate max_risk_scaling
    miss_distance relative_speed relative_position_r relative_position_t
    relative_position_n relative_velocity_r relative_velocity_t
    relative_velocity_n t_time_lastob_start t_time_lastob_end
    t_recommended_od_span t_actual_od_span t_obs_available t_obs_used
    t_residuals_accepted t_weighted_rms t_rcs_estimate t_cd_area_over_mass
    t_cr_area_over_mass t_sedr t_j2k_sma t_j2k_ecc t_j2k_inc t_ct_r t_cn_r
    t_cn_t t_crdot_r t_crdot_t t_crdot_n t_ctdot_r t_ctdot_t t_ctdot_n
    t_ctdot_rdot t_cndot_r t_cndot_t t_cndot_n t_cndot_rdot t_cndot_tdot
    c_object_type c_time_lastob_start c_time_lastob_end c_recommended_od_span
    c_actual_od_span c_obs_available c_obs_used c_residuals_accepted
    c_weighted_rms c_rcs_estimate c_cd_area_over_mass c_cr_area_over_mass
    c_sedr c_j2k_sma c_j2k_ecc c_j2k_inc c_ct_r c_cn_r c_cn_t c_crdot_r
    c_crdot_t c_crdot_n c_ctdot_r c_ctdot_t c_ctdot_n c_ctdot_rdot c_cndot_r
    c_cndot_t c_cndot_n c_cndot_rdot c_cndot_tdot t_span c_span t_h_apo t_h_per
    c_h_apo c_h_per geocentric_latitude azimuth elevation mahalanobis_distance
    t_position_covariance_det c_position_covariance_det t_sigma_r c_sigma_r
    t_sigma_t c_sigma_t t_sigma_n c_sigma_n t_sigma_rdot c_sigma_rdot
    t_sigma_tdot c_sigma_tdot t_sigma_ndot c_sigma_ndot F10 F3M SSN AP
"""

COLUMNS: Final = tuple(_COLUMN_NAMES.split())
"""The 103 columns of the raw table, in order (described in the archive's
``raw_data/raw_data_2015-2019.txt``; units as given there)."""

FEATURES: Final = tuple(
    name for name in COLUMNS if name not in ("event_id", "mission_id", "c_object_type")
)
"""The numeric columns kept per CDM in :attr:`KelvinsEvent.features`."""

_INDEX: Final = {name: index for index, name in enumerate(COLUMNS)}
_FEATURE_INDEX: Final = {name: index for index, name in enumerate(FEATURES)}
_FEATURE_COLUMNS: Final = tuple(_INDEX[name] for name in FEATURES)

StreamDownloader: TypeAlias = Callable[[str, Sink], None]
"""Downloads a URL into a sink, like :func:`keepout.sources.cache.http_stream`."""


def fetch_raw(
    *, cache: Path | None = None, download: StreamDownloader = http_stream
) -> Path:
    """Return a local, verified copy of the raw table, downloading it if needed.

    The archive is downloaded to the cache, verified, the raw table extracted
    and verified, and the archive deleted (also an archive left behind by an
    interrupted run, once the table is verified). The table is kept under
    ``<cache>/kelvins/<ZENODO_RECORD>/``.

    Parameters
    ----------
    cache
        Cache root; :func:`~keepout.sources.cache.cache_dir` if ``None``.
    download
        Downloads the archive.

    Raises
    ------
    SourceUnavailable
        If the download fails.
    ChecksumMismatch
        If the archive or the table does not have its pinned digest.
    KelvinsFormatError
        If the archive is not a zip file holding :data:`RAW_MEMBER`.
    """
    root = cache if cache is not None else cache_dir()
    folder = root / "kelvins" / ZENODO_RECORD
    target = folder / "raw_data_2015-2019.gz"
    archive = folder / "dataset.zip"
    if has_digest(target, RAW_SHA256):
        # An archive left by a run interrupted after storing the table.
        archive.unlink(missing_ok=True)
        return target
    if not has_digest(archive, ARCHIVE_SHA256):
        store(
            archive,
            lambda sink: download(ARCHIVE_URL, sink),
            ARCHIVE_SHA256,
            ARCHIVE_URL,
        )
    try:
        with zipfile.ZipFile(archive) as bundle, bundle.open(RAW_MEMBER) as member:
            store(
                target,
                lambda sink: shutil.copyfileobj(member, sink),
                RAW_SHA256,
                f"{archive}: {RAW_MEMBER}",
            )
    except (KeyError, zipfile.BadZipFile) as exc:
        raise KelvinsFormatError(f"{archive}: no {RAW_MEMBER} ({exc})") from exc
    archive.unlink(missing_ok=True)
    return target


@dataclass(frozen=True, eq=False)
class KelvinsEvent:
    """The CDMs of one event of the data set, in the table's order.

    Parameters
    ----------
    event_id
        The event's number in the raw table.
    mission_id
        ESA's anonymised mission number.
    features
        One row per CDM, one column per name in :data:`FEATURES`; ``NaN``
        where the table is empty. Stored read-only.
    object_types
        ``c_object_type`` of each CDM (``DEBRIS``, ``PAYLOAD``, ...).

    Raises
    ------
    ValueError
        If ``features`` is not a 2-D array with one column per feature and
        one row per object type, or there are no CDMs.
    """

    event_id: int
    mission_id: int
    features: FloatArray
    object_types: tuple[str, ...]

    def __post_init__(self) -> None:
        features = np.array(self.features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(FEATURES):
            raise ValueError(
                f"features must have shape (n, {len(FEATURES)}), got {features.shape}"
            )
        if features.shape[0] == 0 or features.shape[0] != len(self.object_types):
            raise ValueError(
                f"{features.shape[0]} feature rows for "
                f"{len(self.object_types)} object types; need at least one CDM"
            )
        read_only(features)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "object_types", tuple(self.object_types))

    def __len__(self) -> int:
        """Number of CDMs."""
        return len(self.object_types)

    def column(self, name: str) -> FloatArray:
        """One feature of every CDM (read-only).

        Raises
        ------
        KeyError
            If ``name`` is not in :data:`FEATURES`.
        """
        if name not in _FEATURE_INDEX:
            raise KeyError(f"{name!r} is not a Kelvins feature")
        return self.features[:, _FEATURE_INDEX[name]]

    @property
    def time_to_tca(self) -> FloatArray:
        """Seconds from each CDM to TCA (the table gives days)."""
        return self.column("time_to_tca") * DAY

    def observations(self) -> tuple[PcObservation, ...]:
        """ESA's Pc of each CDM, as reported (``risk`` is log10 Pc).

        A CDM at :data:`RISK_FLOOR` gives a ``"below_floor"`` observation of
        1e-30. keepout does not recompute the Pc: the table lacks absolute
        states, and ESA's method is not published.
        """
        observations: list[PcObservation] = []
        for index, (seconds, risk) in enumerate(
            zip(self.time_to_tca, self.column("risk"), strict=True)
        ):
            floored = risk <= RISK_FLOOR
            observations.append(
                PcObservation(
                    time_to_tca=float(seconds),
                    source=SOURCE,
                    pc=10.0**RISK_FLOOR if floored else float(10.0**risk),
                    kind="below_floor" if floored else "estimate",
                    message_id=f"{self.event_id}/{index}",
                )
            )
        return tuple(observations)


def _number(text: str, where: str, name: str) -> float:
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        raise KelvinsFormatError(f"{where}: {name} is not a number: {text!r}") from None


def _integer(text: str, where: str, name: str) -> int:
    value = _number(text, where, name)
    if not (math.isfinite(value) and value == int(value)):
        raise KelvinsFormatError(f"{where}: {name} must be an integer, got {text!r}")
    return int(value)


def _open(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open(encoding="utf-8", newline="")


def _rows(path: Path) -> Iterator[tuple[int, list[str]]]:
    with _open(path) as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None or tuple(header) != COLUMNS:
            missing = sorted(set(COLUMNS) - set(header or ()))
            extra = sorted(set(header or ()) - set(COLUMNS))
            raise KelvinsFormatError(
                f"{path}: header is not the Kelvins raw table's 103 columns in "
                f"order (missing {missing}, unexpected {extra})"
            )
        for line, row in enumerate(reader, start=2):
            if len(row) != len(COLUMNS):
                raise KelvinsFormatError(
                    f"{path}, line {line}: {len(row)} values, expected {len(COLUMNS)}"
                )
            yield line, row


class _Row(NamedTuple):
    """One validated CDM row of the raw table."""

    where: str
    event_id: int
    mission_id: int
    values: list[float]
    object_type: str


def _parsed(source: Path) -> Iterator[_Row]:
    for line, row in _rows(source):
        where = f"{source}, line {line}"
        event_id = _integer(row[_INDEX["event_id"]], where, "event_id")
        mission_id = _integer(row[_INDEX["mission_id"]], where, "mission_id")
        values = [_number(row[i], where, COLUMNS[i]) for i in _FEATURE_COLUMNS]
        for name in ("time_to_tca", "risk"):
            if not math.isfinite(values[_FEATURE_INDEX[name]]):
                raise KelvinsFormatError(f"{where}: {name} is missing")
        if values[_FEATURE_INDEX["risk"]] > 0.0:
            raise KelvinsFormatError(f"{where}: risk (log10 Pc) is above 0")
        yield _Row(where, event_id, mission_id, values, row[_INDEX["c_object_type"]])


def load(path: Path | None = None) -> tuple[KelvinsEvent, ...]:
    """Read the raw table into events.

    Parameters
    ----------
    path
        The raw table (gzipped if it ends in ``.gz``, else plain CSV) or an
        excerpt with the same header; :func:`fetch_raw` if ``None``.

    Returns
    -------
    tuple[KelvinsEvent, ...]
        The events in the table's order.

    Raises
    ------
    KelvinsFormatError
        If the header is not :data:`COLUMNS`, a row has the wrong length, a
        value is not a number, ``event_id`` or ``mission_id`` is not an
        integer, ``time_to_tca`` or ``risk`` is missing, ``risk`` is above 0,
        an event's rows are not contiguous, or its mission changes.
    """
    source = path if path is not None else fetch_raw()
    events: list[KelvinsEvent] = []
    seen: set[int] = set()
    for event_id, group in itertools.groupby(
        _parsed(source), key=lambda row: row.event_id
    ):
        rows: list[_Row] = []
        for row in group:
            if not rows and event_id in seen:
                raise KelvinsFormatError(
                    f"{row.where}: rows of event {event_id} are not contiguous"
                )
            if rows and row.mission_id != rows[0].mission_id:
                raise KelvinsFormatError(
                    f"{row.where}: event {event_id} changes mission from "
                    f"{rows[0].mission_id} to {row.mission_id}"
                )
            rows.append(row)
        seen.add(event_id)
        events.append(
            KelvinsEvent(
                event_id,
                rows[0].mission_id,
                np.array([row.values for row in rows], dtype=np.float64),
                tuple(row.object_type for row in rows),
            )
        )
    return tuple(events)
