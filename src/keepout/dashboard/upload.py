"""CDM files uploaded on the dashboard: ingested all or none, then a
``refresh`` job per event they touched.

Every file is read and parsed before anything is stored: a file over
:data:`MAX_FILE_BYTES` (or gzip data that decompresses to more), more than
:data:`MAX_FILES` files, a file that is not UTF-8 text or not a CDM, and a
``MESSAGE_ID`` stored with other content each stop the whole upload, with an
error naming the file. Otherwise the CDMs are stored in one
:meth:`~keepout.store.Store.ingest` and the Pc their events lack is queued
for the worker (:meth:`~keepout.store.JobQueue.submit_refresh`): the
dashboard computes none.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, NamedTuple

import keepout.cdm
from keepout._environment import MAX_UPLOAD_MB
from keepout.errors import CdmParseError, KeepoutError, StoreError
from keepout.store import CdmInput, IngestReport, JobQueue

if TYPE_CHECKING:
    from collections.abc import Sequence

    from keepout.settings import Settings
    from keepout.store import Store

__all__ = [
    "MAX_FILES",
    "MAX_FILE_BYTES",
    "ORIGIN_PREFIX",
    "UploadError",
    "UploadResult",
    "ingest_upload",
]

MAX_FILE_BYTES: Final = MAX_UPLOAD_MB * 1024 * 1024
"""Most bytes an uploaded file holds, and most its data holds once
decompressed if it is gzip data (:func:`keepout.cdm.decode_text`):
:data:`keepout._environment.MAX_UPLOAD_MB` in Streamlit's MB of 2^20
bytes, the limit its uploader enforces too, checked again here for gzip
data."""

MAX_FILES: Final = 50
"""Most files one upload holds."""

ORIGIN_PREFIX: Final = "upload:"
"""What the origin of an uploaded CDM starts with, before the file's name."""


class UploadError(NamedTuple):
    """An error of an upload.

    Parameters
    ----------
    file
        The name of the file it is of; ``None`` when it is not of one file
        (too many files, a store error of no uploaded CDM, a job that could
        not be queued).
    message
        The error, not repeating the file's name.
    """

    file: str | None
    message: str


@dataclass(frozen=True, slots=True)
class UploadResult:
    """What an upload did.

    Parameters
    ----------
    report
        What the ingest stored; ``None`` if nothing was ingested.
    errors
        Each error, naming its file if it is of one.
    jobs
        The id of each ``refresh`` job queued, or already queued or
        running, for the events the upload touched.
    """

    report: IngestReport | None
    errors: tuple[UploadError, ...]
    jobs: tuple[int, ...]


def _parsed(
    files: Sequence[tuple[str, bytes]],
) -> tuple[list[CdmInput], list[UploadError]]:
    """The CDM of each file, and the error of each that has none."""
    messages: list[CdmInput] = []
    errors: list[UploadError] = []
    for name, data in files:
        try:
            text = keepout.cdm.decode_text(data, name, max_bytes=MAX_FILE_BYTES)
            message = CdmInput.parse(text, ORIGIN_PREFIX + name)
        except CdmParseError as exc:
            errors.append(UploadError(name, str(exc).removeprefix(f"{name}: ")))
            continue
        messages.append(message)
    return messages, errors


def _named(error: KeepoutError, messages: Sequence[CdmInput]) -> UploadError:
    """An ingest error, of the first file of the CDM it names
    (:attr:`StoreError.message_id <keepout.errors.StoreError.message_id>`),
    else of no file."""
    text = str(error)
    message_id = error.message_id if isinstance(error, StoreError) else None
    for message in messages:
        if message_id is not None and message.cdm.message_id == message_id:
            return UploadError(message.origin.removeprefix(ORIGIN_PREFIX), text)
    return UploadError(None, text)


def ingest_upload(
    store: Store, settings: Settings, files: Sequence[tuple[str, bytes]]
) -> UploadResult:
    """Store uploaded CDM files and queue the Pc their events lack.

    Every file is parsed first; on any error nothing is stored. Otherwise
    the CDMs are ingested in one transaction (all or none), then a
    ``refresh`` job is submitted for each event they belong to: none for an
    event whose Pc are all cached, the same job for one already queued or
    running.

    Parameters
    ----------
    store
        The store to ingest into.
    settings
        The settings the Pc are computed with.
    files
        Each file as (name, bytes): KVN or XML, UTF-8, gzip-compressed or
        not; at most :data:`MAX_FILES`, each at most :data:`MAX_FILE_BYTES`.

    Returns
    -------
    UploadResult
        What was ingested and queued, and every error, each naming its
        file. Errors are returned, never raised: the page shows them.
    """
    if len(files) > MAX_FILES:
        message = f"at most {MAX_FILES} files per upload, not {len(files)}"
        return UploadResult(None, (UploadError(None, message),), ())
    messages, errors = _parsed(files)
    if errors:
        return UploadResult(None, tuple(errors), ())
    if not messages:
        return UploadResult(None, (), ())
    try:
        report = store.ingest(messages)
    except KeepoutError as exc:
        return UploadResult(None, (_named(exc, messages),), ())
    queue = JobQueue(store)
    jobs: list[int] = []
    for event_id in report.events:
        try:
            submission = queue.submit_refresh(event_id, settings)
        except (KeepoutError, ValueError) as exc:
            errors.append(UploadError(None, f"event {event_id}: {exc}"))
            continue
        if submission.job is not None:
            jobs.append(submission.job.id)
    return UploadResult(report, tuple(errors), tuple(jobs))
