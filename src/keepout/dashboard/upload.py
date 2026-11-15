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
from typing import TYPE_CHECKING, Final

import keepout.cdm
from keepout.errors import CdmParseError, KeepoutError
from keepout.store import CdmInput, IngestReport, JobQueue

if TYPE_CHECKING:
    from collections.abc import Sequence

    from keepout.settings import Settings
    from keepout.store import Store

__all__ = [
    "MAX_FILES",
    "MAX_FILE_BYTES",
    "ORIGIN_PREFIX",
    "UploadResult",
    "ingest_upload",
]

MAX_FILE_BYTES: Final = 1_000_000
"""Most bytes an uploaded file holds, and most its data holds once
decompressed if it is gzip data (:func:`keepout.cdm.decode_text`)."""

MAX_FILES: Final = 50
"""Most files one upload holds."""

ORIGIN_PREFIX: Final = "upload:"
"""What the origin of an uploaded CDM starts with, before the file's name."""

_TRAILING: Final = ",;:"
"""What may follow a word of a store error's message."""


@dataclass(frozen=True, slots=True)
class UploadResult:
    """What an upload did.

    Parameters
    ----------
    report
        What the ingest stored; ``None`` if nothing was ingested.
    errors
        Each error as (file name, message), the message not repeating the
        name; the name is empty when the error is not of one file (too many
        files, a store error naming none of the uploaded ``MESSAGE_ID``, a
        job that could not be queued).
    jobs
        The id of each ``refresh`` job queued, or already queued or
        running, for the events the upload touched.
    """

    report: IngestReport | None
    errors: tuple[tuple[str, str], ...]
    jobs: tuple[int, ...]


def _parsed(
    files: Sequence[tuple[str, bytes]],
) -> tuple[list[CdmInput], list[tuple[str, str]]]:
    """The CDM of each file, and the error of each that has none (its
    message without the file's name, which the error is paired with)."""
    messages: list[CdmInput] = []
    errors: list[tuple[str, str]] = []
    for name, data in files:
        try:
            text = keepout.cdm.decode_text(data, name, max_bytes=MAX_FILE_BYTES)
            cdm = keepout.cdm.loads(text)
        except CdmParseError as exc:
            errors.append((name, str(exc).removeprefix(f"{name}: ")))
            continue
        format_ = keepout.cdm.format_of(text)
        messages.append(CdmInput(cdm, text, format_, ORIGIN_PREFIX + name))
    return messages, errors


def _named(error: KeepoutError, messages: Sequence[CdmInput]) -> tuple[str, str]:
    """A store error as (file name, message): the first file whose
    ``MESSAGE_ID`` is a word of the message, else no file."""
    text = str(error)
    words = {word.rstrip(_TRAILING) for word in text.split()}
    for message in messages:
        if message.cdm.message_id in words:
            return message.origin.removeprefix(ORIGIN_PREFIX), text
    return "", text


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
        return UploadResult(None, (("", message),), ())
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
            errors.append(("", f"event {event_id}: {exc}"))
            continue
        if submission.job is not None:
            jobs.append(submission.job.id)
    return UploadResult(report, tuple(errors), tuple(jobs))
