"""The record of when each Space-Track request class was last queried, shared
by every process using one cache, which read, check and write it under an
advisory file lock (:meth:`RequestLedger.locked`)."""

from __future__ import annotations

import contextlib
import json
import math
import sys
from typing import TYPE_CHECKING, Final

from keepout.errors import SourceFormatError
from keepout.sources._store import store_json
from keepout.sources.spacetrack._query import CLASS_INTERVALS
from keepout.units import HOUR

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from pathlib import Path

    from keepout._types import JsonValue

__all__ = ["LEDGER_FILE", "LOCK_SUFFIX", "RequestLedger"]

LEDGER_FILE: Final = "requests.json"
"""The file, in the Space-Track cache, recording when each request class was
last queried."""

LOCK_SUFFIX: Final = ".lock"
"""What the name of the ledger's lock file adds to the ledger's."""

_LEDGER_FORMAT: Final = 1
"""Version of the ledger layout; another version is an error."""


class RequestLedger:
    """The file recording when each request class was last queried.

    Parameters
    ----------
    path
        The file; it need not exist yet.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    @contextlib.contextmanager
    def locked(self) -> Generator[None]:
        """The ledger to this process alone while in the context: another
        process (or another :meth:`locked` of this one) waits.

        The lock is advisory, on a file beside the ledger
        (:data:`LOCK_SUFFIX`); the directory is created if missing.

        Raises
        ------
        SourceFormatError
            If the lock file cannot be opened or locked.
        """
        lock_path = self.path.with_name(self.path.name + LOCK_SUFFIX)
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock = lock_path.open("a+b")
        except OSError as exc:
            raise SourceFormatError(
                f"cannot lock the Space-Track request record {self.path}: {exc}"
            ) from exc
        with lock:
            _lock(lock.fileno())
            try:
                yield
            finally:
                _unlock(lock.fileno())

    def read(self) -> dict[str, float]:
        """The Unix time [s] each request class was last queried at (empty if
        the file does not exist).

        Raises
        ------
        SourceFormatError
            If the file cannot be read or is not a record keepout wrote.
        """
        try:
            record: JsonValue = json.loads(self.path.read_bytes())
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            raise SourceFormatError(
                f"cannot read the Space-Track request record {self.path}: {exc}"
            ) from exc
        invalid = (
            f"{self.path} is not a Space-Track request record of this "
            "keepout version; delete it only if no query was sent in the "
            f"last {max(CLASS_INTERVALS.values()) / HOUR:g} hours"
        )
        if not isinstance(record, dict) or record.get("format") != _LEDGER_FORMAT:
            raise SourceFormatError(invalid)
        queried = record.get("queried")
        if not isinstance(queried, dict):
            raise SourceFormatError(invalid)
        sent: dict[str, float] = {}
        for query, sent_at in queried.items():
            if not isinstance(sent_at, float) or not math.isfinite(sent_at):
                raise SourceFormatError(invalid)
            sent[query] = sent_at
        return sent

    def write(self, queried: Mapping[str, float]) -> None:
        """Replace the record by ``queried``: request class to Unix time [s]."""
        record = {"format": _LEDGER_FORMAT, "queried": dict(sorted(queried.items()))}
        store_json(self.path, record)


if sys.platform == "win32":
    import msvcrt

    def _lock(descriptor: int) -> None:
        # LK_LOCK gives up after 10 s; a reservation holds the lock for
        # milliseconds, so try again until it is free.
        while True:
            try:
                msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
            except OSError:
                continue
            return

    def _unlock(descriptor: int) -> None:
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(descriptor: int) -> None:
        fcntl.flock(descriptor, fcntl.LOCK_EX)

    def _unlock(descriptor: int) -> None:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
