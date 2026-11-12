"""The record of when each Space-Track request class was last queried, shared
by every process using one cache."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Final

from keepout.errors import SourceFormatError
from keepout.sources._store import store_json
from keepout.sources.spacetrack._query import CLASS_INTERVALS
from keepout.units import HOUR

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from keepout._types import JsonValue

__all__ = ["LEDGER_FILE", "RequestLedger"]

LEDGER_FILE: Final = "requests.json"
"""The file, in the Space-Track cache, recording when each request class was
last queried."""

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
