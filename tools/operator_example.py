"""keepout's example synthetic operator, as the reports run it: the fleet,
the settings, the run on a catalog, and how the reports name and round what
it gives.

``tools/synthetic_operator_report.py`` reports its run on the frozen
catalog, ``tools/maneuver_report.py`` plans one of its events and
``tools/dashboard_seed.py`` issues its CDMs as the dashboard's seed: all with
the same fleet, thresholds and decision point.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from generated import REPO, significant

from keepout.risk import DecisionConfig, Thresholds
from keepout.screening import DEFAULT_VOLUME
from keepout.synthetic import load_fleet
from keepout.synthetic.operator import run

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from astropy.time import Time

    from keepout.omm import Omm
    from keepout.synthetic import OperatorEvent, OperatorResult

FLEET: Final = REPO / "examples/synthetic-operator/fleet.toml"
"""The example fleet."""

THRESHOLDS: Final = Thresholds()
"""The operator's risk thresholds: keepout's defaults."""

DECISION: Final = DecisionConfig()
"""The operator's decision point: keepout's default."""

NOISE_FLOOR: Final = 1e-6
"""Numbers of a CDM smaller than this in magnitude (in the CDM's units: m,
m/s, km, km/s, m**2, ...) are written as 0.0 (:func:`rounded`). They are
rounding noise, such as the off-diagonal terms (about 1e-9 m**2) of the
model's diagonal RTN covariance after its rotations, whose digits depend on
the machine."""

_NUMBER: Final = re.compile(
    r"^(?P<key>[A-Z0-9_]+\s*=\s*)"
    r"(?P<value>[-+]?(?:\d+\.\d*(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+))"
    r"(?P<unit>\s*\[[^\]]*\])?$"
)


def operator_run(
    catalog: Sequence[Omm],
    start: Time,
    duration: float,
    *,
    workers: int,
    fleet: Path = FLEET,
) -> OperatorResult:
    """The operator's run of ``fleet`` against ``catalog`` for ``duration``
    [s] from ``start``, in the default volume with :data:`THRESHOLDS` and
    :data:`DECISION`, screening with ``workers`` processes.

    Raises
    ------
    ValueError
        If a fleet satellite has no element set in ``catalog``.
    """
    return run(
        load_fleet(fleet),
        catalog,
        start,
        duration,
        DEFAULT_VOLUME,
        workers=workers,
        thresholds=THRESHOLDS,
        decision_config=DECISION,
    )


def shown(path: Path) -> str:
    """``path`` as a page names it: from the repository's root if inside it,
    else as given."""
    resolved = path.resolve()
    if resolved.is_relative_to(REPO):
        return resolved.relative_to(REPO).as_posix()
    return path.as_posix()


def level_key(event: OperatorEvent) -> str | None:
    """The key of an event's risk level; ``None`` without one."""
    level = event.assessment.level
    return None if level is None else level.key


def rounded(text: str, digits: int, floor: float = NOISE_FLOOR) -> str:
    """KVN ``text`` with every real value rounded to ``digits`` significant
    digits, and those smaller than ``floor`` in magnitude to 0.0 (integers,
    times, names and comments unchanged)."""

    def shorten(line: str) -> str:
        match = _NUMBER.match(line)
        if match is None:
            return line
        number = float(match["value"])
        number = 0.0 if abs(number) < floor else significant(number, digits)
        value = repr(number)
        return f"{match['key']}{value}{match['unit'] or ''}"

    return "\n".join(shorten(line) for line in text.split("\n"))
