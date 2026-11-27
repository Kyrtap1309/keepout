"""The Iridium 33-Cosmos 2251 case study: its constants and data, shared by
the report (``tools/iridium_cosmos_report.py``), the data tool
(``tools/iridium_cosmos_data.py``) and the dashboard seed.

On 2009 February 10 Iridium 33 (24946), an operational communications
satellite, and Cosmos 2251 (22675), a long-defunct one, collided over northern
Russia (Kelso 2009). The study runs keepout on the element sets that were
public then: the 2009 bulk TLE file of Space-Track.org, reduced to the
committed excerpt in :data:`DATA` by the data tool.

Times. :data:`TCA_REFERENCE` is the collision time Kelso (2009, sec. 3) uses:
the time of closest approach predicted by the last SOCRATES report before it.
:data:`IRIDIUM_AS_OF` is the decision point two days before
(:data:`~keepout.risk.DEFAULT_DECISION_CONFIG`). An analysis "as known at" a
time sees only the element sets with an epoch up to that time
(:func:`known_at`).

References
----------
T.S. Kelso, "Analysis of the Iridium 33-Cosmos 2251 Collision", 10th AMOS
Conference, Maui, 2009 September 2.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Final

import astropy.units as u
from astropy.time import Time
from generated import REPO

from keepout.omm import load
from keepout.risk import DEFAULT_DECISION_CONFIG
from keepout.units import DAY, KM

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from keepout.omm import Omm

IRIDIUM_33: Final = 24946
"""Iridium 33's catalog number (international designator 1997-051C)."""

COSMOS_2251: Final = 22675
"""Cosmos 2251's catalog number (international designator 1993-036A)."""

PAIR: Final = (IRIDIUM_33, COSMOS_2251)
"""The two objects that collided, Iridium 33 first."""

NAMES: Final[Mapping[int, str]] = {IRIDIUM_33: "IRIDIUM 33", COSMOS_2251: "COSMOS 2251"}
"""The pair's names: the bulk file's TLEs carry none."""

TCA_REFERENCE: Final = Time("2009-02-10T16:55:59.806", scale="utc")
"""The collision time: the TCA of the last SOCRATES report before it (Kelso
2009, sec. 3)."""

IRIDIUM_AS_OF: Final = TCA_REFERENCE - DEFAULT_DECISION_CONFIG.decision_point * u.s
"""The decision point, two days before :data:`TCA_REFERENCE`: the time the
dashboard shows the event at and the ranking sees the catalog at."""

PAIR_SPAN: Final = 30 * DAY
"""How long before :data:`TCA_REFERENCE` the pair's element sets are kept [s]."""

SERIES_SPAN: Final = 7 * DAY
"""How long before :data:`TCA_REFERENCE` the synthetic CDMs are issued [s]."""

BASE_HBR: Final = 6.0
"""The combined hard-body radius [m]: an enclosing-sphere estimate (Iridium 33
with its arrays about 4 m, Cosmos 2251 about 2 m), an assumption."""

HBR_SENSITIVITY: Final = (5.0, 10.0, 20.0)
"""Further combined radii [m] the Pc is shown at."""

EXCERPT_RADIUS: Final = 100 * KM
"""Radius of the sphere around Iridium 33 within which an object's approach
puts it in the excerpt [m]."""

DATA: Final = REPO / "tests/case_study/data/iridium-cosmos"
"""The committed excerpt, its NOTICE and manifest."""

CATALOG_FILE: Final = "iridium-cosmos-2009.csv.gz"
"""The excerpt's element sets, as gzipped CelesTrak CSV."""


def known_at(omms: Iterable[Omm], time: Time) -> tuple[Omm, ...]:
    """The element sets with an epoch up to ``time``: what was known then."""
    return tuple(omm for omm in omms if omm.epoch <= time)


def with_name(omm: Omm) -> Omm:
    """``omm`` with its object's name from :data:`NAMES`, if it has one there."""
    name = NAMES.get(omm.catalog_number)
    return omm if name is None else dataclasses.replace(omm, object_name=name)


def read_catalog(data: Path = DATA, *, named: bool = True) -> tuple[Omm, ...]:
    """The excerpt's element sets; with ``named``, the pair's named
    (:func:`with_name`)."""
    omms = load(data / CATALOG_FILE)
    return tuple(with_name(omm) for omm in omms) if named else omms


def read_pair(data: Path = DATA) -> tuple[Omm, ...]:
    """The pair's element sets from :data:`PAIR_SPAN` before the collision."""
    start = TCA_REFERENCE - PAIR_SPAN * u.s
    return tuple(
        omm
        for omm in read_catalog(data)
        if omm.catalog_number in PAIR and omm.epoch >= start
    )
