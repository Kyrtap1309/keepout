"""The Pc of stored CDMs: their texts parsed once in the process, and their
observations computed in batches.

:meth:`keepout.store.Store.refresh` and a refresh job
(:func:`keepout.store.planning.run_refresh_job`) compute the same way: the
CDMs lacking ``HBR`` are found first (:func:`lacking_hard_body_radius`),
then each CDM's observation (:func:`observation_of`), handed on in batches
(:func:`batches`) to be written as they come.
"""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING, Final

import keepout.cdm
from keepout.errors import HardBodyRadiusError
from keepout.events import (
    ConjunctionEvent,
    missing_hard_body_radius,
    without_hard_body_radius,
)
from keepout.settings import observation_arguments

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from keepout.cdm import Cdm
    from keepout.risk import PcObservation
    from keepout.settings import Settings

__all__ = [
    "BATCH_SECONDS",
    "BATCH_SIZE",
    "batches",
    "lacking_hard_body_radius",
    "observation_of",
    "parsed_cdm",
    "parsed_event",
    "refuse_lacking_hard_body_radius",
]

BATCH_SIZE: Final = 100
"""Most observations in a batch (:func:`batches`), and most CDM texts
:meth:`~keepout.store.Store.refresh` reads in one transaction."""

BATCH_SECONDS: Final = 1.0
"""Longest computing a batch (:func:`batches`) waits for, in seconds."""

_PARSED_CDMS: Final = 1024
"""How many stored CDMs the process keeps parsed (:func:`parsed_cdm`): the
CDMs of some 50 events of 20 CDMs, about 25 MB with their texts."""

_GIVE_HARD_BODY_RADIUS: Final = "give the combined hard-body radius in the settings"
"""The hint of a Pc that lacks a hard-body radius."""


@functools.lru_cache(maxsize=_PARSED_CDMS)
def parsed_cdm(text: str) -> Cdm:
    """A stored CDM's text, parsed; the last :data:`_PARSED_CDMS` texts
    parsed are kept, and shared.

    A stored CDM's text never changes (a ``MESSAGE_ID`` is stored once),
    and a :class:`~keepout.cdm.Cdm` cannot be changed (frozen, its
    covariance read-only): a page shown again reads its event's CDMs
    without parsing them again.

    Raises
    ------
    CdmParseError
        If the text does not parse (a store edited by hand); not kept.
    """
    return keepout.cdm.loads(text)


def parsed_event(event_id: str, texts: Iterable[str]) -> ConjunctionEvent:
    """The event of stored CDM texts, each parsed once in the process.

    Raises
    ------
    CdmParseError
        If a text no longer parses (a store edited by hand).
    EventError
        If the CDMs no longer make an event (likewise).
    """
    return ConjunctionEvent(event_id, tuple(parsed_cdm(text) for text in texts))


def lacking_hard_body_radius(settings: Settings, cdms: Iterable[Cdm]) -> list[Cdm]:
    """The CDMs of ``cdms`` without ``HBR`` for an object, if the settings
    give no hard-body radius (``cdms`` is not read if they do)."""
    if settings.combined_hard_body_radius is not None:
        return []
    return [cdm for cdm in cdms if without_hard_body_radius((cdm,))]


def refuse_lacking_hard_body_radius(lacking: Sequence[Cdm]) -> None:
    """Refuse the Pc of CDMs that lack ``HBR`` (:func:`lacking_hard_body_radius`).

    Raises
    ------
    HardBodyRadiusError
        If there is one.
    """
    if lacking:
        raise HardBodyRadiusError(
            missing_hard_body_radius(lacking), _GIVE_HARD_BODY_RADIUS
        )


def observation_of(event_id: str, cdm: Cdm, settings: Settings) -> PcObservation:
    """The Pc observation of a CDM of an event, computed with ``settings``."""
    (observation,) = ConjunctionEvent(event_id, (cdm,)).observations(
        **observation_arguments(settings)
    )
    return observation


def batches(
    observations: Iterable[PcObservation],
) -> Iterator[list[PcObservation]]:
    """``observations``, computed as they are iterated, in batches of
    :data:`BATCH_SIZE` or of :data:`BATCH_SECONDS` of computing, whichever
    comes first; the time a batch's consumer takes is not counted."""
    pending: list[PcObservation] = []
    since = time.monotonic()
    for observation in observations:
        pending.append(observation)
        if len(pending) >= BATCH_SIZE or time.monotonic() - since >= BATCH_SECONDS:
            yield pending
            pending, since = [], time.monotonic()
    if pending:
        yield pending
