"""The event page's data, without Streamlit: an event read from the store
and assessed, and the figures of its Pc trend, encounter plane and geometry
at TCA drawn with Plotly (``tools/dashboard_benchmark.py`` times them).

Nothing here computes a Pc: the page shows the Pc the store has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from keepout.dashboard import plots
from keepout.figures import (
    Provenance,
    encounter_figure,
    geometry_figure,
    provenance,
    trend_figure,
)
from keepout.risk import DEFAULT_DECISION_CONFIG, assess
from keepout.units import HOUR

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import plotly.graph_objects as go

    from keepout.conjunction import Conjunction
    from keepout.events import ConjunctionEvent
    from keepout.figures import GeometryFigure, TrendFigure
    from keepout.risk import Assessment, PcObservation
    from keepout.settings import Settings
    from keepout.store import EventDetail

__all__ = [
    "EventView",
    "encounter_plot",
    "geometry_of",
    "geometry_plot",
    "is_max_pc",
    "latest_conjunction",
    "read_event",
    "trend_of",
    "trend_plot",
    "view_of",
]


@dataclass(frozen=True, slots=True)
class EventView:
    """What the event page shows of an event.

    Parameters
    ----------
    event
        The event and its CDMs.
    missing
        The ``MESSAGE_ID`` of each CDM whose Pc is not computed, oldest first.
    observations
        The cached Pc observation of each CDM that has one, by ``MESSAGE_ID``.
    assessment
        The event's risk; ``None`` unless every Pc is computed.
    refresh_jobs
        The ids of the event's queued or running ``refresh`` jobs.
    """

    event: ConjunctionEvent
    missing: tuple[str, ...]
    observations: Mapping[str, PcObservation]
    assessment: Assessment | None
    refresh_jobs: tuple[int, ...]


def view_of(detail: EventDetail, settings: Settings) -> EventView:
    """The page's view of an event read from the store: assessed with the
    settings' thresholds only when every Pc is computed, as the events list
    does.

    Parameters
    ----------
    detail
        The event as :meth:`keepout.store.Store.detail` reads it.
    settings
        The settings it was read with.

    Returns
    -------
    EventView
        The view.
    """
    complete = detail.complete
    assessment = assess(complete, thresholds=settings.thresholds) if complete else None
    observations: dict[str, PcObservation] = {
        message_id: item for message_id, item in detail.observations if item is not None
    }
    return EventView(
        detail.event, detail.missing, observations, assessment, detail.refresh_jobs
    )


def read_event(store_path: Path, event_id: str, settings: Settings) -> EventView | None:
    """An event read in one transaction, and its view.

    Parameters
    ----------
    store_path
        The store's file.
    event_id
        The store's id of the event.
    settings
        The settings of the cached Pc and their thresholds.

    Returns
    -------
    EventView or None
        The view; ``None`` if there is no such event.

    Raises
    ------
    StoreError
        If the store cannot be opened or read.
    CdmParseError, EventError
        If its stored CDMs no longer parse or make an event.
    """
    from keepout.store import open_store

    with open_store(store_path, create=False) as store:
        detail = store.detail(event_id, settings)
    return None if detail is None else view_of(detail, settings)


def trend_of(view: EventView, settings: Settings) -> TrendFigure | None:
    """The event's Pc trend with the settings' thresholds; ``None`` until it
    is assessed.

    Parameters
    ----------
    view
        The event.
    settings
        The settings whose thresholds the trend shows.

    Returns
    -------
    TrendFigure or None
        The figure's data.
    """
    if view.assessment is None:
        return None
    return trend_figure(view.assessment, settings.thresholds)


def trend_plot(figure: TrendFigure) -> go.Figure:
    """The Pc trend drawn with Plotly, with the decision point.

    Parameters
    ----------
    figure
        The trend, e.g. of :func:`trend_of`.

    Returns
    -------
    plotly.graph_objects.Figure
        The plot.
    """
    hours = DEFAULT_DECISION_CONFIG.decision_point / HOUR
    return plots.trend(figure, decision_point_hours=hours)


def latest_conjunction(view: EventView, settings: Settings) -> Conjunction:
    """The latest CDM's conjunction with the settings' hard-body radius (or
    the CDM's) and policy, its encounter plane built.

    Parameters
    ----------
    view
        The event.
    settings
        The settings giving the hard-body radius and covariance policy.

    Returns
    -------
    Conjunction
        The conjunction; its ``plane`` is cached.

    Raises
    ------
    HardBodyRadiusError
        If neither the settings nor the CDM give a hard-body radius.
    GeometryError
        If there is no encounter plane at TCA.
    KeepoutError
        If the CDM's states or covariances cannot be used otherwise.
    """
    conjunction = view.event.latest.conjunction(
        combined_hard_body_radius=settings.combined_hard_body_radius,
        policy=settings.policy,
    )
    conjunction.plane  # noqa: B018 (built and cached, or GeometryError)
    return conjunction


def is_max_pc(view: EventView, conjunction: Conjunction) -> bool:
    """Whether the latest Pc is a maximum Pc, or both covariances are
    unknown: no ellipse is drawn then.

    Parameters
    ----------
    view
        The event.
    conjunction
        Its latest CDM's conjunction.

    Returns
    -------
    bool
        Whether the plots leave out the covariance.
    """
    latest = view.event.latest
    observation = view.observations.get(latest.message_id)
    if observation is not None and provenance(observation, latest) is (
        Provenance.MAX_PC
    ):
        return True
    return conjunction.primary.covariance is None and (
        conjunction.secondary.covariance is None
    )


def encounter_plot(conjunction: Conjunction, *, max_pc: bool) -> go.Figure:
    """The encounter plane of a conjunction drawn with Plotly.

    Parameters
    ----------
    conjunction
        The conjunction, its plane built.
    max_pc
        Leave out the covariance ellipse (:func:`is_max_pc`).

    Returns
    -------
    plotly.graph_objects.Figure
        The plot.
    """
    figure = encounter_figure(conjunction.plane, conjunction.combined_hard_body_radius)
    return plots.encounter(figure, covariance=not max_pc)


def geometry_of(conjunction: Conjunction) -> GeometryFigure:
    """The relative geometry at TCA of a conjunction.

    Parameters
    ----------
    conjunction
        The conjunction.

    Returns
    -------
    GeometryFigure
        The figure's data.

    Raises
    ------
    GeometryError
        If the primary's RTN frame is undefined (its position zero or along
        its velocity).
    """
    return geometry_figure(conjunction, conjunction.combined_hard_body_radius)


def geometry_plot(figure: GeometryFigure, *, max_pc: bool) -> go.Figure:
    """The geometry at TCA drawn with Plotly.

    Parameters
    ----------
    figure
        The geometry, e.g. of :func:`geometry_of`.
    max_pc
        Leave out the covariance ellipsoid (:func:`is_max_pc`).

    Returns
    -------
    plotly.graph_objects.Figure
        The plot.
    """
    return plots.geometry(figure, covariance=not max_pc)
