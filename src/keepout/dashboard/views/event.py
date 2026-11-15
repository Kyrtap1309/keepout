"""The event page: one event (query parameter ``event``), its Pc trend, its
CDMs with the provenance of each Pc, and the encounter plane and 3-D
geometry of its latest CDM.

The page computes no Pc: a Pc the store lacks for the settings is "not
computed", and a Compute button queues a ``refresh`` job for it. The Pc
the store has are shown; the trend and the risk wait for all of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import streamlit as st

from keepout._text import pc_text, plural, utc
from keepout.dashboard.event_data import (
    EventView,
    encounter_plot,
    geometry_of,
    geometry_plot,
    is_max_pc,
    latest_conjunction,
    read_event,
    trend_of,
    trend_plot,
)
from keepout.dashboard.text import FAILED, literal, unrated
from keepout.dashboard.widgets import (
    NOT_COMPUTED,
    job_status,
    page_context,
    provenance_badge,
    risk_chip,
    show_error,
)
from keepout.errors import GeometryError, HardBodyRadiusError, KeepoutError
from keepout.events import missing_hard_body_radius, time_to_tca
from keepout.figures import RISK_LABELS, provenance
from keepout.units import HOUR

if TYPE_CHECKING:
    from keepout.cdm import Cdm
    from keepout.conjunction import Conjunction
    from keepout.dashboard.context import DashboardContext
    from keepout.risk import PcObservation, RiskLevel

# (No docstrings on the names below: Streamlit would show a bare string as
# Markdown.)
EVENTS_PAGE: Final = Path(__file__).with_name("events.py")
PLANNER_PAGE: Final = Path(__file__).with_name("planner.py")
EVENT_PARAM: Final = "event"
COLUMNS: Final = (
    ("CDM", 4),
    ("Created (UTC)", 3),
    ("Before TCA", 2),
    ("Miss [m]", 2),
    ("Pc", 2),
    ("Provenance", 3),
    ("Risk", 3),
)


def _job_key(event_id: str) -> str:
    """The session-state key of the refresh job queued for the event."""
    return f"event-refresh-{event_id}"


def _read(ctx: DashboardContext, event_id: str) -> EventView | None:
    """The event, read in one transaction; ``None`` if there is no such
    event.

    Raises
    ------
    StoreError
        If the store cannot be opened or read.
    CdmParseError, EventError
        If its stored CDMs no longer parse or make an event.
    """
    return read_event(ctx.store_path, event_id, ctx.settings)


def _to_list() -> None:
    """The way back to the events list."""
    st.page_link(EVENTS_PAGE, label="Events")


def _to_planner(event_id: str) -> None:
    """The way on to the event's manoeuvre planner."""
    st.page_link(
        PLANNER_PAGE, label="Manoeuvre planner", query_params={EVENT_PARAM: event_id}
    )


def _level_of(data: EventView, message_id: str) -> RiskLevel | None:
    """The risk level of a CDM's point on the trend, if it has one."""
    if data.assessment is None:
        return None
    for point in data.assessment.trend.points:
        if point.observation.message_id == message_id:
            return point.level
    return None


def _header(data: EventView) -> None:
    """The event's objects, its TCA, the latest Pc with its provenance and
    the risk with the recommended action."""
    event = data.event
    latest = event.latest
    st.markdown(
        f"**{literal(event.primary_id)} / {literal(event.secondary_id)}** · TCA "
        f"{utc(latest.tca)} UTC · "
        f"{plural(len(event.cdms), 'CDM')}"
    )
    pc, origin, risk = st.columns([2, 3, 4], vertical_alignment="center")
    observation = data.observations.get(latest.message_id)
    if observation is None:
        pc.markdown(f"Latest Pc: {NOT_COMPUTED}")
    else:
        pc.markdown(f"Latest Pc: {_pc_cell(observation)}")
        with origin:
            provenance_badge(provenance(observation, latest))
    assessment = data.assessment
    with risk:
        if assessment is None:
            risk_chip(None, None)
        elif assessment.level is None:
            risk_chip(None, None, missing=unrated(assessment))
        else:
            risk_chip(assessment.level, assessment.decision.action)
    if assessment is not None:
        for reason in assessment.decision.reasons:
            st.caption(reason)


def _pc_cell(observation: PcObservation) -> str:
    """A CDM's Pc as the page shows it, :data:`~keepout.dashboard.text.FAILED`
    if it could not be computed."""
    return FAILED if observation.pc is None else pc_text(observation.pc)


def _compute(ctx: DashboardContext, event_id: str) -> None:
    """Queue the refresh of the event's missing Pc (or find the one queued)
    and keep its job for the page; every Pc cached meanwhile reruns it."""
    from keepout.store import JobQueue, open_store

    try:
        with open_store(ctx.store_path, create=False) as store:
            submission = JobQueue(store).submit_refresh(event_id, ctx.settings)
    except KeepoutError as exc:
        show_error(exc)
        return
    except ValueError as exc:
        st.error(literal(str(exc)))
        return
    if submission.job is None:
        st.rerun()
    st.session_state[_job_key(event_id)] = submission.job.id


def _not_computed(ctx: DashboardContext, data: EventView) -> None:
    """Which Pc are not computed, and the button that queues them."""
    missing = data.missing
    st.warning(
        f"Pc {NOT_COMPUTED} for {len(missing)} of "
        f"{plural(len(data.event.cdms), 'CDM')} ({literal(', '.join(missing))}): "
        "the trend and risk are shown once a worker has computed them."
    )
    event_id = data.event.event_id
    if st.button("Compute", key=f"event-compute-{event_id}"):
        _compute(ctx, event_id)


def _jobs(ctx: DashboardContext, data: EventView) -> None:
    """The status of the event's refresh jobs: those the store has queued
    or running, and the one this session queued (also once it ended)."""
    queued = st.session_state.get(_job_key(data.event.event_id))
    ids = list(data.refresh_jobs)
    if isinstance(queued, int) and queued not in ids:
        ids.append(queued)
    if ids:
        job_status(ctx.store_path, ids)


def _trend(ctx: DashboardContext, data: EventView) -> None:
    """The Pc against the hours to TCA with the thresholds and the decision
    point, and its data; or what is not computed."""
    st.subheader("Pc trend")
    if data.missing:
        _not_computed(ctx, data)
    _jobs(ctx, data)
    assessment = data.assessment
    figure = trend_of(data, ctx.settings)
    if assessment is None or figure is None:
        return
    if not figure.labels:
        st.info("No Pc before TCA to draw.")
        return
    st.plotly_chart(trend_plot(figure))
    cdms = {cdm.message_id: cdm for cdm in data.event.cdms}
    origins = [
        provenance(point.observation, cdms[label]).value if label in cdms else ""
        for point, label in zip(assessment.trend.points, figure.labels, strict=True)
    ]
    st.dataframe(
        {
            "CDM": list(figure.labels),
            "hours to TCA": figure.hours_to_tca.tolist(),
            "Pc": figure.pc.tolist(),
            "risk": [RISK_LABELS[level] for level in figure.levels],
            "provenance": origins,
        },
        hide_index=True,
    )


def _cdm_row(data: EventView, cdm: Cdm) -> None:
    """A CDM of the event: when it was made, its miss, its Pc next to its
    provenance and the risk of its point on the trend ("-" until the event
    is assessed, or if it has no point)."""
    cells = st.columns([width for _, width in COLUMNS], vertical_alignment="center")
    name, created, before, miss, pc, origin, risk = cells
    name.markdown(literal(cdm.message_id))
    created.markdown(utc(cdm.creation_date))
    before.markdown(f"{time_to_tca(cdm) / HOUR:.1f} h")
    miss.markdown(f"{cdm.miss_distance:.1f}")
    observation = data.observations.get(cdm.message_id)
    if observation is None:
        pc.markdown(NOT_COMPUTED)
        return
    pc.markdown(_pc_cell(observation))
    with origin:
        provenance_badge(provenance(observation, cdm))
    level = _level_of(data, cdm.message_id)
    with risk:
        if level is None:
            st.markdown("-")
        else:
            risk_chip(level, None)


def _cdms(data: EventView) -> None:
    """Every CDM of the event, oldest first."""
    st.subheader("CDMs")
    for column, (title, _) in zip(
        st.columns([width for _, width in COLUMNS]), COLUMNS, strict=True
    ):
        column.markdown(f"**{title}**")
    for cdm in data.event.cdms:
        _cdm_row(data, cdm)


def _conjunction(ctx: DashboardContext, data: EventView) -> Conjunction | None:
    """The latest CDM's conjunction with the settings' hard-body radius (or
    the CDM's), its encounter plane checked; ``None``, with the reason
    shown, if there is none."""
    latest = data.event.latest
    try:
        conjunction = latest_conjunction(data, ctx.settings)
    except HardBodyRadiusError:
        st.warning(
            f"No hard-body radius: {literal(missing_hard_body_radius([latest]))}, "
            "and the settings give none (hbr_m)."
        )
    except GeometryError as exc:
        st.warning(f"No encounter plane or geometry at TCA: {literal(str(exc))}")
    except KeepoutError as exc:
        show_error(exc)
    else:
        return conjunction
    return None


def _encounter(conjunction: Conjunction, *, max_pc: bool) -> None:
    """The latest CDM's encounter plane and its numbers."""
    st.subheader("Encounter plane")
    plane = conjunction.plane
    hbr = conjunction.combined_hard_body_radius
    if max_pc:
        st.caption("no covariance: max-Pc")
    st.plotly_chart(encounter_plot(conjunction, max_pc=max_pc))
    sigmas = np.sqrt(np.clip(np.linalg.eigvalsh(plane.covariance), 0.0, None))
    x, z = (float(value) for value in plane.miss_vector)
    rows = [
        ("miss x [m]", x),
        ("miss z [m]", z),
        ("miss distance [m]", plane.miss_distance),
        ("hard-body radius [m]", hbr),
    ]
    if not max_pc:
        rows += [
            ("1 sigma, major axis [m]", float(sigmas[1])),
            ("1 sigma, minor axis [m]", float(sigmas[0])),
        ]
    st.dataframe(
        {
            "quantity": [name for name, _ in rows],
            "value": [value for _, value in rows],
        },
        hide_index=True,
    )


def _geometry(conjunction: Conjunction, *, max_pc: bool) -> None:
    """The latest CDM's relative geometry at TCA and its numbers."""
    st.subheader("Geometry at TCA")
    try:
        figure = geometry_of(conjunction)
    except GeometryError as exc:
        # The primary's RTN frame: its position zero or along its velocity.
        st.warning(f"No geometry at TCA: {literal(str(exc))}")
        return
    st.plotly_chart(geometry_plot(figure, max_pc=max_pc))
    st.dataframe(
        {
            "axis": [label.split()[0] for label in figure.axes],
            "secondary at TCA [m]": list(figure.at_tca),
            "relative velocity [m/s]": list(figure.velocity),
        },
        hide_index=True,
    )


def _latest_geometry(ctx: DashboardContext, data: EventView) -> None:
    """The encounter plane and the geometry at TCA of the latest CDM, or
    why they cannot be drawn."""
    latest = data.event.latest
    st.caption(
        "Encounter plane and geometry at TCA of the latest CDM, "
        f"{literal(latest.message_id)}."
    )
    conjunction = _conjunction(ctx, data)
    if conjunction is None:
        return
    max_pc = is_max_pc(data, conjunction)
    _encounter(conjunction, max_pc=max_pc)
    _geometry(conjunction, max_pc=max_pc)


def main() -> None:
    """The event page."""
    ctx = page_context()
    event_id = st.query_params.get(EVENT_PARAM)
    if not event_id:
        st.title("Event")
        st.info("Pick an event from the list.")
        _to_list()
        return
    try:
        data = _read(ctx, event_id)
    except KeepoutError as exc:
        st.title("Event")
        show_error(exc)
        _to_list()
        return
    if data is None:
        st.title("Event")
        st.warning(f"No such event: {literal(event_id)}")
        _to_list()
        return
    st.title(f"Event {event_id}")
    _header(data)
    _to_planner(event_id)
    _trend(ctx, data)
    _cdms(data)
    _latest_geometry(ctx, data)


main()
