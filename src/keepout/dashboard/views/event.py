"""The event page: one event (query parameter ``event``), its Pc trend, its
CDMs with the provenance of each Pc, and the encounter plane and 3-D
geometry of its latest CDM.

The page computes no Pc: a Pc the store lacks for the settings is "not
computed", and a Compute button queues a ``refresh`` job for it. The Pc
the store has are shown; the trend and the risk wait for all of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import streamlit as st

from keepout._text import plural, utc
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
from keepout.dashboard.routes import (
    EVENT_PARAM,
    EVENTS_PAGE,
    PLANNER_PAGE,
    refresh_job_key,
)
from keepout.dashboard.text import NOT_COMPUTED, literal, pc_cell
from keepout.dashboard.widgets import (
    assessment_chip,
    job_status,
    page_context,
    provenance_badge,
    risk_chip,
    show_error,
    state,
    table_header,
    table_row,
)
from keepout.errors import GeometryError, HardBodyRadiusError, KeepoutError
from keepout.events import missing_hard_body_radius, time_to_tca
from keepout.figures import PROVENANCE_LABELS, RISK_LABELS
from keepout.risk import provenance
from keepout.units import HOUR

if TYPE_CHECKING:
    from collections.abc import Mapping

    from keepout.cdm import Cdm
    from keepout.conjunction import Conjunction
    from keepout.dashboard.context import DashboardContext
    from keepout.risk import PcObservation, Provenance, RiskLevel

# (No docstring: Streamlit would show a bare string as Markdown.)
COLUMNS: Final = (
    ("CDM", 4),
    ("Created (UTC)", 3),
    ("Before TCA", 2),
    ("Miss [m]", 2),
    ("Pc", 2),
    ("Provenance", 3),
    ("Risk", 3),
)


def _to_list() -> None:
    """The way back to the events list."""
    st.page_link(EVENTS_PAGE, label="Events")


def _to_planner(event_id: str) -> None:
    """The way on to the event's manoeuvre planner."""
    st.page_link(
        PLANNER_PAGE, label="Manoeuvre planner", query_params={EVENT_PARAM: event_id}
    )


def _provenance(observation: PcObservation, cdm: Cdm) -> Provenance:
    """What the Pc of ``cdm`` rests on."""
    return provenance(observation, lambda: cdm.header.comments)


def _header(data: EventView) -> None:
    """The event's objects, its TCA, the latest Pc with its provenance and
    the risk with the recommended action."""
    event = data.detail.event
    latest = event.latest
    st.markdown(
        f"**{literal(event.primary_id)} / {literal(event.secondary_id)}** · TCA "
        f"{utc(latest.tca)} UTC · "
        f"{plural(len(event.cdms), 'CDM')}"
    )
    pc, origin, risk = st.columns([2, 3, 4], vertical_alignment="center")
    observation = data.observations.get(latest.message_id)
    pc.markdown(f"Latest Pc: {pc_cell(observation)}")
    if observation is not None:
        with origin:
            provenance_badge(_provenance(observation, latest))
    assessment = data.assessment
    with risk:
        assessment_chip(assessment)
    if assessment is not None:
        for reason in assessment.decision.reasons:
            st.caption(reason)


def _compute(ctx: DashboardContext, event_id: str) -> None:
    """Queue the refresh of the event's missing Pc (or find the one queued)
    and keep its job for the page; every Pc cached meanwhile reruns it."""
    from keepout.store import JobQueue

    try:
        with ctx.store() as store:
            submission = JobQueue(store).submit_refresh(event_id, ctx.settings)
    except (KeepoutError, ValueError) as exc:
        show_error(exc)
        return
    if submission.job is None:
        st.rerun()
    st.session_state[refresh_job_key(event_id)] = submission.job.id


def _not_computed(ctx: DashboardContext, data: EventView) -> None:
    """Which Pc are not computed, and the button that queues them."""
    event, missing = data.detail.event, data.detail.missing
    st.warning(
        f"Pc {NOT_COMPUTED} for {len(missing)} of "
        f"{plural(len(event.cdms), 'CDM')} ({literal(', '.join(missing))}): "
        "the trend and risk are shown once a worker has computed them."
    )
    event_id = event.event_id
    if st.button("Compute", key=f"event-compute-{event_id}"):
        _compute(ctx, event_id)


def _jobs(ctx: DashboardContext, data: EventView) -> None:
    """The status of the event's refresh jobs: those the store has queued
    or running, and the one this session queued (also once it ended)."""
    queued = state(refresh_job_key(data.detail.event.event_id), int)
    ids = list(data.detail.refresh_jobs)
    if queued is not None and queued not in ids:
        ids.append(queued)
    if ids:
        job_status(ctx.store_path, ids)


def _trend(ctx: DashboardContext, data: EventView) -> None:
    """The Pc against the hours to TCA with the thresholds and the decision
    point, and its data; or what is not computed."""
    st.subheader("Pc trend")
    if data.detail.missing:
        _not_computed(ctx, data)
    _jobs(ctx, data)
    figure = trend_of(data, ctx.settings)
    if figure is None:
        return
    if not figure.labels:
        st.info("No Pc before TCA to draw.")
        return
    st.plotly_chart(trend_plot(figure))
    # Assessed: every CDM has its observation.
    cdms = {cdm.message_id: cdm for cdm in data.detail.event.cdms}
    origins = [
        PROVENANCE_LABELS[_provenance(data.observations[label], cdms[label])]
        if label in cdms
        else ""
        for label in figure.labels
    ]
    st.dataframe(
        {
            "CDM": list(figure.labels),
            "hours to TCA": figure.hours_to_tca.tolist(),
            "Pc": figure.pc.tolist(),
            "risk": [RISK_LABELS[level] for level in figure.levels],
            "provenance": origins,
        },
        column_config={
            "hours to TCA": st.column_config.NumberColumn(format="%.1f"),
            "Pc": st.column_config.NumberColumn(format="%.2e"),
        },
        hide_index=True,
    )


def _cdm_row(data: EventView, cdm: Cdm, level: RiskLevel | None) -> None:
    """A CDM of the event: when it was made, its miss, its Pc next to its
    provenance and the risk ``level`` of its point on the trend ("-" until
    the event is assessed, or if it has no point)."""
    name, created, before, miss, pc, origin, risk = table_row(COLUMNS)
    name.markdown(literal(cdm.message_id))
    created.markdown(utc(cdm.creation_date))
    before.markdown(f"{time_to_tca(cdm) / HOUR:.1f} h")
    miss.markdown(f"{cdm.miss_distance:.1f}")
    observation = data.observations.get(cdm.message_id)
    pc.markdown(pc_cell(observation))
    if observation is None:
        return
    with origin:
        provenance_badge(_provenance(observation, cdm))
    with risk:
        if level is None:
            st.markdown("-")
        else:
            risk_chip(level, None)


def _levels(data: EventView) -> Mapping[str | None, RiskLevel]:
    """The risk level of each CDM's point on the trend, by ``MESSAGE_ID``;
    none until the event is assessed."""
    if data.assessment is None:
        return {}
    return {
        point.observation.message_id: point.level
        for point in data.assessment.trend.points
    }


def _cdms(data: EventView) -> None:
    """Every CDM of the event, oldest first."""
    st.subheader("CDMs")
    table_header(COLUMNS)
    levels = _levels(data)
    for cdm in data.detail.event.cdms:
        _cdm_row(data, cdm, levels.get(cdm.message_id))


def _conjunction(ctx: DashboardContext, data: EventView) -> Conjunction | None:
    """The latest CDM's conjunction with the settings' hard-body radius (or
    the CDM's), its encounter plane checked; ``None``, with the reason
    shown, if there is none."""
    latest = data.detail.event.latest
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
        st.caption("No covariance: the Pc shown is an upper bound (maximum Pc).")
    st.plotly_chart(encounter_plot(conjunction, max_pc=max_pc))
    sigmas = np.sqrt(np.clip(np.linalg.eigvalsh(plane.covariance), 0.0, None))
    x, z = plane.miss_vector.item(0), plane.miss_vector.item(1)
    rows = [
        ("miss x [m]", x),
        ("miss z [m]", z),
        ("miss distance [m]", plane.miss_distance),
        ("hard-body radius [m]", hbr),
    ]
    if not max_pc:
        rows += [
            ("1 sigma, major axis [m]", sigmas.item(1)),
            ("1 sigma, minor axis [m]", sigmas.item(0)),
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
    latest = data.detail.event.latest
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
    data: EventView | None = None
    error: KeepoutError | None = None
    if event_id:
        try:
            data = read_event(ctx.store_path, event_id, ctx.settings)
        except KeepoutError as exc:
            error = exc
    if not event_id or data is None:
        st.title("Event")
        if not event_id:
            st.info("Pick an event from the list.")
        elif error is not None:
            show_error(error)
        else:
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
