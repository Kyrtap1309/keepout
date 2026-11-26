"""The event page: one event (query parameter ``event``), its Pc trend, its
CDMs with the provenance of each Pc, and the encounter plane and 3-D
geometry of its latest CDM.

The page computes no Pc: a Pc the store lacks for the settings is "not
computed", and a Compute button queues a ``refresh`` job for it. The Pc
the store has are shown; the trend and the risk wait for all of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import streamlit as st

from keepout._text import pc_cell, plural, utc
from keepout.dashboard.event_data import (
    EventView,
    encounter_plot,
    event_view,
    geometry_of,
    geometry_plot,
    is_max_pc,
    latest_conjunction,
    trend_of,
    trend_plot,
)
from keepout.dashboard.routes import EVENT_PARAM, PLANNER_PAGE, refresh_job_key
from keepout.dashboard.text import (
    MISSING,
    NOT_COMPUTED,
    british,
    literal,
    risk_text,
)
from keepout.dashboard.widgets import (
    assessment_chip,
    back_links,
    event_or_stop,
    job_status,
    page_context,
    provenance_badge,
    read_jobs,
    shared_store,
    show_error,
    state,
)
from keepout.errors import GeometryError, HardBodyRadiusError, KeepoutError
from keepout.events import missing_hard_body_radius, time_to_tca
from keepout.figures import PROVENANCE_LABELS
from keepout.risk import RiskLevel, provenance
from keepout.units import HOUR

if TYPE_CHECKING:
    from collections.abc import Mapping

    from keepout.cdm import Cdm
    from keepout.conjunction import Conjunction
    from keepout.dashboard.context import DashboardContext
    from keepout.risk import Assessment, PcObservation, Provenance

_METRES = st.column_config.NumberColumn(format="%.1f")


def _provenance(observation: PcObservation, cdm: Cdm) -> Provenance:
    """What the Pc of ``cdm`` rests on."""
    return provenance(observation, lambda: cdm.header.comments)


def _to_planner(event_id: str, assessment: Assessment | None) -> None:
    """The way on to the event's manoeuvre planner: a primary button for a
    red event."""
    query = {EVENT_PARAM: event_id}
    label, icon = "Manoeuvre planner", ":material/rocket_launch:"
    if assessment is not None and assessment.level is RiskLevel.RED:
        if st.button(label, type="primary", icon=icon, key=f"event-plan-{event_id}"):
            st.switch_page(PLANNER_PAGE, query_params=query)
        return
    st.page_link(PLANNER_PAGE, label=label, icon=icon, query_params=query)


def _why(assessment: Assessment) -> None:
    """The reasons for the event's risk and action."""
    if assessment.level is None or not assessment.decision.reasons:
        return
    action = british(assessment.decision.action.label)
    reasons = "\n".join(
        f"- {literal(british(reason))}" for reason in assessment.decision.reasons
    )
    st.info(f"**{assessment.level.key} · {action}**, because:\n\n{reasons}")


def _header(data: EventView) -> None:
    """The event's objects, its TCA, the latest Pc with its provenance and
    the risk with the recommended action, and why."""
    event = data.detail.event
    latest = event.latest
    st.markdown(
        f"**{literal(event.primary_id)} / {literal(event.secondary_id)}** · TCA "
        f"{utc(latest.tca)} UTC · "
        f"{plural(len(event.cdms), 'CDM')}"
    )
    line = st.container(horizontal=True, vertical_alignment="center")
    observation = data.observations.get(latest.message_id)
    with line:
        st.markdown(f"Latest Pc: {pc_cell(observation, missing=NOT_COMPUTED)}")
        if observation is not None:
            provenance_badge(_provenance(observation, latest))
        assessment_chip(data.assessment)
    if data.assessment is not None:
        _why(data.assessment)


def _compute(ctx: DashboardContext, event_id: str) -> None:
    """Queue the refresh of the event's missing Pc (or find the one queued)
    and keep its job for the page; every Pc cached meanwhile reruns it."""
    from keepout.store import JobQueue

    try:
        queue = JobQueue(shared_store(ctx.store_path))
        submission = queue.submit_refresh(event_id, ctx.settings)
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
    if not ids:
        return
    try:
        jobs = read_jobs(ctx.store_path, ids)
    except KeepoutError as exc:
        show_error(exc)
        return
    job_status(ctx.store_path, jobs)


def _trend(ctx: DashboardContext, data: EventView) -> None:
    """The Pc against the hours to TCA with the thresholds and the decision
    point (its data: the CDMs table); or what is not computed."""
    st.subheader("Pc trend")
    if data.detail.missing:
        _not_computed(ctx, data)
    _jobs(ctx, data)
    figure = trend_of(data)
    if figure is None:
        return
    if not figure.labels:
        st.info("No Pc before TCA to draw.")
        return
    st.plotly_chart(trend_plot(figure))


def _levels(data: EventView) -> Mapping[str | None, RiskLevel]:
    """The risk level of each CDM's point on the trend, by ``MESSAGE_ID``;
    none until the event is assessed."""
    if data.assessment is None:
        return {}
    return {
        point.observation.message_id: point.level
        for point in data.assessment.trend.points
    }


def _cdm_row(data: EventView, cdm: Cdm, level: RiskLevel | None) -> dict[str, object]:
    """A CDM of the event; its risk is its point's on the trend."""
    observation = data.observations.get(cdm.message_id)
    origin = None if observation is None else _provenance(observation, cdm)
    return {
        "CDM": cdm.message_id,
        "Created (UTC)": utc(cdm.creation_date),
        "Before TCA (h)": time_to_tca(cdm) / HOUR,
        "Miss (m)": cdm.miss_distance,
        "Pc": pc_cell(observation, missing=NOT_COMPUTED),
        "Provenance": None if origin is None else PROVENANCE_LABELS[origin],
        "Risk": None if level is None else risk_text(level),
    }


def _cdms(data: EventView) -> None:
    """Every CDM of the event, oldest first: the trend's data too."""
    st.subheader("CDMs")
    levels = _levels(data)
    st.dataframe(
        [
            _cdm_row(data, cdm, levels.get(cdm.message_id))
            for cdm in data.detail.event.cdms
        ],
        column_config={
            # A MESSAGE_ID is long: the other columns show beside it.
            "CDM": st.column_config.TextColumn(width="medium"),
            "Before TCA (h)": st.column_config.NumberColumn(format="%.1f"),
            "Miss (m)": _METRES,
            "Pc": st.column_config.TextColumn(alignment="right"),
        },
        hide_index=True,
        placeholder=MISSING,
    )


def _conjunction(ctx: DashboardContext, data: EventView) -> Conjunction | None:
    """The latest CDM's conjunction; ``None``, with the reason shown, if none."""
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
        column_config={"value": _METRES},
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
        column_config={
            "secondary at TCA [m]": _METRES,
            "relative velocity [m/s]": st.column_config.NumberColumn(format="%.3f"),
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


def _read(ctx: DashboardContext, event_id: str) -> EventView | None:
    """The event read from the store; ``None`` if there is none."""
    return event_view(shared_store(ctx.store_path), event_id, ctx.settings)


def main() -> None:
    """The event page."""
    ctx = page_context()
    event_id, data = event_or_stop("Event", lambda event_id: _read(ctx, event_id))
    st.title(f"Event {event_id}")
    back_links()
    _header(data)
    _to_planner(event_id, data.assessment)
    _trend(ctx, data)
    _cdms(data)
    _latest_geometry(ctx, data)


main()
