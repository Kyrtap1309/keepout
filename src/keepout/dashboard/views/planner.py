"""The manoeuvre planner page: an event's plan (query parameter ``event``),
its Pc grid over burn time and Δv, the smallest burn each way and what the
re-screen of each found, as ``keepout maneuver --event`` shows them.

The page plans nothing: the plan of the operator's settings, or of a
what-if target Pc and hard-body radius, is read from the store's cache, and
a button queues a job for ``keepout worker`` when it is not there. A
what-if is cached under its own key, as any settings are.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import streamlit as st

from keepout._text import ASSUMED_COVARIANCES, minute, pc_text, utc
from keepout._validation import require_open_probability
from keepout.dashboard import plots
from keepout.dashboard.text import literal
from keepout.dashboard.widgets import (
    job_status,
    page_context,
    provenance_badge,
    risk_chip,
    show_error,
)
from keepout.errors import KeepoutError
from keepout.figures import RISK_LABELS, Provenance, covariance_provenance, grid_figure
from keepout.maneuver import BurnDirection
from keepout.maneuver._words import (
    conflict_line,
    event_line,
    gap_lines,
    refusal,
    rescreen_head,
    return_lines,
)
from keepout.units import HOUR

if TYPE_CHECKING:
    from collections.abc import Sequence

    from keepout.dashboard.context import DashboardContext
    from keepout.events import ConjunctionEvent
    from keepout.maneuver import (
        ManeuverOption,
        ManeuverOutcome,
        NewApproach,
        RescreenResult,
    )
    from keepout.settings import Settings
    from keepout.store import Job

# (No docstrings on the names below: Streamlit would show a bare string as
# Markdown.)
EVENTS_PAGE: Final = Path(__file__).with_name("events.py")
EVENT_PAGE: Final = Path(__file__).with_name("event.py")
EVENT_PARAM: Final = "event"
TITLE: Final = "Manoeuvre planner"


@dataclass(frozen=True, slots=True)
class _Planned:
    """What the page shows of an event's plan, read from the store in one
    go."""

    event: ConjunctionEvent
    outcome: ManeuverOutcome | None
    jobs: tuple[Job, ...]
    queued: bool
    """Whether a job for the settings is queued or running."""


def _what_if_key(event_id: str) -> str:
    """The session-state key of the event's what-if: its target Pc and
    combined hard-body radius (``None``: the CDMs' radii)."""
    return f"planner-chosen-{event_id}"


def _settings(ctx: DashboardContext, event_id: str) -> Settings:
    """The settings the page plans the event with: the operator's, or the
    what-if chosen for the event."""
    chosen = st.session_state.get(_what_if_key(event_id))
    if not isinstance(chosen, tuple):
        return ctx.settings
    target, hbr = chosen
    return replace(ctx.settings, target_pc=target, combined_hard_body_radius=hbr)


def _read(ctx: DashboardContext, event_id: str, settings: Settings) -> _Planned | None:
    """The event, its plan cached for ``settings`` (``None``: none), and
    its jobs to show: every manoeuvre job of the event still queued or
    running, and the latest one for ``settings`` on its latest CDM if it
    has ended without a plan (failed or cancelled); ``None`` if there is no
    such event.

    Raises
    ------
    StoreError
        If the store cannot be read, or the cached plan cannot be.
    CdmParseError, EventError
        If its stored CDMs no longer parse or make an event.
    ValueError
        If a settings file cannot hold ``settings``.
    """
    from keepout.store import JobQueue, open_store

    with open_store(ctx.store_path, create=False) as store:
        jobs = JobQueue(store).plan_jobs(event_id, settings)
        if jobs is None:
            return None
        event = store.event(event_id)
        outcome = store.maneuver(event_id, settings)
    shown = list(jobs.active)
    current = jobs.current
    queued = current is not None and current.status.is_active
    if outcome is None and current is not None and not queued:
        # Why the last job for these settings ended.
        shown.append(current)
    return _Planned(event, outcome, tuple(shown), queued)


def _hbr_words(hbr: float | None) -> str:
    """A combined hard-body radius, or where it comes from without one."""
    return "the CDMs' radii" if hbr is None else f"{hbr:g} m"


def _settings_words(settings: Settings) -> str:
    """The settings a plan changes, in words."""
    return (
        f"target Pc {pc_text(settings.target_pc)}, combined hard-body radius "
        f"{_hbr_words(settings.combined_hard_body_radius)}"
    )


def _submit(ctx: DashboardContext, event_id: str, settings: Settings) -> bool:
    """Queue the plan of the event with ``settings``, unless it is cached or
    queued already; if it cannot be, show why.

    Returns
    -------
    bool
        Whether the submission went through (the page then runs again).
    """
    from keepout.store import JobQueue, open_store

    try:
        with open_store(ctx.store_path, create=False) as store:
            JobQueue(store).submit(event_id, settings)
    except KeepoutError as exc:
        show_error(exc)
        return False
    except ValueError as exc:
        st.error(literal(str(exc)))
        return False
    return True


def _target_pc(text: str) -> float | None:
    """The target Pc typed, as ``keepout maneuver --target-pc`` reads it: a
    number in (0, 1); ``None``, with an error shown, if it is not."""
    try:
        value = float(text.strip())
        require_open_probability(value, "the target Pc")
    except ValueError:
        st.error(
            "The target Pc must be a number in (0, 1), got "
            f"{literal(repr(text.strip()))}."
        )
        return None
    return value


def _what_if(ctx: DashboardContext, event_id: str, shown: Settings) -> None:
    """The settings the plan is shown for, and the form that plans it for
    another target Pc or combined hard-body radius: cached, or queued as a
    job of its own."""
    if shown == ctx.settings:
        st.markdown(f"Operator's settings: {_settings_words(shown)}.")
    else:
        st.markdown(f"**What-if**: {_settings_words(shown)}.")
        st.caption(f"The operator's settings: {_settings_words(ctx.settings)}.")
        if st.button("Back to the operator's settings", key=f"planner-own-{event_id}"):
            st.session_state.pop(_what_if_key(event_id), None)
            st.rerun()
    with st.expander("What if"), st.form(f"planner-form-{event_id}"):
        target = st.text_input(
            "Target Pc",
            value=repr(ctx.settings.target_pc),
            help="A probability in (0, 1), e.g. 1e-6.",
            key=f"planner-target-{event_id}",
        )
        hbr = st.number_input(
            "Combined hard-body radius [m]",
            min_value=0.0,
            value=ctx.settings.combined_hard_body_radius,
            placeholder=_hbr_words(None),
            key=f"planner-hbr-{event_id}",
        )
        submitted = st.form_submit_button(
            "Plan with these", key=f"planner-what-if-{event_id}"
        )
    if not submitted:
        return
    target_pc = _target_pc(target)
    if target_pc is None:
        return
    try:
        chosen = replace(
            ctx.settings,
            target_pc=target_pc,
            combined_hard_body_radius=None if hbr is None else float(hbr),
        )
    except ValueError as exc:
        st.error(literal(str(exc)))
        return
    if chosen == ctx.settings:
        st.session_state.pop(_what_if_key(event_id), None)
    else:
        st.session_state[_what_if_key(event_id)] = (
            chosen.target_pc,
            chosen.combined_hard_body_radius,
        )
    if _submit(ctx, event_id, chosen):
        st.rerun()


def _jobs(ctx: DashboardContext, jobs: Sequence[Job]) -> None:
    """The event's manoeuvre jobs: the status of each with its settings
    (with none, only a refused cancel of a job no longer shown)."""
    if jobs:
        st.subheader("Jobs")
    job_status(
        ctx.store_path,
        [job.id for job in jobs],
        labels={job.id: f"{_settings_words(job.settings)}." for job in jobs},
    )


def _no_plan(
    ctx: DashboardContext, event_id: str, settings: Settings, *, queued: bool
) -> None:
    """Why there is no plan, and the button that queues one."""
    if queued:
        st.info("The plan is shown once a worker has made it (keepout worker).")
        return
    st.info(
        "No plan for these settings on the event's latest CDM: queue one for "
        "keepout worker, which takes tens of seconds on the burn grid."
    )
    if st.button("Plan manoeuvre", key=f"planner-plan-{event_id}") and _submit(
        ctx, event_id, settings
    ):
        st.rerun()


def _option_row(name: str, option: ManeuverOption, *, burn: bool) -> dict[str, object]:
    """A row of the burns table, as ``keepout maneuver`` prints it."""
    return {
        "Option": name,
        "Before TCA (h)": round(option.burn_lead / HOUR, 1) if burn else None,
        "Δv (m/s)": option.delta_v if burn else None,
        "Pc": option.pc,
        "Range (m)": round(option.range_at_tca),
        "Method": option.method,
        "Target": "met" if option.meets_target else "missed",
    }


def _burns(outcome: ManeuverOutcome) -> None:
    """The baseline and the smallest burn each way, and the direction with
    none."""
    st.subheader("Smallest burn each way")
    rows = [_option_row("no burn", outcome.baseline, burn=False)]
    rows += [_option_row(str(b.direction), b.option, burn=True) for b in outcome.burns]
    st.dataframe(rows, hide_index=True)
    for line in gap_lines(outcome):
        st.info(line)


def _grid(outcome: ManeuverOutcome) -> None:
    """The heatmap of the Pc grid and its cells."""
    if outcome.grid is None:
        return
    st.subheader("Pc over burn time and Δv")
    figure = grid_figure(outcome)
    st.plotly_chart(plots.grid(figure))
    cells = outcome.grid
    leads, changes = np.meshgrid(cells.burn_leads, cells.delta_v, indexing="ij")
    st.dataframe(
        {
            "direction": [str(BurnDirection.of(dv)) for dv in changes.ravel()],
            "hours before TCA": (leads.ravel() / HOUR).round(2).tolist(),
            "|Δv| [m/s]": np.abs(changes.ravel()).tolist(),
            "Pc": cells.pc.ravel().tolist(),
            "range at TCA [m]": cells.range_at_tca.ravel().round(1).tolist(),
        },
        hide_index=True,
    )


APPROACH_COLUMNS: Final = (
    ("Approach", 2),
    ("TCA (UTC)", 3),
    ("Range [m]", 2),
    ("Assumed Pc", 2),
    ("Provenance", 3),
    ("Risk", 2),
)


def _approaches(result: RescreenResult) -> list[tuple[str, NewApproach]]:
    """The approaches the burn brings (``new``) or brings closer
    (``closer``), the highest Pc first."""
    found = [("new", item) for item in result.new_approaches]
    found += [("closer", item) for item in result.closer_approaches]
    return sorted(found, key=lambda pair: pair[1].pc, reverse=True)


def _approach_rows(approaches: Sequence[tuple[str, NewApproach]]) -> None:
    """A row per approach, its risk as a chip (colour and words), its Pc
    next to its provenance; then the same as a data table."""
    header = st.columns([width for _, width in APPROACH_COLUMNS])
    for cell, (name, _) in zip(header, APPROACH_COLUMNS, strict=True):
        cell.markdown(f"**{name}**")
    for kind, item in approaches:
        approach = item.approach
        cells = st.columns(
            [width for _, width in APPROACH_COLUMNS], vertical_alignment="center"
        )
        what, tca, distance, pc, origin, risk = cells
        what.markdown(f"{kind} · {literal(str(approach.secondary_number))}")
        tca.markdown(minute(approach.tca))
        distance.markdown(f"{approach.range_at_tca:.0f}")
        pc.markdown(pc_text(item.pc))
        with origin:
            provenance_badge(Provenance.GP_AGE_MODEL)
        with risk:
            risk_chip(item.level, None)
    st.dataframe(
        [
            {
                "approach": kind,
                "secondary": item.approach.secondary_number,
                "TCA (UTC)": minute(item.approach.tca),
                "range [m]": round(item.approach.range_at_tca),
                "assumed Pc": item.pc,
                "provenance": Provenance.GP_AGE_MODEL.value,
                "risk": RISK_LABELS[item.level],
            }
            for kind, item in approaches
        ],
        hide_index=True,
    )


def _rescreens(outcome: ManeuverOutcome) -> None:
    """What the re-screen of each burn found, or that there was none."""
    st.subheader("Re-screen")
    label = outcome.catalog_label
    if label is None:
        st.info(
            "Not re-screened: no catalog stored with the event's latest CDM, "
            "so nothing is known of the approaches a burn brings."
        )
        return
    secondary = outcome.event.latest.object2.object_id
    for burn in outcome.burns:
        result = burn.rescreen
        if result is None:
            continue
        st.markdown(rescreen_head(burn.direction, result, literal(label)))
        st.markdown(event_line(result, literal(secondary)))
        if result.conflict:
            st.error(conflict_line(result))
        else:
            st.markdown(conflict_line(result))
        for line in return_lines(result):
            st.warning(line)
        approaches = _approaches(result)
        if approaches:
            _approach_rows(approaches)
        if approaches or result.event_returns:
            st.caption(ASSUMED_COVARIANCES)


def _plan(outcome: ManeuverOutcome) -> None:
    """The plan: its verdict, the burns, the grid and the re-screens."""
    latest = outcome.event.latest
    st.markdown(
        f"Planned from the latest CDM, {literal(latest.message_id)}: TCA "
        f"{utc(latest.tca)} UTC, target Pc "
        f"{pc_text(outcome.target_pc)}, baseline Pc "
        f"{pc_text(outcome.baseline.pc)} ({outcome.baseline.method})."
    )
    provenance_badge(covariance_provenance(latest))
    if outcome.verdict.recommendable:
        st.success("A burn can be recommended: it meets the target Pc.")
    else:
        st.warning(f"{refusal(outcome)}.")
    _burns(outcome)
    _grid(outcome)
    for note in outcome.notes:
        st.caption(f"Note: {note}")
    _rescreens(outcome)


def _links(event_id: str | None) -> None:
    """The ways back: the events list, and the event's page."""
    st.page_link(EVENTS_PAGE, label="Events")
    if event_id:
        st.page_link(EVENT_PAGE, label="Event", query_params={EVENT_PARAM: event_id})


def main() -> None:
    """The manoeuvre planner page."""
    ctx = page_context()
    event_id = st.query_params.get(EVENT_PARAM)
    if not event_id:
        st.title(TITLE)
        st.info("Pick an event from the list.")
        _links(None)
        return
    settings = _settings(ctx, event_id)
    try:
        data = _read(ctx, event_id, settings)
    except KeepoutError as exc:
        st.title(TITLE)
        show_error(exc)
        _links(None)
        return
    except ValueError as exc:
        st.title(TITLE)
        st.error(literal(str(exc)))
        _links(event_id)
        return
    if data is None:
        st.title(TITLE)
        st.warning(f"No such event: {literal(event_id)}")
        _links(None)
        return
    st.title(f"{TITLE}: event {event_id}")
    event = data.event
    st.markdown(
        f"**{literal(event.primary_id)} / {literal(event.secondary_id)}** · TCA "
        f"{utc(event.latest.tca)} UTC"
    )
    _links(event_id)
    _what_if(ctx, event_id, settings)
    _jobs(ctx, data.jobs)
    if data.outcome is None:
        _no_plan(ctx, event_id, settings, queued=data.queued)
        return
    _plan(data.outcome)


main()
