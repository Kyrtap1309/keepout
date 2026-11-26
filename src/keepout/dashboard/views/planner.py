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
from functools import partial
from typing import TYPE_CHECKING, Final

import numpy as np
import streamlit as st

from keepout._text import ASSUMED_COVARIANCES, minute, pc_text, utc
from keepout._validation import require_open_probability
from keepout.dashboard import plots
from keepout.dashboard.routes import WhatIf, what_if_key
from keepout.dashboard.text import (
    MISSING,
    literal,
    merged_notes,
    risk_text,
)
from keepout.dashboard.widgets import (
    back_links,
    event_or_stop,
    job_status,
    page_context,
    provenance_badge,
    shared_store,
    show_error,
    state,
)
from keepout.errors import KeepoutError
from keepout.figures import grid_figure
from keepout.maneuver import BurnDirection
from keepout.maneuver.wording import (
    NO_BURN_NEEDED,
    ApproachKind,
    burn_rows,
    conflict_line,
    event_line,
    gap_lines,
    ranked_approaches,
    refusal,
    rescreen_head,
    return_lines,
)
from keepout.risk import covariance_provenance
from keepout.units import HOUR

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import plotly.graph_objects as go

    from keepout.dashboard.context import DashboardContext
    from keepout.events import ConjunctionEvent
    from keepout.maneuver import ManeuverGrid, ManeuverOutcome, NewApproach
    from keepout.settings import Settings
    from keepout.store import Job

TITLE: Final = "Manoeuvre planner"

_PC = st.column_config.TextColumn("Pc", alignment="right")


@dataclass(frozen=True, slots=True)
class _Planned:
    """What the page shows of an event's plan, read from the store in one
    go."""

    event: ConjunctionEvent
    settings: Settings
    message_id: str
    key: str
    """The plan's cache key: its settings and catalog."""
    outcome: ManeuverOutcome | None
    jobs: tuple[Job, ...]
    queued: bool
    """Whether a job for the settings is queued or running."""


def _with(settings: Settings, what_if: WhatIf | None) -> Settings:
    """The operator's settings, changed by a what-if if any (ValueError if bad)."""
    if what_if is None:
        return settings
    return replace(
        settings, target_pc=what_if.target_pc, combined_hard_body_radius=what_if.hbr
    )


def _read(ctx: DashboardContext, event_id: str) -> _Planned | None:
    """The event, its plan for the page's settings and the jobs to show, in one read."""
    settings = _with(ctx.settings, state(what_if_key(event_id), WhatIf))
    detail = shared_store(ctx.store_path).plan_detail(event_id, settings)
    if detail is None:
        return None
    outcome = detail.outcome
    shown = list(detail.jobs.active)
    current = detail.jobs.current
    queued = current is not None and current.status.is_active
    if outcome is None and current is not None and not queued:
        # Why the last job for these settings ended.
        shown.append(current)
    return _Planned(
        detail.event,
        settings,
        detail.message_id,
        detail.key,
        outcome,
        tuple(shown),
        queued,
    )


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
    """Queue the plan unless cached or queued; whether it went through."""
    from keepout.store import JobQueue

    try:
        JobQueue(shared_store(ctx.store_path)).submit(event_id, settings)
    except (KeepoutError, ValueError) as exc:
        show_error(exc)
        return False
    return True


def _target_pc(text: str) -> float | None:
    """The target Pc typed, as ``keepout maneuver --target-pc`` reads it: a
    number in (0, 1); ``None``, with an error shown, if it is not."""
    typed = text.strip()
    try:
        value = float(typed)
        require_open_probability(value, "the target Pc")
    except ValueError:
        st.error(
            f"The target Pc must be a number in (0, 1), got {literal(repr(typed))}."
        )
        return None
    return value


def _form_keys(event_id: str) -> tuple[str, str]:
    """The keys of the what-if form's target Pc and hard-body radius."""
    return f"planner-target-{event_id}", f"planner-hbr-{event_id}"


def _shown_settings(ctx: DashboardContext, event_id: str, shown: Settings) -> None:
    """The settings shown; for a what-if, the operator's and the way back."""
    if shown == ctx.settings:
        st.markdown(f"Operator's settings: {_settings_words(shown)}.")
        return
    st.markdown(f"**What-if**: {_settings_words(shown)}.")
    st.caption(f"The operator's settings: {_settings_words(ctx.settings)}.")
    if st.button("Back to the operator's settings", key=f"planner-own-{event_id}"):
        for key in (what_if_key(event_id), *_form_keys(event_id)):
            st.session_state.pop(key, None)
        st.rerun()


def _what_if_form(event_id: str, shown: Settings) -> tuple[str, float | None] | None:
    """The what-if form, from the settings shown; what was typed once submitted."""
    target_key, hbr_key = _form_keys(event_id)
    with st.expander("What if"), st.form(f"planner-form-{event_id}"):
        target = st.text_input(
            "Target Pc",
            value=repr(shown.target_pc),
            help="A probability in (0, 1), e.g. 1e-6.",
            key=target_key,
        )
        hbr = st.number_input(
            "Combined hard-body radius [m]",
            min_value=0.0,
            value=shown.combined_hard_body_radius,
            placeholder=_hbr_words(None),
            key=hbr_key,
        )
        submitted = st.form_submit_button(
            "Plan with these", key=f"planner-what-if-{event_id}"
        )
    if not submitted:
        return None
    return target, None if hbr is None else float(hbr)


def _apply_what_if(
    ctx: DashboardContext, event_id: str, target: str, hbr: float | None
) -> None:
    """Keep a what-if for the page and queue its plan; or show the error."""
    target_pc = _target_pc(target)
    if target_pc is None:
        return
    what_if = WhatIf(target_pc, hbr)
    try:
        chosen = _with(ctx.settings, what_if)
    except ValueError as exc:
        show_error(exc)
        return
    if chosen == ctx.settings:
        st.session_state.pop(what_if_key(event_id), None)
    else:
        st.session_state[what_if_key(event_id)] = what_if
    if _submit(ctx, event_id, chosen):
        st.rerun()


def _what_if(ctx: DashboardContext, event_id: str, shown: Settings) -> None:
    """The settings shown, and the form planning for another target or radius."""
    _shown_settings(ctx, event_id, shown)
    typed = _what_if_form(event_id, shown)
    if typed is not None:
        _apply_what_if(ctx, event_id, *typed)


def _jobs(ctx: DashboardContext, jobs: Sequence[Job]) -> None:
    """The event's manoeuvre jobs: the status of each with its settings."""
    if not jobs:
        return
    st.subheader("Jobs")
    job_status(
        ctx.store_path,
        jobs,
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


def _burns(outcome: ManeuverOutcome) -> None:
    """The baseline and the smallest burn each way, and the direction with
    none."""
    st.subheader("Smallest burn each way")
    st.dataframe(
        [{**row, "Pc": pc_text(row["Pc"])} for row in burn_rows(outcome)],
        column_config={
            "Before TCA (h)": st.column_config.NumberColumn(format="%.1f"),
            "Δv (m/s)": st.column_config.NumberColumn(format="%+.4f"),
            "Pc": _PC,
            "Range (m)": st.column_config.NumberColumn(format="%.0f"),
        },
        hide_index=True,
        placeholder=MISSING,
    )
    for line in gap_lines(outcome):
        st.info(line)


_Cells = dict[str, list[str] | list[float] | list[float | None] | list[str | None]]


@st.cache_data(show_spinner=False, max_entries=16)
def _grid_view(
    store_path: Path,  # noqa: ARG001 (the key)
    event_id: str,  # noqa: ARG001
    message_id: str,  # noqa: ARG001
    key: str,  # noqa: ARG001
    _outcome: ManeuverOutcome,
    _cells: ManeuverGrid,
) -> tuple[go.Figure, _Cells]:
    """The heatmap and its cells as a table, made once per cached plan."""
    cells = _cells
    leads, changes = np.meshgrid(cells.burn_leads, cells.delta_v, indexing="ij")
    pc = cells.pc.ravel()
    ranges = cells.range_at_tca.ravel()
    table: _Cells = {
        "direction": [str(BurnDirection.of(dv)) for dv in changes.ravel()],
        "hours before TCA": (leads.ravel() / HOUR).tolist(),
        "|Δv| [m/s]": np.abs(changes.ravel()).tolist(),
        "Pc": [pc_text(float(v)) if np.isfinite(v) else None for v in pc],
        "range at TCA [m]": [float(v) if np.isfinite(v) else None for v in ranges],
    }
    return plots.grid(grid_figure(_outcome)), table


def _grid(planned: _Planned, outcome: ManeuverOutcome, store_path: Path) -> None:
    """The heatmap of the Pc grid, and its cells in a collapsed table."""
    if outcome.grid is None:
        return
    st.subheader("Pc over burn time and Δv")
    figure, cells = _grid_view(
        store_path,
        planned.event.event_id,
        planned.message_id,
        planned.key,
        outcome,
        outcome.grid,
    )
    st.plotly_chart(figure)
    with st.expander(f"Grid cells ({outcome.grid.pc.size})"):
        st.dataframe(
            cells,
            column_config={
                "hours before TCA": st.column_config.NumberColumn(format="%.2f"),
                "|Δv| [m/s]": st.column_config.NumberColumn(format="%.4f"),
                "Pc": _PC,
                "range at TCA [m]": st.column_config.NumberColumn(format="%.1f"),
            },
            hide_index=True,
            placeholder=MISSING,
        )


def _approach_rows(approaches: Sequence[tuple[ApproachKind, NewApproach]]) -> None:
    """A data table of the approaches, a row each: its assumed Pc, and its
    risk level in words."""
    st.dataframe(
        [
            {
                "approach": kind,
                "secondary": item.approach.secondary_number,
                "TCA (UTC)": minute(item.approach.tca),
                "range [m]": item.approach.range_at_tca,
                "assumed Pc": pc_text(item.pc),
                "risk": risk_text(item.level),
            }
            for kind, item in approaches
        ],
        column_config={
            "secondary": st.column_config.NumberColumn(format="%d"),
            "range [m]": st.column_config.NumberColumn(format="%.0f"),
            "assumed Pc": st.column_config.TextColumn(alignment="right"),
        },
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
        tolerance = outcome.event.tca_tolerance
        st.markdown(event_line(result, literal(secondary), tolerance))
        if result.conflict:
            st.error(conflict_line(result))
        else:
            st.markdown(conflict_line(result))
        for line in return_lines(result):
            st.warning(line)
        approaches = ranked_approaches(result)
        if approaches:
            _approach_rows(approaches)
        if approaches or result.event_returns:
            st.caption(ASSUMED_COVARIANCES)


def _notes(outcome: ManeuverOutcome) -> None:
    """The plan's notes, each once, collapsed."""
    notes = merged_notes(outcome.notes)
    if notes:
        with st.expander(f"Notes ({len(notes)})"):
            st.markdown("\n".join(f"- {literal(note)}" for note in notes))


def _plan(planned: _Planned, outcome: ManeuverOutcome, store_path: Path) -> None:
    """The plan: its verdict, the burns, the grid and the re-screens."""
    latest = outcome.event.latest
    st.markdown(
        f"Planned from the latest CDM, {literal(latest.message_id)}: TCA "
        f"{utc(latest.tca)} UTC, target Pc "
        f"{pc_text(outcome.target_pc)}, baseline Pc "
        f"{pc_text(outcome.baseline.pc)} ({outcome.baseline.method})."
    )
    provenance_badge(covariance_provenance(latest))
    verdict = outcome.verdict
    if verdict.no_burn_needed:
        st.success(NO_BURN_NEEDED)
    elif verdict.recommendable:
        st.success("A burn can be recommended: it meets the target Pc.")
    else:
        st.warning(f"{refusal(outcome)}.")
    _burns(outcome)
    _grid(planned, outcome, store_path)
    _notes(outcome)
    _rescreens(outcome)


def main() -> None:
    """The manoeuvre planner page."""
    ctx = page_context()
    event_id, data = event_or_stop(TITLE, partial(_read, ctx))
    st.title(f"{TITLE}: event {event_id}")
    back_links(event_id)
    event = data.event
    st.markdown(
        f"**{literal(event.primary_id)} / {literal(event.secondary_id)}** · TCA "
        f"{utc(event.latest.tca)} UTC"
    )
    _what_if(ctx, event_id, data.settings)
    _jobs(ctx, data.jobs)
    if data.outcome is None:
        _no_plan(ctx, event_id, data.settings, queued=data.queued)
        return
    _plan(data, data.outcome, ctx.store_path)


main()
