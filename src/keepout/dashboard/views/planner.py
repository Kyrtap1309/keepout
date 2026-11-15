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
from typing import TYPE_CHECKING, Final

import numpy as np
import streamlit as st

from keepout._text import ASSUMED_COVARIANCES, minute, pc_text, utc
from keepout._validation import require_open_probability
from keepout.dashboard import plots
from keepout.dashboard.routes import (
    EVENT_PAGE,
    EVENT_PARAM,
    EVENTS_PAGE,
    what_if_key,
)
from keepout.dashboard.text import literal
from keepout.dashboard.widgets import (
    job_status,
    page_context,
    provenance_badge,
    show_error,
    state,
)
from keepout.errors import KeepoutError
from keepout.figures import PROVENANCE_LABELS, RISK_LABELS, grid_figure
from keepout.maneuver import BurnDirection
from keepout.maneuver._words import (
    conflict_line,
    event_line,
    gap_lines,
    refusal,
    rescreen_head,
    return_lines,
)
from keepout.risk import Provenance, covariance_provenance
from keepout.settings import Settings
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
    from keepout.store import Job

# (No docstrings on the names below: Streamlit would show a bare string as
# Markdown.)
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


def _settings(ctx: DashboardContext, event_id: str) -> Settings:
    """The settings the page plans the event with: the operator's, or the
    what-if chosen for the event (its target Pc and combined hard-body
    radius on the operator's settings as they are now)."""
    chosen = state(what_if_key(event_id), Settings)
    if chosen is None:
        return ctx.settings
    return replace(
        ctx.settings,
        target_pc=chosen.target_pc,
        combined_hard_body_radius=chosen.combined_hard_body_radius,
    )


def _read(ctx: DashboardContext, event_id: str, settings: Settings) -> _Planned | None:
    """The event, its plan cached for ``settings`` (``None``: none), and
    its jobs to show: every manoeuvre job of the event still queued or
    running, and the latest one for ``settings`` on its latest CDM if it
    has ended without a plan (failed or cancelled); ``None`` if there is no
    such event. All are read in one transaction
    (:meth:`~keepout.store.Store.plan_detail`).

    Raises
    ------
    StoreError
        If the store cannot be read, or the cached plan cannot be.
    CdmParseError, EventError
        If its stored CDMs no longer parse or make an event.
    ValueError
        If a settings file cannot hold ``settings``.
    """
    from keepout.store import open_store

    with open_store(ctx.store_path, create=False) as store:
        detail = store.plan_detail(event_id, settings)
    if detail is None:
        return None
    outcome = detail.outcome
    shown = list(detail.jobs.active)
    current = detail.jobs.current
    queued = current is not None and current.status.is_active
    if outcome is None and current is not None and not queued:
        # Why the last job for these settings ended.
        shown.append(current)
    return _Planned(detail.event, outcome, tuple(shown), queued)


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
    """The settings the plan is shown for; for a what-if, the operator's
    too and the button back to them (and the form starts from them
    again: its widgets keep what was typed otherwise)."""
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
    """The form of a what-if, starting from the settings the plan is shown
    for (``shown``: the what-if in effect, if any); once submitted, the
    target Pc typed and the combined hard-body radius (``None``: the CDMs'
    radii)."""
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
    """Plan the event with a what-if: keep it for the page (unless it is
    the operator's settings), then queue its plan unless it is cached or
    queued; an error is shown instead."""
    target_pc = _target_pc(target)
    if target_pc is None:
        return
    try:
        chosen = replace(
            ctx.settings, target_pc=target_pc, combined_hard_body_radius=hbr
        )
    except ValueError as exc:
        show_error(exc)
        return
    if chosen == ctx.settings:
        st.session_state.pop(what_if_key(event_id), None)
    else:
        st.session_state[what_if_key(event_id)] = chosen
    if _submit(ctx, event_id, chosen):
        st.rerun()


def _what_if(ctx: DashboardContext, event_id: str, shown: Settings) -> None:
    """The settings the plan is shown for, and the form that plans it for
    another target Pc or combined hard-body radius: cached, or queued as a
    job of its own."""
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
        "Before TCA (h)": option.burn_lead / HOUR if burn else None,
        "Δv (m/s)": option.delta_v if burn else None,
        "Pc": option.pc,
        "Range (m)": option.range_at_tca,
        "Method": option.method,
        "Target": "met" if option.meets_target else "missed",
    }


def _burns(outcome: ManeuverOutcome) -> None:
    """The baseline and the smallest burn each way, and the direction with
    none."""
    st.subheader("Smallest burn each way")
    rows = [_option_row("no burn", outcome.baseline, burn=False)]
    rows += [_option_row(str(b.direction), b.option, burn=True) for b in outcome.burns]
    st.dataframe(
        rows,
        column_config={
            "Before TCA (h)": st.column_config.NumberColumn(format="%.1f"),
            "Pc": st.column_config.NumberColumn(format="%.2e"),
            "Range (m)": st.column_config.NumberColumn(format="%.0f"),
        },
        hide_index=True,
    )
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
            "hours before TCA": (leads.ravel() / HOUR).tolist(),
            "|Δv| [m/s]": np.abs(changes.ravel()).tolist(),
            "Pc": cells.pc.ravel().tolist(),
            "range at TCA [m]": cells.range_at_tca.ravel().tolist(),
        },
        column_config={
            "hours before TCA": st.column_config.NumberColumn(format="%.2f"),
            "Pc": st.column_config.NumberColumn(format="%.2e"),
            "range at TCA [m]": st.column_config.NumberColumn(format="%.1f"),
        },
        hide_index=True,
    )


def _approaches(result: RescreenResult) -> list[tuple[str, NewApproach]]:
    """The approaches the burn brings (``new``) or brings closer
    (``closer``), the highest Pc first."""
    found = [("new", item) for item in result.new_approaches]
    found += [("closer", item) for item in result.closer_approaches]
    return sorted(found, key=lambda pair: pair[1].pc, reverse=True)


def _approach_rows(approaches: Sequence[tuple[str, NewApproach]]) -> None:
    """A data table of the approaches, a row each: its Pc with its
    provenance, and its risk level in words."""
    st.dataframe(
        [
            {
                "approach": kind,
                "secondary": item.approach.secondary_number,
                "TCA (UTC)": minute(item.approach.tca),
                "range [m]": item.approach.range_at_tca,
                "assumed Pc": item.pc,
                "provenance": PROVENANCE_LABELS[Provenance.GP_AGE_MODEL],
                "risk": RISK_LABELS[item.level],
            }
            for kind, item in approaches
        ],
        column_config={
            "range [m]": st.column_config.NumberColumn(format="%.0f"),
            "assumed Pc": st.column_config.NumberColumn(format="%.2e"),
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
    settings = _settings(ctx, event_id) if event_id else ctx.settings
    data: _Planned | None = None
    error: KeepoutError | ValueError | None = None
    if event_id:
        try:
            data = _read(ctx, event_id, settings)
        except (KeepoutError, ValueError) as exc:
            error = exc
    if not event_id or data is None:
        st.title(TITLE)
        if not event_id:
            st.info("Pick an event from the list.")
        elif error is not None:
            show_error(error)
        else:
            st.warning(f"No such event: {literal(event_id)}")
        # A settings file that cannot hold the settings: the event is there.
        _links(event_id if isinstance(error, ValueError) else None)
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
