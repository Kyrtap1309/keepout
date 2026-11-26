"""What the dashboard's pages share: the context, the session's state and the
store, the time the data are seen at, the ways between pages, risk chips,
provenance badges, errors and job status.

Risk is shown as colour and words together: a chip filled with the level's
colour (:data:`~keepout.figures.RISK_COLOURS`) carries its label and the
action's, in black or white, whichever contrasts more with the fill, and
has a grey outline. The fill, not the page, is behind the text, so the chip
reads the same on Streamlit's light and dark themes (the dark red of
:attr:`~keepout.risk.RiskLevel.RED` gets white text). A table gives the
level a coloured mark instead (:func:`~keepout.dashboard.text.risk_text`).
"""

from __future__ import annotations

import html
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NoReturn, TypeVar, cast

import streamlit as st

from keepout._text import utc
from keepout.dashboard.context import DashboardContext, open_existing
from keepout.dashboard.routes import (
    CONTEXT_KEY,
    EVENT_PAGE,
    EVENT_PARAM,
    EVENTS_PAGE,
    cancel_job_key,
)
from keepout.dashboard.text import NOT_COMPUTED, british, literal, unrated
from keepout.errors import KeepoutError
from keepout.figures import (
    PROVENANCE_HELP,
    PROVENANCE_LABELS,
    RISK_COLOURS,
)
from keepout.risk import Provenance

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from keepout.risk import Action, Assessment, RiskLevel
    from keepout.store import Job, Store

__all__ = [
    "JOB_REFRESH_SECONDS",
    "as_of_header",
    "assessment_chip",
    "back_links",
    "event_or_stop",
    "ink",
    "job_status",
    "page_context",
    "provenance_badge",
    "read_jobs",
    "risk_chip",
    "shared_store",
    "show_error",
    "state",
]

JOB_REFRESH_SECONDS: Final = 2
"""How often [s] the status of a queued or running job is read again."""

_T = TypeVar("_T")

_BadgeColour = Literal["blue", "orange", "violet", "gray"]

_PROVENANCE_COLOURS: Final[Mapping[Provenance, _BadgeColour]] = MappingProxyType(
    {
        Provenance.CDM_COVARIANCE: "blue",
        Provenance.GP_AGE_MODEL: "orange",
        Provenance.ASSUMED: "orange",
        Provenance.MAX_PC: "violet",
    }
)
"""The colour of each provenance badge; the badge names it in words."""

_CHIP_STYLE: Final = (
    "display:inline-block;padding:0.1rem 0.6rem;border-radius:999px;"
    "border:1px solid rgba(128,128,128,0.7);font-weight:600;font-size:0.9rem;"
)
"""The pill of a risk chip; its fill and text colour are added."""


def state(key: str, kind: type[_T]) -> _T | None:
    """What ``st.session_state`` keeps under ``key``, if it is a ``kind``;
    else ``None``."""
    value = st.session_state.get(key)
    return value if isinstance(value, kind) else None


def page_context() -> DashboardContext:
    """The context the app put in ``st.session_state`` before it ran the
    page."""
    return cast("DashboardContext", st.session_state[CONTEXT_KEY])


def _close(store: Store) -> None:
    """Close a store the cache lets go of."""
    store.close()


@st.cache_resource(show_spinner=False, on_release=_close)
def shared_store(store_path: Path) -> Store:
    """The store, opened once per file for every page and session (its
    schema checked then): its engine is safe to share between threads. Do
    not close it.

    Raises
    ------
    StoreError
        If it cannot be opened; it is opened again on the next call.
    """
    return open_existing(store_path)


def as_of_header(ctx: DashboardContext) -> None:
    """The time the data are seen at, to the second in UTC, marked
    ``(fixed)`` when given rather than read from the clock."""
    stamp = utc(ctx.as_of)
    fixed = " (fixed)" if ctx.as_of_fixed else ""
    st.markdown(f"Data as of {stamp} UTC{fixed}")


def back_links(event_id: str | None = None) -> None:
    """The ways back: to the events list, and to the event's page if
    ``event_id`` is given."""
    links = st.container(horizontal=True)
    links.page_link(EVENTS_PAGE, label="Events", icon=":material/arrow_back:")
    if event_id:
        links.page_link(
            EVENT_PAGE,
            label=f"Event {event_id}",
            icon=":material/arrow_back:",
            query_params={EVENT_PARAM: event_id},
        )


def event_or_stop(title: str, read: Callable[[str], _T | None]) -> tuple[str, _T]:
    """The event the address names (query parameter ``event``) and what
    ``read`` reads of it; else the page says why, under ``title``, with the
    ways back, and stops (``st.stop``): no event named, an error reading it,
    or no such event.

    A ``ValueError`` that is not a keepout error (a settings file that cannot
    hold the settings) is of an event that is there: the way back to it is
    given too.
    """
    event_id = st.query_params.get(EVENT_PARAM)
    error: Exception | None = None
    found: _T | None = None
    if event_id:
        try:
            found = read(event_id)
        except (KeepoutError, ValueError) as exc:
            error = exc
    if not event_id or found is None:
        _no_event(title, event_id, error)
    return event_id, found


def _no_event(title: str, event_id: str | None, error: Exception | None) -> NoReturn:
    """Why the page has no event, the ways back, and a stopped page."""
    st.title(title)
    if not event_id:
        st.info("Pick an event from the list.")
    elif error is not None:
        show_error(error)
    else:
        st.warning(f"No such event: {literal(event_id)}")
    there = isinstance(error, ValueError) and not isinstance(error, KeepoutError)
    back_links(event_id if there else None)
    st.stop()


def provenance_badge(p: Provenance) -> None:
    """A badge naming what a Pc rests on, with its meaning on hover."""
    st.badge(
        PROVENANCE_LABELS[p], color=_PROVENANCE_COLOURS[p], help=PROVENANCE_HELP[p]
    )


def _luminance(colour: str) -> float:
    """The relative luminance (WCAG 2) of a ``#rrggbb`` colour."""
    channels = [int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5)]
    linear = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def ink(fill: str) -> str:
    """Black or white, whichever contrasts more with ``fill`` (``#rrggbb``)."""
    lum = _luminance(fill)
    on_black = (lum + 0.05) / 0.05
    on_white = 1.05 / (lum + 0.05)
    return "#000000" if on_black >= on_white else "#ffffff"


def risk_chip(
    level: RiskLevel | None, action: Action | None, *, missing: str = NOT_COMPUTED
) -> None:
    """An event's risk as a chip: its level's colour with the level and the
    action in words; ``None`` is an outlined chip in the theme's text colour
    saying why there is no level, ``missing`` (default: no Pc for the
    settings, "not computed")."""
    if level is None:
        style = f"{_CHIP_STYLE}border-style:dashed;color:inherit;"
        text = missing
    else:
        fill = RISK_COLOURS[level]
        style = f"{_CHIP_STYLE}background:{fill};color:{ink(fill)};"
        text = level.key
        if action is not None:
            text += f" · {british(action.label)}"
    st.markdown(
        f'<span style="{style}">{html.escape(text)}</span>', unsafe_allow_html=True
    )


def assessment_chip(assessment: Assessment | None) -> None:
    """An event's risk as a chip (:func:`risk_chip`): its level and action;
    without a level, why (:func:`~keepout.dashboard.text.unrated`); without
    an assessment, not computed."""
    if assessment is None:
        risk_chip(None, None)
    elif assessment.level is None:
        risk_chip(None, None, missing=unrated(assessment))
    else:
        risk_chip(assessment.level, assessment.decision.action)


def show_error(error: Exception) -> None:
    """An error as the page shows it: its message, verbatim
    (:func:`~keepout.dashboard.text.literal`), no traceback."""
    st.error(literal(str(error)))


def read_jobs(store_path: Path, job_ids: Sequence[int]) -> list[Job]:
    """The jobs ``job_ids`` as they are now, read in one transaction; one
    no longer in the store (purged) is left out.

    Raises
    ------
    StoreError
        If the store cannot be opened or read.
    """
    from keepout.store import JobQueue

    return JobQueue(shared_store(store_path)).jobs_by_id(job_ids, missing_ok=True)


def _cancel(store_path: Path, job_id: int) -> None:
    """Cancel a job; if it cannot be, say why in a toast, which outlives
    the rerun that follows."""
    from keepout.store import JobQueue

    try:
        JobQueue(shared_store(store_path)).cancel(job_id)
    except KeepoutError as exc:
        st.toast(literal(str(exc)), icon=":material/error:")


def _job_rows(store_path: Path, jobs: Sequence[Job], labels: Mapping[int, str]) -> None:
    """A line per job: kind, status, label, Cancel while active, error."""
    for job in jobs:
        text, button = st.columns([5, 1], vertical_alignment="center")
        kind = british(job.kind.value)
        text.markdown(f"Job {job.id} ({kind}): **{job.status.value}**")
        if job.id in labels:
            text.caption(labels[job.id])
        if job.status.is_active and button.button("Cancel", key=cancel_job_key(job.id)):
            _cancel(store_path, job.id)
            st.rerun()
        if job.error:
            st.error(literal(job.error))


@st.fragment(run_every=JOB_REFRESH_SECONDS)
def _live_jobs(
    store_path: Path, job_ids: Sequence[int], labels: Mapping[int, str]
) -> None:
    """The jobs, read on every run of the fragment until none is active."""
    try:
        jobs = read_jobs(store_path, job_ids)
    except KeepoutError as exc:
        show_error(exc)
        return
    _job_rows(store_path, jobs, labels)
    if not any(job.status.is_active for job in jobs):
        st.rerun()


def job_status(
    store_path: Path,
    jobs: Sequence[Job],
    *,
    labels: Mapping[int, str] | None = None,
) -> bool:
    """The status of jobs the page has read, with a Cancel button per queued
    or running one.

    Ended jobs are shown as given. While one is queued or running, they are
    read again every :data:`JOB_REFRESH_SECONDS` (an ``st.fragment``, which
    reads them on its first run too) until all have ended; a job no longer
    in the store (purged) is left out then. A failed job's error is shown
    with ``st.error``, a store error while polling with :func:`show_error`
    (until a later read succeeds) and a refused cancel's error in a toast.

    Parameters
    ----------
    store_path
        The store's file.
    jobs
        The jobs to show, in order.
    labels
        A line under a job's status, by job id (e.g. its settings).

    Returns
    -------
    bool
        Whether one of ``jobs`` is queued or running.
    """
    named = dict(labels or {})
    if any(job.status.is_active for job in jobs):
        _live_jobs(store_path, tuple(job.id for job in jobs), named)
        return True
    _job_rows(store_path, jobs, named)
    return False
